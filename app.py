from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json
from datetime import datetime
import uuid
from dotenv import load_dotenv
load_dotenv()

# Import your SQL agent functions
from sql_agent import (
    get_or_create_agent,
    get_user_priorities,
    update_user_priority,
    init_preferences_db,
    db,
    _global_agent
)

# Initialize FastAPI app
app = FastAPI(
    title="SQL Agent API",
    description="AI-powered SQL agent with preference learning",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize DB on startup
@app.on_event("startup")
async def startup_event():
    init_preferences_db()
    print("✅ FastAPI server started")
    print("✅ Database initialized")

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = "default_user"

class QueryResponse(BaseModel):
    user_id: str
    query: str
    summary: str
    sql_query: Optional[str] = None
    raw_result: Optional[str] = None
    timestamp: str
    model: str = "Gemini 2.0 Flash"

class PreferenceRequest(BaseModel):
    user_id: str
    priority_key: str
    priority_value: str
    context: Optional[str] = None
    feedback_text: Optional[str] = None
    source_query: Optional[str] = None

class PreferenceResponse(BaseModel):
    message: str
    user_id: str
    preferences: Dict[str, str]

class ConversationClearRequest(BaseModel):
    user_id: Optional[str] = None

# Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "SQL Agent API",
        "version": "1.0.0",
        "endpoints": {
            "POST /query": "Execute SQL query",
            "GET /preferences/{user_id}": "Get user preferences",
            "POST /preferences": "Update user preferences",
            "GET /tables": "List database tables",
            "POST /clear": "Clear conversation memory",
            "DELETE /preferences/{user_id}": "Delete user preferences",
            "WebSocket /ws/{user_id}": "Real-time query streaming"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/query", response_model=QueryResponse)
async def execute_query(request: QueryRequest):
    """
    Execute a natural language query using the SQL agent
    
    - **query**: Natural language question
    - **user_id**: User identifier for session management
    """
    try:
        agent = get_or_create_agent()
        config = {"configurable": {"thread_id": request.user_id}}
        enhanced_input = f"[User ID: {request.user_id}]\n{request.query}"
        
        # Stream the agent response
        events = agent.stream(
            {"messages": [("user", enhanced_input)]},
            config=config,
            stream_mode="values",
        )
        
        # Collect response data
        summary = ""
        sql_query = None
        raw_result = None
        
        for event in events:
            if "messages" in event:
                msg = event["messages"][-1]
                
                # Extract SQL query from tool calls
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        if tool_call.get('name') == 'sql_db_query':
                            sql_query = tool_call.get('args', {}).get('query', '')
                
                # Extract content
                if hasattr(msg, 'content') and isinstance(msg.content, str):
                    if hasattr(msg, 'name') and msg.name == 'sql_db_query':
                        raw_result = msg.content
                    elif not hasattr(msg, 'name') or not msg.name:
                        summary = msg.content
        
        return QueryResponse(
            user_id=request.user_id,
            query=request.query,
            summary=summary or "Query processed successfully",
            sql_query=sql_query,
            raw_result=raw_result,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")

@app.get("/preferences/{user_id}")
async def get_preferences(user_id: str):
    """
    Get saved preferences for a specific user
    
    - **user_id**: User identifier
    """
    try:
        prefs_json = get_user_priorities.invoke({"user_id": user_id})
        preferences = json.loads(prefs_json)
        
        return {
            "user_id": user_id,
            "preferences": preferences,
            "count": len(preferences)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve preferences: {str(e)}")

@app.post("/preferences", response_model=PreferenceResponse)
async def save_preference(request: PreferenceRequest):
    """
    Save or update a user preference
    
    - **user_id**: User identifier
    - **priority_key**: Preference key (e.g., "cost", "coverage")
    - **priority_value**: Preference value (e.g., "low", "high")
    - **context**: Optional context information
    - **feedback_text**: Optional feedback text
    - **source_query**: Optional source query
    """
    try:
        result = update_user_priority.invoke({
            "user_id": request.user_id,
            "priority_key": request.priority_key,
            "priority_value": request.priority_value,
            "context": request.context,
            "feedback_text": request.feedback_text,
            "source_query": request.source_query
        })
        
        # Get updated preferences
        prefs_json = get_user_priorities.invoke({"user_id": request.user_id})
        preferences = json.loads(prefs_json)
        
        return PreferenceResponse(
            message=result,
            user_id=request.user_id,
            preferences=preferences
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save preference: {str(e)}")

@app.delete("/preferences/{user_id}")
async def delete_preferences(user_id: str):
    """
    Delete all preferences for a specific user
    
    - **user_id**: User identifier
    """
    try:
        import sqlite3
        from sql_agent import USER_PREFERENCES_DB
        
        conn = sqlite3.connect(USER_PREFERENCES_DB)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM user_preferences WHERE user_id = ?", (user_id,))
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        return {
            "message": f"Deleted {deleted_count} preferences for user {user_id}",
            "user_id": user_id,
            "deleted_count": deleted_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete preferences: {str(e)}")

@app.get("/tables")
async def list_tables():
    """
    List all available database tables
    """
    try:
        tables = db.get_table_names()
        return {
            "tables": tables,
            "count": len(tables)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list tables: {str(e)}")

@app.get("/schema/{table_name}")
async def get_table_schema(table_name: str):
    """
    Get schema for a specific table
    
    - **table_name**: Name of the table
    """
    try:
        schema = db.get_table_info([table_name])
        return {
            "table": table_name,
            "schema": schema
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get schema: {str(e)}")

@app.post("/clear")
async def clear_conversation(request: ConversationClearRequest):
    """
    Clear conversation memory for a user or globally
    
    - **user_id**: Optional user ID. If not provided, clears global agent memory
    """
    try:
        global _global_agent
        import sql_agent
        
        # Reset global agent
        sql_agent._global_agent = None
        
        new_user_id = request.user_id or str(uuid.uuid4())
        
        return {
            "message": "Conversation memory cleared",
            "new_user_id": new_user_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear memory: {str(e)}")

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """
    WebSocket endpoint for real-time streaming responses
    
    - **user_id**: User identifier for session management
    
    Send JSON: {"query": "your question here"}
    Receive streaming updates as the agent processes
    """
    await websocket.accept()
    
    try:
        while True:
            # Receive query from client
            data = await websocket.receive_text()
            request_data = json.loads(data)
            query = request_data.get("query", "")
            
            if not query:
                await websocket.send_json({
                    "error": "Query is required"
                })
                continue
            
            # Send processing started message
            await websocket.send_json({
                "status": "processing",
                "user_id": user_id,
                "query": query
            })
            
            try:
                agent = get_or_create_agent()
                config = {"configurable": {"thread_id": user_id}}
                enhanced_input = f"[User ID: {user_id}]\n{query}"
                
                # Stream events to client
                events = agent.stream(
                    {"messages": [("user", enhanced_input)]},
                    config=config,
                    stream_mode="values",
                )
                
                sql_query = None
                raw_result = None
                
                for event in events:
                    if "messages" in event:
                        msg = event["messages"][-1]
                        
                        # Send tool calls
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            for tool_call in msg.tool_calls:
                                await websocket.send_json({
                                    "type": "tool_call",
                                    "tool": tool_call.get('name'),
                                    "args": tool_call.get('args', {})
                                })
                                
                                if tool_call.get('name') == 'sql_db_query':
                                    sql_query = tool_call.get('args', {}).get('query', '')
                        
                        # Send content updates
                        if hasattr(msg, 'content') and isinstance(msg.content, str):
                            if hasattr(msg, 'name') and msg.name == 'sql_db_query':
                                raw_result = msg.content
                                await websocket.send_json({
                                    "type": "raw_result",
                                    "content": msg.content
                                })
                            elif not hasattr(msg, 'name') or not msg.name:
                                await websocket.send_json({
                                    "type": "message",
                                    "content": msg.content
                                })
                
                # Send completion message
                await websocket.send_json({
                    "status": "completed",
                    "sql_query": sql_query,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                await websocket.send_json({
                    "status": "error",
                    "error": str(e)
                })
    
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for user: {user_id}")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        try:
            await websocket.send_json({
                "status": "error",
                "error": str(e)
            })
        except:
            pass

# Run with: uvicorn app:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)