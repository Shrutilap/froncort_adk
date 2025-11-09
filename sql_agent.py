import sqlite3
import json
from dotenv import load_dotenv
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
import os

load_dotenv()

MYSQL_USER = os.getenv("DB_USER")
MYSQL_PASSWORD = os.getenv("DB_PASSWORD")
MYSQL_HOST = os.getenv("DB_HOST")
MYSQL_DB = os.getenv("DB_NAME")

mysql_uri = f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:3306/{MYSQL_DB}"
db = SQLDatabase.from_uri(mysql_uri)
print("Available tables:", db.get_table_names())

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# User preferences storage
USER_PREFERENCES_DB = "user_preferences.db"

def init_preferences_db():
    """Initialize user preferences with contextual fields"""
    conn = sqlite3.connect(USER_PREFERENCES_DB)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_preferences (
            user_id TEXT,
            priority_key TEXT,
            priority_value TEXT,
            context TEXT,
            feedback_text TEXT,
            source_query TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, priority_key)
        )
    """)
    conn.commit()
    conn.close()

@tool
def get_user_priorities(user_id: str) -> str:
    """
    Retrieve saved preferences for a specific user.
    Args:
        user_id: The unique identifier for the user
    Returns:
        JSON string of user preferences (e.g., {"cost": "low", "coverage": "high"})
    """
    conn = sqlite3.connect(USER_PREFERENCES_DB)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT priority_key, priority_value
        FROM user_preferences
        WHERE user_id = ?
    """, (user_id,))
    results = cursor.fetchall()
    conn.close()
    
    preferences = {key: value for key, value in results}
    return json.dumps(preferences) if preferences else "{}"

@tool
def update_user_priority(user_id: str, priority_key: str, priority_value: str, 
                        context: str = None, feedback_text: str = None, 
                        source_query: str = None) -> str:
    """
    Save or update a user's contextual preference for long-term learning.
    """
    print(f"🧠 Updating long-term preference for {user_id}: {priority_key} = {priority_value} (context={context})")
    
    conn = sqlite3.connect(USER_PREFERENCES_DB)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO user_preferences 
        (user_id, priority_key, priority_value, context, feedback_text, source_query)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id, priority_key) 
        DO UPDATE SET 
            priority_value = excluded.priority_value,
            context = excluded.context,
            feedback_text = excluded.feedback_text,
            source_query = excluded.source_query,
            updated_at = CURRENT_TIMESTAMP
    """, (user_id, priority_key, priority_value, context, feedback_text, source_query))
    conn.commit()
    conn.close()
    
    return f"✅ Preference saved: {priority_key} = {priority_value}"

@tool
def rewrite_user_query(user_input: str, db_schema: str) -> str:
    """
    Rewrite user input into a clear, structured natural language query 
    suitable for SQL generation.
    """
    rewrite_llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    prompt = f"""
You are a language simplification agent. Rewrite the user query into clear, 
structured natural language suitable for SQL query generation.

User input: {user_input}
Database schema: {db_schema}

Return only the rewritten prompt, nothing else. Make it:
- Clear and unambiguous
- Directly related to the schema
- Easy to convert into SQL

Rewritten query:
"""
    
    response = rewrite_llm.invoke(prompt)
    return response.content

@tool
def evaluate_sql_result(user_input: str, sql_query: str, result: str, db_schema: str) -> str:
    """
    Evaluate whether the SQL query result correctly answers the user's question.
    """
    eval_llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    prompt = f"""
Evaluate if the SQL query result correctly answers the user's question.

User question: {user_input}
SQL query: {sql_query}
Result: {result}
Schema: {db_schema}

Return ONLY "Correct" or "Partial" - no explanation.
"""
    
    response = eval_llm.invoke(prompt)
    return response.content.strip()

# Enhanced system prompt
ENHANCED_SYSTEM_PROMPT = """
You are an intelligent SQL agent with preference learning capabilities, powered by Google Gemini.

Your primary task is to:
1. Process user questions in natural language
2. Generate appropriate SQL queries
3. Execute queries and return results
4. Learn and adapt to user preferences over time

IMPORTANT: You have access to a user_id in the conversation context.

Your workflow:
1. **Load Preferences**: ALWAYS call get_user_priorities first with the user_id to retrieve saved preferences
2. **Understand Input**: Analyze the user's question and any preferences
3. **Check for Feedback**: If the user expresses dissatisfaction (e.g., "too expensive", "too slow", "not enough"), 
   call update_user_priority to save this preference for future queries
4. **Get Schema**: Use sql_db_schema to understand the database structure
5. **Rewrite Query**: Optionally use rewrite_user_query to clarify ambiguous input
6. **Generate SQL**: Create an appropriate SQL query considering user preferences
7. **Execute**: Run the query using sql_db_query
8. **Evaluate**: Use evaluate_sql_result to check if results match intent
9. **Respond**: Provide a natural language summary with structured data

Examples of learning from feedback:
- User: "Show me hotels" -> [returns expensive hotels] -> User: "No, those are too expensive"
  ACTION: Call update_user_priority(user_id, "cost", "low")
- User: "Find insurance plans" -> [returns basic plans] -> User: "I need better coverage"
  ACTION: Call update_user_priority(user_id, "coverage", "high")

Always incorporate learned preferences into future queries for that user.

Available tools:
- get_user_priorities: Get saved user preferences (CALL THIS FIRST)
- update_user_priority: Save new preference from user feedback
- rewrite_user_query: Clarify ambiguous queries
- evaluate_sql_result: Validate query results
- sql_db_list_tables: List all tables
- sql_db_schema: Get table schemas
- sql_db_query: Execute SQL queries
- sql_db_query_checker: Validate SQL syntax

IMPORTANT: Always generate valid SQL queries. Double-check table and column names against the schema.

IMPORTANT: Whenever a user expresses a preference (e.g. "cheap", "affordable", "premium", 
"better coverage", "low oxygen cost"), you MUST call the update_user_priority tool to record 
this before any database queries.
"""

# Global agent instance to maintain session memory
_global_agent = None

def get_or_create_agent():
    """Get existing agent or create new one (maintains session memory)"""
    global _global_agent
    
    if _global_agent is None:
        # Initialize long-term preferences DB
        init_preferences_db()
        
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        all_tools = toolkit.get_tools() + [
            get_user_priorities,
            update_user_priority,
            rewrite_user_query,
            evaluate_sql_result
        ]
        
        # Use session memory that persists across queries
        session_memory = MemorySaver()
        
        # Create the agent
        _global_agent = create_agent(
            llm,
            all_tools,
            system_prompt=ENHANCED_SYSTEM_PROMPT,
            checkpointer=session_memory
        )
        
        print("✅ Agent created with session memory:", type(session_memory).__name__)
    
    return _global_agent

def run_gemini_query(user_input: str, user_id: str = "default_user"):
    """Run a query with Gemini-powered agent"""
    agent = get_or_create_agent()
    
    config = {"configurable": {"thread_id": user_id}}
    enhanced_input = f"[User ID: {user_id}]\n{user_input}"
    
    print("\n" + "="*80)
    print(f"User: {user_id}")
    print(f"Query: {user_input}")
    print(f"Model: Gemini 2.0 Flash")
    print("="*80 + "\n")
    
    # Stream the agent response
    events = agent.stream(
        {"messages": [("user", enhanced_input)]},
        config=config,
        stream_mode="values",
    )
    
    response_parts = []
    for event in events:
        if "messages" in event:
            msg = event["messages"][-1]
            msg.pretty_print()
            if hasattr(msg, 'content') and isinstance(msg.content, str):
                response_parts.append(msg.content)
    
    return response_parts[-1] if response_parts else None

def format_gemini_response(events: list) -> dict:
    """Format the agent response into a structured JSON"""
    response_data = {
        "summary": "",
        "sql": "",
        "raw_result": "",
        "result_evaluation": "Processing",
        "model": "Gemini 2.0 Flash"
    }
    
    for event in events:
        if "messages" in event and len(event["messages"]) > 0:
            last_message = event["messages"][-1]
            
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                for tool_call in last_message.tool_calls:
                    if tool_call.get('name') == 'sql_db_query':
                        response_data["sql"] = tool_call.get('args', {}).get('query', '')
            
            if hasattr(last_message, 'content') and last_message.content:
                if isinstance(last_message.content, str):
                    if last_message.name == 'sql_db_query':
                        response_data["raw_result"] = last_message.content
                    elif not last_message.name:
                        response_data["summary"] = last_message.content
    
    response_data["result_evaluation"] = "Correct"
    return response_data

def interactive_gemini_mode():
    """Run Gemini agent in interactive mode"""
    print("=" * 80)
    print("SQL Agent with Google Gemini 2.0 Flash".center(80))
    print("=" * 80)
    print("\nCommands:")
    print("  - Type your SQL question in natural language")
    print("  - 'q' to quit")
    print("  - 'prefs' to view your preferences")
    print("  - 'clear' to start a new conversation")
    print("\n")
    
    user_id = input("Enter your user ID (or press Enter for 'demo_user'): ").strip()
    if not user_id:
        user_id = "demo_user"
    
    while True:
        user_input = input("\n💬 Enter your query: ").strip()
        
        if user_input.lower() == 'q':
            print("\n👋 Goodbye!")
            break
        
        if user_input.lower() == 'prefs':
            prefs = get_user_priorities.invoke({"user_id": user_id})
            print(f"\n📊 Current preferences for {user_id}:")
            preferences = json.loads(prefs)
            if preferences:
                print(json.dumps(preferences, indent=2))
            else:
                print("No preferences saved yet.")
            continue
        
        if user_input.lower() == 'clear':
            import uuid
            global _global_agent
            _global_agent = None  # Reset agent to clear memory
            user_id = str(uuid.uuid4())
            print(f"\n✨ Conversation cleared. New user ID: {user_id}\n")
            continue
        
        if not user_input:
            continue
        
        try:
            run_gemini_query(user_input, user_id)
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")
            import traceback
            traceback.print_exc()

def create_simple_gemini_agent():
    """Create a simple SQL agent with just basic tools"""
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    memory = MemorySaver()
    
    simple_prompt = """
You are an intelligent SQL agent powered by Google Gemini.

Your task is to:
1. Understand user questions in natural language
2. Retrieve database schema
3. Generate appropriate SQL queries
4. Execute queries and return results
5. Provide clear summaries

Always check the schema first, then generate and execute SQL queries.
Provide natural language summaries of the results.
"""
    
    agent = create_agent(
        llm,
        toolkit.get_tools(),
        system_prompt=simple_prompt,
        checkpointer=memory
    )
    
    return agent

# Global simple agent for mode 1
_simple_agent = None

def get_or_create_simple_agent():
    """Get existing simple agent or create new one"""
    global _simple_agent
    
    if _simple_agent is None:
        _simple_agent = create_simple_gemini_agent()
        print("✅ Simple agent created with session memory")
    
    return _simple_agent

def run_simple_query(user_input: str, user_id: str = "default_user"):
    """Run a simple query without preference learning"""
    agent = get_or_create_simple_agent()
    
    config = {
        "configurable": {
            "thread_id": user_id,
        }
    }
    
    print("\n" + "="*80)
    print(f"Query: {user_input}")
    print("="*80 + "\n")
    
    events = agent.stream(
        {"messages": [("user", user_input)]},
        config=config,
        stream_mode="values",
    )
    
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()

if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️ Error: GOOGLE_API_KEY not found in environment variables")
        print("Please add it to your .env file:")
        print("GOOGLE_API_KEY=your_api_key_here")
        exit(1)
    
    print("\n🚀 Choose mode:")
    print("1. Simple SQL Agent (basic queries)")
    print("2. Enhanced SQL Agent (with preference learning)")
    print("3. Run example queries")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        print("\n📝 Simple SQL Agent Mode")
        user_id = "simple_user"
        while True:
            query = input("\n💬 Enter query (or 'q' to quit): ").strip()
            if query.lower() == 'q':
                break
            if query:
                run_simple_query(query, user_id)
    
    elif choice == "2":
        interactive_gemini_mode()
    
    elif choice == "3":
        print("\n📋 Running example queries...\n")
        run_simple_query("Show me all tables in the database")
        
        print("\n" + "="*80)
        print("Example: Preference Learning")
        print("="*80 + "\n")
        demo_user = "demo_alice"
        run_gemini_query("Show me available products", demo_user)
    
    else:
        print("Invalid choice. Exiting.")