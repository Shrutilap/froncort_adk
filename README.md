# Froncort - Medical Resource Allocation Agent

A multi-agent system that converts natural language queries into SQL for hospital resource management, built with Google's Agent Development Kit (ADK).

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Google API Key (Gemini / Vertex AI)

### Setup
```bash
# Clone repository
git clone -b main https://github.com/Shrutilap/froncort_adk.git
cd froncort_adk

# Create .env file
cat > .env << EOF
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_GENAI_USE_VERTEXAI=FALSE
DB_USER=hospital_user
DB_PASSWORD=hospital_pass
DB_HOST=db
DB_PORT=3306
DB_NAME=hospital_data
DATABASE_URL=sqlite:///./my_chatbot_data.db
VITE_REACT_APP_FASTAPI_URL=http://localhost:8000
DEBUG=true
EOF

# Start services
docker compose up -d --build
```

---

## Problem Approach

### Challenge
Healthcare administrators need to query complex hospital databases without SQL expertise. Manual queries are time-consuming, error-prone, and lack contextual insights.

### Solution: Multi-Agent Natural Language to SQL System

**Key Capabilities:**
1. **Natural Language Understanding** - Plain English queries (e.g., "Which hospitals have available ICU beds?")
2. **Intelligent SQL Generation** - Auto-analyzes schema, creates optimized queries with joins/aggregations
3. **Query Validation** - Rewrites ambiguous queries, validates results match intent
4. **Contextual Responses** - Natural language answers with insights, not raw data

**Workflow:**
```
User Query → Schema Retrieval → Query Rewriting → SQL Generation 
→ Execution → Validation → Natural Language Response
```

**Example:**
```
Input: "Which hospitals have oxygen levels below 5000 liters?"
↓
System generates SQL with proper joins
↓
Output: "3 hospitals currently have oxygen levels below 5000 liters: 
        Ruby Hill Hospital (2,768 liters), Pune NMC Hospital (3,743 liters), 
        Narhe Hospital (2,370 liters)."
```

**Impact:**
- ⚡ Query time: Minutes → Seconds
- 👥 Non-technical staff can access data
- ✓ Automated validation reduces errors
- 📊 Handles 50+ hospitals effortlessly

---

## Data Sources

### Mock Hospital Database
Synthetically generated data for **50 hospitals in Pune, Maharashtra** using statistically realistic distributions.

**Generation:** Python script (`database_gen.py`) with NumPy/Pandas
- Seed: 7777 (reproducible)
- Hospital sizes: Small (45%), Medium (40%), Large (15%)
- Ownership: Government (55%), Private (40%), Trust (5%)

### Database Schema (MySQL)

#### 1. **hospitals** - Core metadata
```sql
hospital_id, hospital_name, region, latitude, longitude, 
ownership_type, max_capacity_beds, ward_capacity_beds
```
*50 hospitals, Pune coordinates, varied capacities (25-800 beds)*

#### 2. **hospital_resource_timeseries** - Real-time operations
```sql
timestamp, hospital_id, occupied_beds, total_beds, icu_beds, 
icu_occupied, ventilators, in_use_ventilators, available_oxygen_liters,
doctors (on_shift/required), nurses, ambulance_arrivals, critical_cases, 
ed_turnaround_time, ...
```
*25 metrics per hospital: bed utilization, equipment, staffing, patient flow*

#### 3. **hospital_finance_monthly** - Financial tracking
```sql
hospital_id, period, total_expenditure, operational_expenditure, 
staff_cost, supply_cost, revenue, budget_allocated, budget_remaining
```
*₹1M-₹9M monthly expenditure, staff costs 50-70% of total*

#### 4. **suppliers** - Vendor management
```sql
vendor_id, vendor_name, vendor_type, contact (JSON), 
lead_time_days, payment_terms_days
```
*3 vendors: Oxygen, Equipment, Diagnostics*

#### 5. **inventory_items** - Supply catalog
```sql
item_id, item_name, unit, reorder_level, reorder_qty, unit_cost, asset_flag
```
*Critical items: Oxygen (liters), Ventilators (units)*

**Data Realism:**
- Normal distributions for occupancy (not all at 100%)
- Geographic coordinates within Pune boundaries
- Staff shortages reflected (required > on_shift)
- Financial ratios match industry norms

---

## Agent Architecture

### Multi-Agent Orchestrator-Worker Model

```
┌─────────────────────────────────────┐
│   Root Agent (Gemini 2.5 Flash)    │ ← Orchestrates workflow
│   - SQL generation                  │
│   - Natural language responses      │
└─────┬───────────────────────────────┘
      │
      ├──────┬──────────┬─────────────┐
      ▼      ▼          ▼             ▼
  get_schema  run_sql  rewrite_     evaluate_
  _tool       _query   prompt       result
  [Function]  _tool    _agent       _agent
              [Function] [Sub-Agent] [Sub-Agent]
```

### Core Components

#### 1. **Root Agent** (`sql_agent/agent.py`)
**Role:** Central orchestrator
- Coordinates all tools/agents
- Generates SQL from refined queries
- Formats natural language responses
- Manages conversation state (SQLite sessions)

#### 2. **Rewrite Prompt Agent** (`subagents/rewrite_prompt.py`)
**Role:** Query clarification
- Converts ambiguous queries → structured prompts
- Maps colloquial terms → database columns
- Example: "busy hospitals" → "hospitals with bed occupancy > 80%"

#### 3. **Evaluate Result Agent** (`subagents/evaluate_results.py`)
**Role:** Quality assurance
- Validates results match user intent
- Returns "Correct" or "Partial"
- Enables retry logic for incorrect queries

#### 4. **Function Tools** (`tools.py`)
- **get_schema_tool:** Retrieves database structure (LangChain SQLDatabase)
- **run_sql_query_tool:** Executes queries against MySQL

### Design Rationale

**Why Multi-Agent?**
1. **Separation of Concerns** - Each agent has one responsibility
2. **Modularity** - Components reusable independently
3. **Cognitive Load Reduction** - Specialized agents excel at specific tasks
4. **Error Isolation** - One failure doesn't crash entire system

**Technology Choices:**
- **Gemini 2.5 Flash** - Fast inference, high quality
- **LangChain** - Standardized database utilities
- **SQLite** - Session persistence (Google ADK DatabaseSessionService)
- **FastAPI** - Backend API
- **React** - Frontend UI

---

## Adaptive Learning & Reasoning

### Conversational Context via Session Management

**SQLite-Based Persistence:**
- Stores full conversation history in `my_chatbot_data.db`
- Enables follow-up questions without context loss
- Cross-session memory for same user

**Robust Session Lifecycle:**

1. **Creation with Exponential Backoff**
```python
async def ensure_session_with_retries(max_retries=5):
    # Attempts: 0.1s, 0.2s, 0.4s, 0.8s, 1.6s delays
    # Handles database locks, race conditions
```

2. **Automatic Recovery During Execution**
```python
async def run_agent_with_session_recovery(max_attempts=3):
    # Detects "Session not found" errors
    # Recreates session transparently
    # Retries agent execution
```

3. **History Reconstruction**
- Parses SQLite state JSON (multiple formats)
- Extracts sender/text chronologically
- Supports ADK native + custom formats

**Example: Context Retention**
```
User: "Which hospitals have available ICU beds?"
Agent: [Stores results in SQLite] "15 hospitals have ICU beds available..."

User: "What about their oxygen levels?"
Agent: [Retrieves session → knows "their" = those 15 hospitals]
       "Among those 15 hospitals, oxygen levels range from..."
```

### Complete Reasoning Flow

```
POST /chat → ensure_session_with_retries() → run_agent_with_session_recovery()
    ↓
Root Agent receives query → Loads context from SQLite
    ↓
get_schema_tool() → rewrite_prompt_agent → Generate SQL
    ↓
run_sql_query_tool() → evaluate_result_agent
    ↓
Format response → Update SQLite session → Return to user
```

**Key Endpoints:**
- `POST /sessions/ensure` - Create/verify session
- `GET /history/{user_id}/{session_id}` - Retrieve conversation
- `POST /chat` - Main query processing

---

## Limitations & Future Extensions

### Current Limitations

1. **Localhost Database** - Not scalable, single point of failure
2. **Static Snapshot Data** - No real-time HMS/EHR integration
3. **Synchronous Processing** - No event streaming (Kafka)
4. **Gemini API Only** - Not using Vertex AI fine-tuning/context optimization
5. **No Query Caching** - Redis could improve performance
6. **English Only** - No multilingual support (Hindi/Marathi)

### Proposed Extensions

1. **Cloud Database Migration**
   - AWS RDS / Google Cloud SQL / Azure Database
   - Automated backups, read replicas, IAM auth

2. **Real-Time Integration**
   - WebSocket endpoints for live updates
   - HMS/EHR API webhooks
   - Critical threshold alerts

3. **Event Streaming**
   - Apache Kafka cluster for async processing
   - Kafka Streams for real-time aggregation
   - Event replay for fault tolerance

4. **Vertex AI Enterprise**
   - Custom model fine-tuning on hospital data
   - Context optimization for relevant schema
   - MLOps pipelines, feature store

5. **Performance Optimization**
   - Redis caching for frequent queries
   - Database indexing on hot columns
   - Connection pooling (RDS Proxy)

6. **Multilingual Support**
   - Google Cloud Translation API
   - Marathi/Hindi query detection
   - Medical term translation accuracy

---

## Contributors

**Shruti Lapalikar** - Initial development

---


## Acknowledgments

- Google Agent Development Kit (ADK)
- LangChain for database utilities
- FastAPI & React frameworks