# sql_agent/tools.py

import os
import logging
import json
import sqlite3  
from typing import Optional
from urllib.parse import quote_plus
from pydantic import BaseModel
from dotenv import load_dotenv

from google.adk.tools.function_tool import FunctionTool
from langchain_community.utilities import SQLDatabase
from .config import DB_USER, DB_PASSWORD, DB_HOST, DB_NAME
from urllib.parse import quote_plus
logger = logging.getLogger("chat-api.tools")

# try:
#     from .config import (
#         DB_USER, DB_PASSWORD, DB_HOST, DB_NAME, DATABASE_URL
#     )
# except ImportError:
#     load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
#     logger.warning("Could not import from .config, loading .env directly.")
#     DB_USER = os.getenv("DB_USER")
#     DB_PASSWORD = quote_plus(os.getenv("DB_PASSWORD", ""))
#     DB_HOST = os.getenv("DB_HOST")
#     DB_NAME = os.getenv("DB_NAME")
#     DATABASE_URL = os.getenv("DATABASE_URL")

DB_PASSWORD_ENCODED = quote_plus(DB_PASSWORD)

connection_uri = f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD_ENCODED}@{DB_HOST}/{DB_NAME}"
logger.info(f"Connecting to DB: mysql+mysqlconnector://{DB_USER}:*****@{DB_HOST}/{DB_NAME}")


try:
    db = SQLDatabase.from_uri(connection_uri)
    logger.info("External database connected successfully ✅")
except Exception as e:
    logger.error(f"Failed to connect to external database: {e}")
   
    class DummyDB:
        def get_table_info(self, *args, **kwargs): return "ERROR: DB NOT CONNECTED"
        def run(self, *args, **kwargs): return "ERROR: DB NOT CONNECTED"
    db = DummyDB()

def get_schema(input: Optional[dict] = None) -> dict:
    try:
        if isinstance(input, dict) and input.get("table"):
            table_name = input["table"]
            schema = db.get_table_info([table_name])
        else:
            schema = db.get_table_info()
        return {"schema_description": schema}
    except Exception as e:
        logger.error(f"Error in get_schema: {e}")
        return {"error": str(e)}

def run_sql_query(input: Optional[dict] = None) -> dict:
    sql_query = input.get("query")
    if not sql_query:
        return {"error": "No query provided in the 'query' key."}
    try:
        result = db.run(sql_query)
        return {"raw_result": result}
    except Exception as e:
        logger.error(f"Error in run_sql_query: {e}")
        return {"error": str(e)}




get_schema_tool = FunctionTool(get_schema)
run_sql_query_tool = FunctionTool(run_sql_query)

