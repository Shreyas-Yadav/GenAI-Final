#!/usr/bin/env python3
"""
PostgreSQL Natural-Language Agent
Combines configuration, tools, LLM setup, and REPL into a single script.
"""

import os
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from dotenv import load_dotenv
load_dotenv()

from llama_index.llms.openrouter import OpenRouter
from llama_index.core import ServiceContext
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool

# ─── Configuration ─────────────────────────────────────────────────────────────

DB_HOST     = os.getenv("DB_HOST", "localhost")
DB_PORT     = int(os.getenv("DB_PORT", 5432))
DB_NAME     = os.getenv("DB_NAME", "your_database")
DB_USER     = os.getenv("DB_USER", "your_username")
DB_PASSWORD = os.getenv("DB_PASSWORD", "your_password")

database_env = {
    "host":     DB_HOST,
    "port":     DB_PORT,
    "database": DB_NAME,
    "user":     DB_USER,
    "password": DB_PASSWORD,
}

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise EnvironmentError("Please set the OPENROUTER_API_KEY environment variable")

MODEL_NAME = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-medium")


# ─── Tool Implementations ──────────────────────────────────────────────────────

def run_query(query: str) -> str:
    """Execute a SQL query and return results as a CSV-formatted string."""
    conn = psycopg2.connect(**database_env)
    try:
        df = pd.read_sql_query(query, conn)
        return df.to_csv(index=False)
    finally:
        conn.close()

def get_table_names() -> str:
    """Return a CSV listing all table names in the public schema."""
    conn = psycopg2.connect(**database_env)
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        tables = [row[0] for row in cur.fetchall()]
        df = pd.DataFrame({"table_name": tables})
        return df.to_csv(index=False)
    finally:
        conn.close()

def get_column_names(table: str) -> str:
    """Return a CSV listing all column names for a given table."""
    conn = psycopg2.connect(**database_env)
    try:
        cur = conn.cursor()
        cur.execute(f"""
            SELECT *
            FROM information_schema.columns
            WHERE table_name = {table}
            ORDER BY ordinal_position;
        """, (table,))
        cols = [row[0] for row in cur.fetchall()]
        df = pd.DataFrame({"column_name": cols})
        return df.to_csv(index=False)
    finally:
        conn.close()

def save_data_to_csv(data: str, filename: str) -> str:
    """Save a CSV-formatted string `data` to a file named `filename`."""
    df = pd.read_csv(StringIO(data))
    df.to_csv(filename, index=False)
    return f"Data saved to {filename}"

def generate_visualization(data: str, x_column: str, y_column: str, chart_type: str) -> str:
    """Generate and save a chart (scatter, bar, or line) from CSV `data`."""
    df = pd.read_csv(StringIO(data))
    ct = chart_type.lower()
    if ct == "scatter":
        plt.scatter(df[x_column], df[y_column])
    elif ct == "bar":
        plt.bar(df[x_column], df[y_column])
    elif ct == "line":
        plt.plot(df[x_column], df[y_column])
    else:
        raise ValueError(f"Unsupported chart type: {chart_type}")
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f"{chart_type.title()} of {y_column} vs {x_column}")
    plt.tight_layout()
    out_file = f"{ct}_{x_column}_{y_column}.png"
    plt.savefig(out_file)
    plt.close()
    return f"Visualization saved to {out_file}"


# ─── Agent Setup ────────────────────────────────────────────────────────────────

def build_agent() -> ReActAgent:
    llm = OpenRouter(api_key=OPENROUTER_API_KEY, model=MODEL_NAME)
    

    tools = [
        FunctionTool.from_defaults(
            name="run_query",
            fn=run_query,
            description="Execute an SQL query and return results in CSV format."
        ),
        FunctionTool.from_defaults(
            name="get_table_names",
            fn=get_table_names,
            description="List all table names in the public schema as CSV."
        ),
        FunctionTool.from_defaults(
            name="get_column_names",
            fn=get_column_names,
            description="List all column names for a given table as CSV."
        ),
        FunctionTool.from_defaults(
            name="save_data_to_csv",
            fn=save_data_to_csv,
            description="Save provided CSV data to a file with the given filename."
        ),
        FunctionTool.from_defaults(
            name="generate_visualization",
            fn=generate_visualization,
            description="Generate a scatter, bar, or line chart from CSV data."
        ),
    ]

    return ReActAgent.from_tools(tools=tools,llm=llm, verbose=True) 


# ─── Main Loop ─────────────────────────────────────────────────────────────────

def main():
    agent = build_agent()
    print("Welcome to the PostgreSQL NL Agent. Type 'exit' or 'quit' to stop.")
    while True:
        user_input = input("\nUser: ")
        if user_input.strip().lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        try:
            reply = agent.chat(user_input)
        except Exception as e:
            reply = f"Error during processing: {e}"
        print(f"Agent: {reply}")


if __name__ == "__main__":
    main()
