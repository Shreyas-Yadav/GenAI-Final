# test_connection.py
import os


import psycopg2 
from dotenv import load_dotenv
load_dotenv()

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

def test_postgres_connection():
    """
    Verifies that we can connect to Postgres and run a trivial query.
    """
    conn = psycopg2.connect(**database_env)
    try:
        cur = conn.cursor()
        cur.execute("SELECT * from users;")
        # row = cur.fetchone()
        for row in cur.fetchall():
            print(row)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

test_postgres_connection()