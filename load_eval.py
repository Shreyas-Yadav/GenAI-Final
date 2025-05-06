#!/usr/bin/env python3
import csv
import psycopg2
from psycopg2.extras import execute_batch

# --- 1. Configuration: update with your connection info ---
DB_CONFIG = {
    "host":     "ep-steep-mouse-a4vsg15p-pooler.us-east-1.aws.neon.tech",
    "port":     5432,
    "dbname":   "neondb",
    "user":     "neondb_owner",
    "password": "npg_Ae1yDC8FohSX"
}
CSV_PATH = "github_agent_convergence_results.csv"

# --- 2. Insert logic ---
def load_csv_and_insert(csv_path, db_config):
    # Connect
    conn = psycopg2.connect(**db_config)
    cur  = conn.cursor()
    
    # Read CSV and prepare records
    questions = {}   # question_id -> question_text
    runs      = []   # list of (run_id, run_num, question_id, response_text)
    
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = int(row["question_id"])
            # collect unique questions
            if qid not in questions:
                questions[qid] = row["question"]
            # collect run entries
            runs.append((
                row["run_id"],
                int(row["run_num"]),
                qid,
                row["response"]
            ))
    
    # 2a) Insert questions
    question_tuples = [(qid, text) for qid, text in questions.items()]
    execute_batch(cur, """
        INSERT INTO question (question_id, question_text)
        VALUES (%s, %s)
        ON CONFLICT (question_id) DO NOTHING
    """, question_tuples)
    
    # 2b) Insert runs
    execute_batch(cur, """
        INSERT INTO run (run_id, run_num, question_id, response_text)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (run_id) DO NOTHING
    """, runs)
    
    # Commit & clean up
    conn.commit()
    cur.close()
    conn.close()
    print(f"Inserted {len(questions)} questions and {len(runs)} runs.")

# --- 3. Main execution ---
if __name__ == "__main__":
    load_csv_and_insert(CSV_PATH, DB_CONFIG)
