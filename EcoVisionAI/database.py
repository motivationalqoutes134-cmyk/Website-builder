import sqlite3
from datetime import datetime, timezone

SCHEMA = """CREATE TABLE IF NOT EXISTS predictions (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 predicted_category TEXT NOT NULL,
 confidence REAL NOT NULL,
 created_at TEXT NOT NULL
)"""

def connect(path):
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con

def init_db(path):
    with connect(path) as con:
        con.execute(SCHEMA)
        con.commit()

def save_prediction(path, category, confidence):
    with connect(path) as con:
        cur = con.execute("INSERT INTO predictions (predicted_category, confidence, created_at) VALUES (?, ?, ?)", (category, confidence, datetime.now(timezone.utc).isoformat()))
        con.commit()
        return cur.lastrowid

def fetch_history(path, limit=50):
    with connect(path) as con:
        rows = con.execute("SELECT id, predicted_category, confidence, created_at FROM predictions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]

def dashboard_stats(path):
    with connect(path) as con:
        total = con.execute("SELECT COUNT(*) AS n FROM predictions").fetchone()["n"]
        rows = con.execute("SELECT predicted_category AS category, COUNT(*) AS count FROM predictions GROUP BY predicted_category ORDER BY count DESC, category").fetchall()
        avg = con.execute("SELECT COALESCE(AVG(confidence), 0) AS value FROM predictions").fetchone()["value"]
    return {"total": total, "average_confidence": round(float(avg or 0), 4), "categories": [dict(r) for r in rows]}
