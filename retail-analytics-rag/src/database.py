from pathlib import Path
import sqlite3
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data/db/retail.db"

def get_connection():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"{DB_PATH} does not exist. Run: python build_db.py"
        )
    return sqlite3.connect(DB_PATH)

def run_query(query: str, params=None) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql_query(query, conn, params=params or ())
