from pathlib import Path
import sqlite3
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "property_intelligence.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

def setup_database():
    neighborhoods = pd.read_csv(DATA_DIR / "neighborhoods.csv")
    properties = pd.read_csv(DATA_DIR / "properties.csv")
    sales = pd.read_csv(DATA_DIR / "sales.csv")

    with get_connection() as conn:
        neighborhoods.to_sql("neighborhoods", conn, if_exists="replace", index=False)
        properties.to_sql("properties", conn, if_exists="replace", index=False)
        sales.to_sql("sales", conn, if_exists="replace", index=False)

def show_database_summary():
    with get_connection() as conn:
        for table in ["neighborhoods", "properties", "sales"]:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"{table}: {count} rows")
