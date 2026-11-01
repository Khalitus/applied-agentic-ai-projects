"""SQLite storage. SQL tool functions will use bounded, parameterized queries."""
import sqlite3
from pathlib import Path

from src.config import DB_PATH
from src.data import load_hourly


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def initialize(path: Path = DB_PATH) -> int:
    df = load_hourly()
    with connect(path) as connection:
        df.to_sql("bike_rentals", connection, if_exists="replace", index=False)
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_bike_date_hour "
            "ON bike_rentals (dteday, hr)"
        )
        count = connection.execute("SELECT COUNT(*) FROM bike_rentals").fetchone()[0]
    return count


if __name__ == "__main__":
    print(f"Loaded {initialize():,} rows into {DB_PATH}")
