"""SQLite storage. SQL tool functions will use bounded, parameterized queries."""
import sqlite3
from pathlib import Path

from src.config import DB_PATH
from src.data import load_hourly


SCHEMA = """
CREATE TABLE bike_rentals (
    instant INTEGER PRIMARY KEY,
    dteday TEXT NOT NULL,
    season INTEGER NOT NULL,
    yr INTEGER NOT NULL,
    mnth INTEGER NOT NULL,
    hr INTEGER NOT NULL CHECK (hr BETWEEN 0 AND 23),
    holiday INTEGER NOT NULL CHECK (holiday IN (0, 1)),
    weekday INTEGER NOT NULL CHECK (weekday BETWEEN 0 AND 6),
    workingday INTEGER NOT NULL CHECK (workingday IN (0, 1)),
    weathersit INTEGER NOT NULL CHECK (weathersit BETWEEN 1 AND 4),
    temp REAL NOT NULL,
    atemp REAL NOT NULL,
    hum REAL NOT NULL CHECK (hum BETWEEN 0 AND 1),
    windspeed REAL NOT NULL,
    casual INTEGER NOT NULL CHECK (casual >= 0),
    registered INTEGER NOT NULL CHECK (registered >= 0),
    cnt INTEGER NOT NULL CHECK (cnt >= 0),
    CHECK (casual + registered = cnt)
)
"""


COLUMNS = (
    "instant",
    "dteday",
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "casual",
    "registered",
    "cnt",
)

def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection



def initialize(path: Path = DB_PATH) -> int:
    df = load_hourly()

    column_names = ", ".join(COLUMNS)
    placeholders = ", ".join("?" for _ in COLUMNS)

    insert_sql = (
        f"INSERT INTO bike_rentals ({column_names}) "
        f"VALUES ({placeholders})"
    )

    rows = df[list(COLUMNS)].itertuples(
        index=False,
        name=None,
    )

    with connect(path) as connection:
        connection.execute("BEGIN")

        connection.execute("DROP TABLE IF EXISTS bike_rentals")
        connection.execute(SCHEMA)

        connection.executemany(insert_sql, rows)

        connection.execute(
            "CREATE UNIQUE INDEX idx_bike_date_hour "
            "ON bike_rentals (dteday, hr)"
        )

        connection.execute(
            "CREATE INDEX idx_bike_hr "
            "ON bike_rentals (hr)"
        )

        count = connection.execute(
            "SELECT COUNT(*) FROM bike_rentals"
        ).fetchone()[0]

        if count != len(df):
            raise RuntimeError(
                f"Database count mismatch: {count} != {len(df)}"
            )

    return count

if __name__ == "__main__":
    print(f"Loaded {initialize():,} rows into {DB_PATH}")
