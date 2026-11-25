"""Import source CSV to SQLite, preserving all source columns for SQL analysis."""

import sqlite3
import pandas as pd

from src.config import DATABASE, RAW_CSV
from src.features import FEATURES, TARGET


def main():
    if not RAW_CSV.exists():
        raise SystemExit("CSV missing: run python -m scripts.fetch_data first")
    frame = pd.read_csv(RAW_CSV, sep=";", encoding="utf-8")
    required = set(FEATURES + [TARGET])
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")
    if frame.empty or not frame[TARGET].isin(["yes", "no"]).all():
        raise ValueError("Invalid target values or empty dataset")

    # Dataset has no user identifier: row_id is only a source-row surrogate.
    frame.insert(0, "row_id", range(1, len(frame) + 1))
    DATABASE.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DATABASE) as connection:
        frame.to_sql("campaign_records", connection, if_exists="replace", index=False)
        connection.execute("CREATE UNIQUE INDEX idx_campaign_row ON campaign_records(row_id)")
        connection.execute("CREATE INDEX idx_campaign_outcome ON campaign_records(y)")
        connection.execute("CREATE INDEX idx_campaign_job ON campaign_records(job)")
    print(f"Imported {len(frame):,} rows into {DATABASE}")


if __name__ == "__main__":
    main()