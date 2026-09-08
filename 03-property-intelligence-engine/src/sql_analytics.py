import pandas as pd

from src.database import get_connection


def run_query(query, params=()):
    with get_connection() as conn:
        return pd.read_sql_query(
            query,
            conn,
            params=params,
        )