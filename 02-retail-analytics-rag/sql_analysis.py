import sqlite3
import pandas as pd


DB_PATH = "retail.db"


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df


def create_database(df):
    connection = sqlite3.connect(DB_PATH)

    df.to_sql(
        "orders",
        connection,
        if_exists="replace",
        index=False
    )

    connection.close()


def run_query(query):
    connection = sqlite3.connect(DB_PATH)

    result = pd.read_sql_query(
        query,
        connection
    )

    connection.close()

    return result