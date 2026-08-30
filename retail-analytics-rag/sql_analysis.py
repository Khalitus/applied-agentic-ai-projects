from pathlib import Path
import sqlite3
import pandas as pd

# FILE PATHS

BASE_DIR = Path(__file__).resolve().parent

ORDERS_CSV = BASE_DIR / "data" / "retail_orders.csv"
PRODUCTS_CSV = BASE_DIR / "data" / "products.csv"
DB_PATH = BASE_DIR / "data" / "retail.db"

# LOAD DATA

def load_data():
    """Load both CSV files into Pandas DataFrames."""

    orders = pd.read_csv(
        ORDERS_CSV,
        parse_dates=["order_date"]
    )

    products = pd.read_csv(PRODUCTS_CSV)

    return orders, products

# CLEAN DATA

def clean_data(orders, products):
    """Basic cleaning before storing data in SQL."""

    orders = orders.copy()
    products = products.copy()

    orders["customer_segment"] = (
        orders["customer_segment"]
        .fillna("Unknown")
    )

    orders["rating"] = (
        orders["rating"]
        .fillna(orders["rating"].median())
    )

    return orders, products


# CREATE SQLITE DATABASE

def create_database(orders, products):
    """Store both DataFrames as SQL tables."""

    with sqlite3.connect(DB_PATH) as connection:

        orders.to_sql(
            "orders",
            connection,
            if_exists="replace",
            index=False
        )

        products.to_sql(
            "products",
            connection,
            if_exists="replace",
            index=False
        )

# RUN SQL

def run_query(query):
    """Run an SQL query and return a DataFrame."""

    with sqlite3.connect(DB_PATH) as connection:

        result = pd.read_sql_query(
            query,
            connection
        )

    return result

def get_overall_kpis():

    query = """
    SELECT
        COUNT(*) AS total_orders,

        ROUND(
            SUM(quantity * unit_price * (1 - discount_pct)),
            2
        ) AS total_sales,

        ROUND(
            AVG(quantity * unit_price * (1 - discount_pct)),
            2
        ) AS average_order_value

    FROM orders
    """

    return run_query(query)

def sales_by_region():

    query = """
    SELECT
        region,

        ROUND(
            SUM(quantity * unit_price * (1 - discount_pct)),
            2
        ) AS sales

    FROM orders

    GROUP BY region

    ORDER BY sales DESC
    """

    return run_query(query)

def monthly_sales():
    pass


def product_profitability():
    pass


def return_rate_by_category():
    pass