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



# INSPECT RAW DATA


def inspect_data(orders, products):
    """Display basic information about the raw datasets."""

    print("\n=== ORDERS DATA ===")

    print("Rows:", len(orders))
    print("Columns:", len(orders.columns))

    print("\nFirst 5 rows:")
    print(orders.head())

    print("\nMissing values:")
    print(orders.isnull().sum())

    print("\nDuplicate rows:")
    print(orders.duplicated().sum())


    print("\n=== PRODUCTS DATA ===")

    print("Rows:", len(products))
    print("Columns:", len(products.columns))

    print("\nFirst 5 rows:")
    print(products.head())

    print("\nMissing values:")
    print(products.isnull().sum())

    print("\nDuplicate rows:")
    print(products.duplicated().sum())



# CLEAN DATA


def clean_data(orders, products):
    """Clean obvious data-quality problems."""

    orders = orders.copy()
    products = products.copy()

    orders = orders.drop_duplicates()
    products = products.drop_duplicates()

    orders["customer_segment"] = (
        orders["customer_segment"]
        .fillna("Unknown")
        .str.strip()
    )

    orders["region"] = orders["region"].str.strip()
    orders["city"] = orders["city"].str.strip()
    orders["channel"] = orders["channel"].str.strip()
    orders["payment_method"] = orders["payment_method"].str.strip()
    orders["returned"] = orders["returned"].str.strip()
    orders["order_id"] = orders["order_id"].str.strip()
    orders["product_id"] = orders["product_id"].str.strip()

    products["product_id"] = products["product_id"].str.strip()
    products["product_name"] = products["product_name"].str.strip()
    products["category"] = products["category"].str.strip()
    products["supplier"] = products["supplier"].str.strip()

    # Missing ratings remain NaN intentionally.

    return orders, products



# VALIDATE DATA


def validate_data(orders, products):
    """Check important business rules before loading data into SQL."""

    invalid_quantity = (
        orders["quantity"] <= 0
    ).sum()

    invalid_price = (
        orders["unit_price"] <= 0
    ).sum()

    invalid_discount = (
        (orders["discount_pct"] < 0)
        |
        (orders["discount_pct"] > 1)
    ).sum()

    invalid_rating = (
        ~orders["rating"]
        .dropna()
        .between(1, 5)
    ).sum()

    valid_product_ids = set(
        products["product_id"]
    )

    invalid_product_ids = (
        ~orders["product_id"]
        .isin(valid_product_ids)
    ).sum()

    duplicate_order_ids = (
        orders["order_id"]
        .duplicated()
        .sum()
    )

    duplicate_product_ids = (
        products["product_id"]
        .duplicated()
        .sum()
    )

    results = {
        "invalid_quantities": int(invalid_quantity),
        "invalid_prices": int(invalid_price),
        "invalid_discounts": int(invalid_discount),
        "invalid_ratings": int(invalid_rating),
        "invalid_product_ids": int(invalid_product_ids),
        "duplicate_order_ids": int(duplicate_order_ids),
        "duplicate_product_ids": int(duplicate_product_ids),
    }

    print("\n=== DATA VALIDATION ===")

    for name, value in results.items():
        print(
            f"{name.replace('_', ' ').title()}: {value}"
        )

    return results



# CREATE SQLITE DATABASE


def create_database(orders, products):
    """Store both cleaned DataFrames as SQLite tables."""

    DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

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



# RUN SQL QUERY


def run_query(query):
    """Run an SQL query and return the result as a DataFrame."""

    with sqlite3.connect(DB_PATH) as connection:

        result = pd.read_sql_query(
            query,
            connection
        )

    return result



# VERIFY DATABASE


def verify_database():
    """Check that both SQL tables were created successfully."""

    orders_check = run_query(
        """
        SELECT COUNT(*) AS rows
        FROM orders
        """
    )

    products_check = run_query(
        """
        SELECT COUNT(*) AS rows
        FROM products
        """
    )

    orders_count = orders_check.iloc[0]["rows"]
    products_count = products_check.iloc[0]["rows"]

    print("\n=== DATABASE CHECK ===")

    print(
        "Orders stored:",
        orders_count
    )

    print(
        "Products stored:",
        products_count
    )

    return {
        "orders": int(orders_count),
        "products": int(products_count)
    }



# OVERALL KPIs


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



# SALES BY REGION


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



# FUTURE TASKS


def monthly_sales():
    pass


def product_profitability():
    pass


def return_rate_by_category():
    pass