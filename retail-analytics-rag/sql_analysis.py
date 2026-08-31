from pathlib import Path
import sqlite3

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
ORDERS_CSV = BASE_DIR / "data" / "retail_orders.csv"
PRODUCTS_CSV = BASE_DIR / "data" / "products.csv"
DB_PATH = BASE_DIR / "data" / "retail.db"


def load_data():
    """Load both CSV files into Pandas DataFrames."""
    orders = pd.read_csv(ORDERS_CSV, parse_dates=["order_date"])
    products = pd.read_csv(PRODUCTS_CSV)
    return orders, products


def inspect_data(orders, products):
    """Display basic information about the raw datasets."""
    print("\n=== Orders data ===")
    print("Rows:", len(orders))
    print("Columns:", len(orders.columns))
    print("\nFirst 5 rows:")
    print(orders.head())
    print("\nMissing values:")
    print(orders.isnull().sum())
    print("\nDuplicate rows:")
    print(orders.duplicated().sum())

    print("\n=== Products data ===")
    print("Rows:", len(products))
    print("Columns:", len(products.columns))
    print("\nFirst 5 rows:")
    print(products.head())
    print("\nMissing values:")
    print(products.isnull().sum())
    print("\nDuplicate rows:")
    print(products.duplicated().sum())


def clean_data(orders, products):
    """Clean basic data-quality problems."""
    orders = orders.copy().drop_duplicates()
    products = products.copy().drop_duplicates()

    orders["customer_segment"] = orders["customer_segment"].fillna("Unknown").str.strip()

    order_text_columns = [
        "region",
        "city",
        "channel",
        "payment_method",
        "returned",
        "order_id",
        "product_id",
    ]
    for column in order_text_columns:
        orders[column] = orders[column].str.strip()

    product_text_columns = [
        "product_id",
        "product_name",
        "category",
        "supplier",
    ]
    for column in product_text_columns:
        products[column] = products[column].str.strip()

    # Missing ratings remain NaN because no rating is different from an average rating.
    return orders, products


def validate_data(orders, products):
    """Check important business rules before loading data into SQL."""
    invalid_quantity = (orders["quantity"] <= 0).sum()
    invalid_price = (orders["unit_price"] <= 0).sum()
    invalid_discount = (
        (orders["discount_pct"] < 0) | (orders["discount_pct"] > 1)
    ).sum()
    invalid_rating = (~orders["rating"].dropna().between(1, 5)).sum()

    valid_product_ids = set(products["product_id"])
    invalid_product_ids = (~orders["product_id"].isin(valid_product_ids)).sum()

    duplicate_order_ids = orders["order_id"].duplicated().sum()
    duplicate_product_ids = products["product_id"].duplicated().sum()

    results = {
        "invalid_quantities": int(invalid_quantity),
        "invalid_prices": int(invalid_price),
        "invalid_discounts": int(invalid_discount),
        "invalid_ratings": int(invalid_rating),
        "invalid_product_ids": int(invalid_product_ids),
        "duplicate_order_ids": int(duplicate_order_ids),
        "duplicate_product_ids": int(duplicate_product_ids),
    }

    print("\n=== Data validation ===")
    for name, value in results.items():
        print(f"{name.replace('_', ' ').capitalize()}: {value}")

    return results


def create_database(orders, products):
    """Store cleaned DataFrames as SQLite tables."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as connection:
        orders.to_sql("orders", connection, if_exists="replace", index=False)
        products.to_sql("products", connection, if_exists="replace", index=False)


def run_query(query):
    """Run an SQL query and return the result as a DataFrame."""
    with sqlite3.connect(DB_PATH) as connection:
        return pd.read_sql_query(query, connection)


def verify_database():
    """Check that both SQL tables were created successfully."""
    orders_count = run_query(
        """
        SELECT COUNT(*) AS rows
        FROM orders
        """
    ).iloc[0]["rows"]

    products_count = run_query(
        """
        SELECT COUNT(*) AS rows
        FROM products
        """
    ).iloc[0]["rows"]

    print("\n=== Database check ===")
    print("Orders stored:", orders_count)
    print("Products stored:", products_count)

    return {
        "orders": int(orders_count),
        "products": int(products_count),
    }


def get_overall_kpis():
    query = """
    SELECT
        COUNT(*) AS total_orders,
        SUM(quantity) AS units_sold,
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


def recent_orders():
    """Return the 10 most recent orders."""
    query = """
    SELECT
        order_id,
        order_date,
        region,
        city,
        channel,
        product_id,
        quantity,
        unit_price
    FROM orders
    ORDER BY order_date DESC
    LIMIT 10
    """
    return run_query(query)


def south_online_orders():
    """Return online orders from the South region."""
    query = """
    SELECT
        order_id,
        order_date,
        city,
        customer_segment,
        product_id,
        quantity,
        unit_price
    FROM orders
    WHERE region = 'South'
      AND channel = 'Online'
    ORDER BY order_date DESC
    """
    return run_query(query)


def high_discount_orders():
    """Return orders with discounts of 10% or higher."""
    query = """
    SELECT
        order_id,
        order_date,
        region,
        channel,
        customer_segment,
        product_id,
        quantity,
        unit_price,
        discount_pct
    FROM orders
    WHERE discount_pct >= 0.10
    ORDER BY discount_pct DESC
    """
    return run_query(query)


def largest_orders():
    """Return the 10 largest individual order values."""
    query = """
    SELECT
        order_id,
        order_date,
        region,
        product_id,
        quantity,
        unit_price,
        discount_pct,
        ROUND(
            quantity * unit_price * (1 - discount_pct),
            2
        ) AS order_value
    FROM orders
    ORDER BY order_value DESC
    LIMIT 10
    """
    return run_query(query)


def low_rated_orders():
    """Return orders with a rating of 2 or lower."""
    query = """
    SELECT
        order_id,
        region,
        city,
        rating,
        delivery_days
    FROM orders
    WHERE rating <= 2
    ORDER BY rating
    """
    return run_query(query)


def region_performance():
    """Summarize sales performance by region."""
    query = """
    SELECT
        region,
        COUNT(*) AS total_orders,
        SUM(quantity) AS units_sold,
        ROUND(
            SUM(quantity * unit_price * (1 - discount_pct)),
            2
        ) AS total_sales,
        ROUND(
            AVG(quantity * unit_price * (1 - discount_pct)),
            2
        ) AS average_order_value,
        ROUND(
            AVG(delivery_days),
            2
        ) AS average_delivery_days
    FROM orders
    GROUP BY region
    ORDER BY total_sales DESC
    """
    return run_query(query)


def channel_performance():
    """Summarize performance by sales channel."""
    query = """
    SELECT
        channel,
        COUNT(*) AS total_orders,
        SUM(quantity) AS units_sold,
        ROUND(
            SUM(quantity * unit_price * (1 - discount_pct)),
            2
        ) AS total_sales,
        ROUND(
            AVG(quantity * unit_price * (1 - discount_pct)),
            2
        ) AS average_order_value
    FROM orders
    GROUP BY channel
    ORDER BY total_sales DESC
    """
    return run_query(query)


def large_customer_segments():
    """Return customer segments with more than 200 orders."""
    query = """
    SELECT
        customer_segment,
        COUNT(*) AS total_orders,
        SUM(quantity) AS units_sold,
        ROUND(
            SUM(quantity * unit_price * (1 - discount_pct)),
            2
        ) AS total_sales
    FROM orders
    GROUP BY customer_segment
    HAVING COUNT(*) > 200
    ORDER BY total_orders DESC
    """
    return run_query(query)


def slow_regions():
    """Return regions averaging more than 2 delivery days."""
    query = """
    SELECT
        region,
        ROUND(AVG(delivery_days), 2) AS average_delivery_days
    FROM orders
    GROUP BY region
    HAVING AVG(delivery_days) > 2
    ORDER BY average_delivery_days DESC
    """
    return run_query(query)


def city_order_summary():
    """Summarize order activity by city."""
    query = """
    SELECT
        city,
        COUNT(*) AS total_orders,
        SUM(quantity) AS units_sold,
        ROUND(AVG(delivery_days), 2) AS average_delivery_days
    FROM orders
    GROUP BY city
    ORDER BY total_orders DESC
    """
    return run_query(query)


def monthly_sales():
    """Summarize sales performance by month."""
    query = """
    SELECT
        SUBSTR(order_date, 1, 7) AS month,
        COUNT(*) AS total_orders,
        SUM(quantity) AS units_sold,
        ROUND(
            SUM(quantity * unit_price * (1 - discount_pct)),
            2
        ) AS total_sales,
        ROUND(
            AVG(quantity * unit_price * (1 - discount_pct)),
            2
        ) AS average_order_value,
        ROUND(
            AVG(discount_pct) * 100,
            2
        ) AS average_discount_pct
    FROM orders
    GROUP BY month
    ORDER BY month
    """
    return run_query(query)


def joined_order_sample():
    """Return sample orders joined with product information."""
    query = """
    SELECT
        o.order_id,
        o.order_date,
        o.product_id,
        p.product_name,
        p.category,
        o.quantity,
        o.unit_price,
        p.unit_cost
    FROM orders AS o
    INNER JOIN products AS p
        ON o.product_id = p.product_id
    LIMIT 10
    """
    return run_query(query)


def product_profitability():
    """Summarize sales and profit by product."""
    query = """
    SELECT
        p.product_id,
        p.product_name,
        p.category,
        COUNT(*) AS total_orders,
        SUM(o.quantity) AS units_sold,
        ROUND(
            SUM(o.quantity * o.unit_price * (1 - o.discount_pct)),
            2
        ) AS total_sales,
        ROUND(
            SUM(o.quantity * p.unit_cost),
            2
        ) AS total_cost,
        ROUND(
            SUM(
                o.quantity * o.unit_price * (1 - o.discount_pct)
                - o.quantity * p.unit_cost
            ),
            2
        ) AS total_profit,
        ROUND(
            SUM(
                o.quantity * o.unit_price * (1 - o.discount_pct)
                - o.quantity * p.unit_cost
            )
            / SUM(o.quantity * o.unit_price * (1 - o.discount_pct))
            * 100,
            2
        ) AS profit_margin_pct
    FROM orders AS o
    INNER JOIN products AS p
        ON o.product_id = p.product_id
    GROUP BY
        p.product_id,
        p.product_name,
        p.category
    ORDER BY total_profit DESC
    """
    return run_query(query)


def category_profitability():
    """Summarize profitability by product category."""
    query = """
    SELECT
        p.category,
        COUNT(*) AS total_orders,
        SUM(o.quantity) AS units_sold,
        ROUND(
            SUM(o.quantity * o.unit_price * (1 - o.discount_pct)),
            2
        ) AS total_sales,
        ROUND(
            SUM(o.quantity * p.unit_cost),
            2
        ) AS total_cost,
        ROUND(
            SUM(
                o.quantity * o.unit_price * (1 - o.discount_pct)
                - o.quantity * p.unit_cost
            ),
            2
        ) AS total_profit,
        ROUND(
            SUM(
                o.quantity * o.unit_price * (1 - o.discount_pct)
                - o.quantity * p.unit_cost
            )
            / SUM(o.quantity * o.unit_price * (1 - o.discount_pct))
            * 100,
            2
        ) AS profit_margin_pct
    FROM orders AS o
    INNER JOIN products AS p
        ON o.product_id = p.product_id
    GROUP BY p.category
    ORDER BY total_profit DESC
    """
    return run_query(query)


def supplier_summary():
    """Summarize supplier contribution to sales and profit."""
    query = """
    SELECT
        p.supplier,
        COUNT(*) AS total_orders,
        SUM(o.quantity) AS units_sold,
        ROUND(
            SUM(o.quantity * o.unit_price * (1 - o.discount_pct)),
            2
        ) AS total_sales,
        ROUND(
            SUM(
                o.quantity * o.unit_price * (1 - o.discount_pct)
                - o.quantity * p.unit_cost
            ),
            2
        ) AS total_profit
    FROM orders AS o
    INNER JOIN products AS p
        ON o.product_id = p.product_id
    GROUP BY p.supplier
    ORDER BY total_profit DESC
    """
    return run_query(query)


def return_rate_by_category():
    """Calculate return rate by product category."""
    query = """
    SELECT
        p.category,
        COUNT(*) AS total_orders,
        SUM(o.quantity) AS units_sold,
        SUM(
            CASE
                WHEN o.returned = 'Yes' THEN 1
                ELSE 0
            END
        ) AS returned_orders,
        ROUND(
            SUM(
                CASE
                    WHEN o.returned = 'Yes' THEN 1
                    ELSE 0
                END
            ) * 100.0 / COUNT(*),
            2
        ) AS return_rate_pct
    FROM orders AS o
    INNER JOIN products AS p
        ON o.product_id = p.product_id
    GROUP BY p.category
    ORDER BY return_rate_pct DESC
    """
    return run_query(query)