from pathlib import Path
import csv
import sqlite3

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data/db/retail.db"
ORDERS_CSV = BASE_DIR / "data/raw/retail_orders.csv"
PRODUCTS_CSV = BASE_DIR / "data/raw/products.csv"

DB_PATH.parent.mkdir(parents=True, exist_ok=True)

with sqlite3.connect(DB_PATH) as conn:
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS orders")
    cur.execute("DROP TABLE IF EXISTS products")

    cur.execute("""
        CREATE TABLE products (
            product_id TEXT PRIMARY KEY,
            product_name TEXT NOT NULL,
            category TEXT NOT NULL,
            list_price REAL NOT NULL,
            unit_cost REAL NOT NULL,
            supplier TEXT NOT NULL,
            warranty_months INTEGER NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE orders (
            order_id TEXT PRIMARY KEY,
            order_date TEXT NOT NULL,
            region TEXT NOT NULL,
            city TEXT NOT NULL,
            channel TEXT NOT NULL,
            customer_segment TEXT,
            product_id TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            unit_price REAL NOT NULL,
            discount_pct REAL NOT NULL,
            delivery_days INTEGER NOT NULL,
            rating REAL,
            returned TEXT NOT NULL,
            payment_method TEXT NOT NULL,
            FOREIGN KEY(product_id) REFERENCES products(product_id)
        )
    """)

    with open(PRODUCTS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        product_rows = [
            (
                r["product_id"], r["product_name"], r["category"],
                float(r["list_price"]), float(r["unit_cost"]),
                r["supplier"], int(r["warranty_months"])
            )
            for r in reader
        ]

    cur.executemany("""
        INSERT INTO products
        (product_id, product_name, category, list_price, unit_cost, supplier, warranty_months)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, product_rows)

    with open(ORDERS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        order_rows = [
            (
                r["order_id"], r["order_date"], r["region"], r["city"], r["channel"],
                r["customer_segment"] or None, r["product_id"], int(r["quantity"]),
                float(r["unit_price"]), float(r["discount_pct"]), int(r["delivery_days"]),
                float(r["rating"]) if r["rating"] else None,
                r["returned"], r["payment_method"]
            )
            for r in reader
        ]

    cur.executemany("""
        INSERT INTO orders
        (order_id, order_date, region, city, channel, customer_segment,
         product_id, quantity, unit_price, discount_pct, delivery_days,
         rating, returned, payment_method)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, order_rows)

    conn.commit()

print(f"Database created: {DB_PATH}")
print("Tables: orders, products")
