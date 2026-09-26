import sqlite3

import pandas as pd

from config import DB_PATH, RAW_DIR


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def build_database():
    tables = {
        "customers": RAW_DIR / "customers.csv",
        "products": RAW_DIR / "products.csv",
        "tickets": RAW_DIR / "tickets.csv",
    }

    with get_connection() as conn:
        for table, path in tables.items():
            pd.read_csv(path).to_sql(table, conn, if_exists="replace", index=False)

        conn.execute("CREATE INDEX IF NOT EXISTS idx_tickets_customer ON tickets(customer_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tickets_product ON tickets(product_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tickets_issue ON tickets(issue_type)")

    print(f"Database created: {DB_PATH}")


def query(sql, params=None):
    with get_connection() as conn:
        return pd.read_sql_query(sql, conn, params=params or ())


def load_modeling_frame():
    """Return one row per ticket using only fields available at ticket intake."""
    # TODO Task 3: write the JOIN yourself.
    # Required output should include ticket predictors plus:
    # region, customer_tier, account_age_months, total_orders,
    # lifetime_value, category, warranty_months, unit_price.
    raise NotImplementedError("Complete the SQL JOIN in Task 3.")


def get_ticket(ticket_id):
    sql = """
    SELECT
        t.*,
        c.region,
        c.customer_tier,
        c.account_age_months,
        c.total_orders,
        c.lifetime_value,
        p.product_name,
        p.category,
        p.warranty_months,
        p.unit_price
    FROM tickets t
    LEFT JOIN customers c ON t.customer_id = c.customer_id
    LEFT JOIN products p ON t.product_id = p.product_id
    WHERE t.ticket_id = ?
    """
    frame = query(sql, (ticket_id,))
    return None if frame.empty else frame.iloc[0].to_dict()


if __name__ == "__main__":
    build_database()
