from src.database import run_query

checks = {
    "orders": "SELECT COUNT(*) AS n FROM orders",
    "products": "SELECT COUNT(*) AS n FROM products",
    "duplicate_order_ids": """
        SELECT COUNT(*) AS n
        FROM (
            SELECT order_id
            FROM orders
            GROUP BY order_id
            HAVING COUNT(*) > 1
        )
    """,
    "regions": "SELECT COUNT(DISTINCT region) AS n FROM orders",
}

print("PROJECT DATA CHECK")
print("=" * 40)

for name, query in checks.items():
    value = run_query(query).iloc[0, 0]
    print(f"{name:22s}: {value}")

print("\nExpected:")
print("orders                : 900")
print("products              : 12")
print("duplicate_order_ids   : 0")
print("regions               : 4")
