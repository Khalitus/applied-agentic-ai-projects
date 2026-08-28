from src.database import run_query

def recent_south_orders(limit: int = 10):
    """SELECT + WHERE + ORDER BY — working example."""
    query = """
        SELECT
            order_id,
            order_date,
            city,
            product_id,
            quantity,
            unit_price
        FROM orders
        WHERE region = 'South'
        ORDER BY order_date DESC
        LIMIT ?
    """
    return run_query(query, (limit,))

def sales_by_region():
    """JOIN + GROUP BY + SUM — working example."""
    query = """
        SELECT
            o.region,
            COUNT(*) AS total_orders,
            ROUND(SUM(o.quantity * o.unit_price * (1 - o.discount_pct)), 2) AS net_sales,
            ROUND(SUM(
                o.quantity * o.unit_price * (1 - o.discount_pct)
                - o.quantity * p.unit_cost
            ), 2) AS profit
        FROM orders AS o
        INNER JOIN products AS p
            ON o.product_id = p.product_id
        GROUP BY o.region
        ORDER BY net_sales DESC
    """
    return run_query(query)

def high_volume_categories():
    """
    HANDS-ON:
    Rebuild this query yourself after running it once.
    Concepts: JOIN + GROUP BY + HAVING + ORDER BY.
    """
    query = """
        SELECT
            p.category,
            COUNT(*) AS order_lines,
            SUM(o.quantity) AS units_sold
        FROM orders AS o
        INNER JOIN products AS p
            ON o.product_id = p.product_id
        GROUP BY p.category
        HAVING COUNT(*) > 100
        ORDER BY units_sold DESC
    """
    return run_query(query)

def product_profitability():
    """
    HANDS-ON:
    Rebuild without looking at the query.
    """
    query = """
        SELECT
            p.product_name,
            p.category,
            ROUND(SUM(o.quantity * o.unit_price * (1 - o.discount_pct)), 2) AS net_sales,
            ROUND(SUM(
                o.quantity * o.unit_price * (1 - o.discount_pct)
                - o.quantity * p.unit_cost
            ), 2) AS profit
        FROM orders AS o
        INNER JOIN products AS p
            ON o.product_id = p.product_id
        GROUP BY p.product_id, p.product_name, p.category
        ORDER BY profit DESC
    """
    return run_query(query)

def data_quality_summary():
    query = """
        SELECT
            SUM(CASE WHEN customer_segment IS NULL THEN 1 ELSE 0 END) AS missing_segments,
            SUM(CASE WHEN rating IS NULL THEN 1 ELSE 0 END) AS missing_ratings
        FROM orders
    """
    return run_query(query)
