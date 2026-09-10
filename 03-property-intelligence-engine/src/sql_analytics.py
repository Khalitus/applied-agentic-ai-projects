import pandas as pd

from src.database import get_connection


def run_query(query, params=()):
    with get_connection() as conn:
        return pd.read_sql_query(query, conn, params=params)


def get_property_catalog(limit=10):
    query = """
        SELECT
            p.property_id,
            p.property_type,
            p.bedrooms,
            p.bathrooms,
            p.area_sqm,
            p.year_built,
            p.parking_spaces,
            n.neighborhood_name
        FROM properties AS p
        INNER JOIN neighborhoods AS n
            ON p.neighborhood_id = n.neighborhood_id
        ORDER BY p.area_sqm DESC
        LIMIT ?
    """

    return run_query(query, params=(limit,))


def get_filtered_properties(property_type, min_bedrooms=1, limit=10):
    query = """
        SELECT
            p.property_id,
            p.property_type,
            p.bedrooms,
            p.bathrooms,
            p.area_sqm,
            p.parking_spaces,
            n.neighborhood_name,
            n.distance_to_center_km
        FROM properties AS p
        INNER JOIN neighborhoods AS n
            ON p.neighborhood_id = n.neighborhood_id
        WHERE
            p.property_type = ?
            AND p.bedrooms >= ?
        ORDER BY p.area_sqm DESC
        LIMIT ?
    """

    return run_query(
        query,
        params=(property_type, min_bedrooms, limit),
    )


def get_recent_sales(limit=10):
    query = """
        SELECT
            s.sale_id,
            s.sale_date,
            s.sale_price,
            p.property_id,
            p.property_type,
            p.bedrooms,
            p.area_sqm,
            n.neighborhood_name
        FROM sales AS s
        INNER JOIN properties AS p
            ON s.property_id = p.property_id
        INNER JOIN neighborhoods AS n
            ON p.neighborhood_id = n.neighborhood_id
        ORDER BY
            s.sale_date DESC,
            s.sale_id DESC
        LIMIT ?
    """

    return run_query(query, params=(limit,))


def get_neighborhood_summary():
    query = """
        SELECT
            n.neighborhood_name,
            COUNT(p.property_id) AS property_count,
            ROUND(AVG(p.area_sqm), 1) AS avg_area_sqm,
            ROUND(AVG(p.bedrooms), 2) AS avg_bedrooms
        FROM neighborhoods AS n
        LEFT JOIN properties AS p
            ON n.neighborhood_id = p.neighborhood_id
        GROUP BY
            n.neighborhood_id,
            n.neighborhood_name
        ORDER BY
            property_count DESC,
            n.neighborhood_name
    """

    return run_query(query)


def get_latest_property_sales(limit=20):
    query = """
        WITH ranked_sales AS (
            SELECT
                s.sale_id,
                s.property_id,
                s.sale_date,
                s.sale_price,
                ROW_NUMBER() OVER (
                    PARTITION BY s.property_id
                    ORDER BY
                        s.sale_date DESC,
                        s.sale_id DESC
                ) AS sale_rank
            FROM sales AS s
        )
        SELECT
            p.property_id,
            p.property_type,
            p.bedrooms,
            p.area_sqm,
            n.neighborhood_name,
            rs.sale_date,
            rs.sale_price
        FROM ranked_sales AS rs
        INNER JOIN properties AS p
            ON rs.property_id = p.property_id
        INNER JOIN neighborhoods AS n
            ON p.neighborhood_id = n.neighborhood_id
        WHERE rs.sale_rank = 1
        ORDER BY
            rs.sale_date DESC,
            rs.sale_id DESC
        LIMIT ?
    """

    return run_query(query, params=(limit,))


def get_neighborhood_price_analytics(limit=20):
    query = """
        WITH ranked_sales AS (
            SELECT
                s.sale_id,
                s.property_id,
                s.sale_date,
                s.sale_price,
                ROW_NUMBER() OVER (
                    PARTITION BY s.property_id
                    ORDER BY
                        s.sale_date DESC,
                        s.sale_id DESC
                ) AS sale_rank
            FROM sales AS s
        )
        SELECT
            p.property_id,
            p.property_type,
            n.neighborhood_name,
            rs.sale_price,
            ROUND(
                AVG(rs.sale_price) OVER (
                    PARTITION BY n.neighborhood_id
                ),
                2
            ) AS neighborhood_avg_price,
            RANK() OVER (
                PARTITION BY n.neighborhood_id
                ORDER BY rs.sale_price DESC
            ) AS price_rank,
            ROUND(
                rs.sale_price - AVG(rs.sale_price) OVER (
                    PARTITION BY n.neighborhood_id
                ),
                2
            ) AS price_difference_from_avg
        FROM properties AS p
        INNER JOIN neighborhoods AS n
            ON p.neighborhood_id = n.neighborhood_id
        INNER JOIN ranked_sales AS rs
            ON p.property_id = rs.property_id
        WHERE rs.sale_rank = 1
        ORDER BY
            n.neighborhood_name,
            rs.sale_price DESC
        LIMIT ?
    """

    return run_query(query, params=(limit,))


def get_sale_history_growth(limit=30):
    query = """
        WITH sale_history AS (
            SELECT
                s.sale_id,
                s.property_id,
                s.sale_date,
                s.sale_price,
                LAG(s.sale_price) OVER (
                    PARTITION BY s.property_id
                    ORDER BY
                        s.sale_date,
                        s.sale_id
                ) AS previous_sale_price
            FROM sales AS s
        )
        SELECT
            sh.sale_id,
            sh.property_id,
            p.property_type,
            n.neighborhood_name,
            sh.sale_date,
            sh.sale_price,
            sh.previous_sale_price,
            ROUND(
                sh.sale_price - sh.previous_sale_price,
                2
            ) AS price_change,
            ROUND(
                (sh.sale_price - sh.previous_sale_price)
                / sh.previous_sale_price * 100.0,
                2
            ) AS price_change_pct
        FROM sale_history AS sh
        INNER JOIN properties AS p
            ON sh.property_id = p.property_id
        INNER JOIN neighborhoods AS n
            ON p.neighborhood_id = n.neighborhood_id
        ORDER BY
            sh.property_id,
            sh.sale_date,
            sh.sale_id
        LIMIT ?
    """

    return run_query(query, params=(limit,))