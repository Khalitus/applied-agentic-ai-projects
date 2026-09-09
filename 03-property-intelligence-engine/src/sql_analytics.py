import pandas as pd

from src.database import get_connection


def run_query(query, params=()):
    with get_connection() as conn:
        return pd.read_sql_query(
            query,
            conn,
            params=params,
        )
    
def get_property_catalog(limit=10):
    query = """
        SELECT
            p.property_id,
            p.property_type,
            p.bedrooms,
            p.bathrooms,
            p.area_sqm,
            p.year_built,
            n.neighborhood_name,
            p.parking_spaces
        FROM properties AS p
        INNER JOIN neighborhoods AS n
            ON p.neighborhood_id = n.neighborhood_id
        ORDER BY p.area_sqm DESC
        LIMIT ?
    """

    return run_query(
        query,
        params=(limit,),
    )

def get_filtered_properties(
    property_type,
    min_bedrooms=1,
    limit=10,
):
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
        params=(
            property_type,
            min_bedrooms,
            limit,
        ),
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

        ORDER BY s.sale_date DESC

        LIMIT ?
    """

    return run_query(
        query,
        params=(limit,),
    )

def get_neighborhood_summary():
    query = """
        SELECT
            n.neighborhood_id,
            COUNT(p.property_id) AS property_count,
            AVG(p.area_sqm) AS avg_area_sqm,
            AVG(p.bedrooms) AS avg_bedrooms

        FROM neighborhoods AS n

        LEFT JOIN properties AS p
            ON n.neighborhood_id = p.neighborhood_id

        GROUP BY n.neighborhood_id

        ORDER BY property_count DESC
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
                    ORDER BY s.sale_date DESC
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

        INNER JOIN properties as p
            ON rs.property_id = p.property_id

        INNER JOIN neighborhoods as n
            ON p.neighborhood_id = n.neighborhood_id

        WHERE sale_rank = 1
        ORDER BY sale_date DESC
        LIMIT ?
    """

    return run_query(
        query,
        params=(limit,),
    )

def get_neighborhood_price_analytics(limit=20):
    query = """
        WITH ranked_sales AS (
            SELECT
                s.property_id,
                s.sale_date,
                s.sale_price,

                ROW_NUMBER() OVER (
                    PARTITION BY s.property_id
                    ORDER BY s.sale_date DESC
                ) AS sale_rank

            FROM sales AS s
        ),

        latest_sales AS (
            SELECT
                property_id,
                sale_date,
                sale_price
            FROM ranked_sales
            WHERE sale_rank = 1
        )

        SELECT
            p.property_id,
            p.property_type,
            n.neighborhood_name,
            ls.sale_price,
             
            RANK() OVER (
                PARTITION BY n.neighborhood_name
                ORDER BY ls.sale_price DESC
            ) AS price_rank,           

            ROUND(
                AVG(ls.sale_price)
                OVER (
                    PARTITION BY n.neighborhood_name
                ),
                2
            ) AS neighborhood_avg_price,

            ROUND(
                ls.sale_price - AVG(ls.sale_price)
                OVER(
                    PARTITION BY n.neighborhood_name
                ),
                2
            ) AS price_difference_from_avg
            
        FROM properties AS p

        INNER JOIN neighborhoods AS n
            ON p.neighborhood_id = n.neighborhood_id

        INNER JOIN latest_sales AS ls
            ON p.property_id = ls.property_id

        ORDER BY
            n.neighborhood_name,
            ls.sale_price DESC

        LIMIT ?
    """

    return run_query(
        query,
        params=(limit,),
    )