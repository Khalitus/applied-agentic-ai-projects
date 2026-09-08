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