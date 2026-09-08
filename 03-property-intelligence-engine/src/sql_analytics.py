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