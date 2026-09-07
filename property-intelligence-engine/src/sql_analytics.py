from src.database import get_connection

def get_sample_properties(limit=5):
    query = '''
        SELECT
            property_id,
            property_type,
            bedrooms,
            bathrooms,
            area_sqm
        FROM properties
        LIMIT ?
    '''

    with get_connection() as conn:
        return conn.execute(query, (limit,)).fetchall()
