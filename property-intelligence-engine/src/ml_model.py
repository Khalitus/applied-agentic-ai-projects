from src.sql_analytics import run_query


def load_modeling_dataset():
    query = """
        WITH ranked_sales AS (
            SELECT
                s.property_id,
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
            p.bedrooms,
            p.bathrooms,
            p.area_sqm,
            p.year_built,
            p.parking_spaces,
            p.floor,
            p.furnished,
            n.distance_to_center_km,
            n.school_score,
            n.transit_score,
            n.safety_score,
            rs.sale_price

        FROM properties AS p

        INNER JOIN neighborhoods AS n
            ON p.neighborhood_id = n.neighborhood_id

        INNER JOIN ranked_sales AS rs
            ON p.property_id = rs.property_id

        WHERE rs.sale_rank = 1

        ORDER BY p.property_id
    """

    return run_query(query)