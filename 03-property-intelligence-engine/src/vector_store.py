from src.sql_analytics import run_query

def load_property_search_data():
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
            p.property_type,
            p.bedrooms,
            p.bathrooms,
            p.area_sqm,
            p.parking_spaces,
            p.floor,
            p.furnished,
            n.neighborhood_name,
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

def build_property_document(row):
    furnished_text = (
        "furnished"
        if row["furnished"]
        else "unfurnished"
    )

    document = (
        f"{row['property_type']} in {row['neighborhood_name']} with "
        f"{row['bedrooms']} bedrooms, "
        f"{row['bathrooms']} bathrooms and "
        f"{row['area_sqm']} square meters area. "
        f"It has {row['parking_spaces']} parking space(s), and "
        f"is {row['distance_to_center_km']}km away from center of the city. "
        f"The neighborhood has following scores: "
        f"School quality: {row['school_score']} out of 10, "
        f"Transit quality: {row['transit_score']} out of 10, "
        f"and Safety quality: {row['safety_score']} out of 10. "
        f"The property is on floor {row['floor']} . "
        f"The property is {furnished_text}."
    )

    return document

def prepare_property_documents(data):
    documents = []
    metadatas = []
    ids = []
    
    for _, row in data.iterrows():
        ids.append(str(row["property_id"]))
        documents.append(build_property_document(row))
        
        metadatas.append({
            "property_id": str(row["property_id"]),
            "property_type": str(row["property_type"]),
            "neighborhood_name": str(row["neighborhood_name"]), 
            "bedrooms": int(row["bedrooms"]),
            "bathrooms": int(row["bathrooms"]),
            "area_sqm": int(row["area_sqm"]),
            "parking_spaces": int(row["parking_spaces"]),
            "sale_price": float(row["sale_price"]),
            "school_score": float(row["school_score"]),
            "transit_score": float(row["transit_score"]),
            "safety_score": float(row["safety_score"])
        })
        
    return ids, documents, metadatas
