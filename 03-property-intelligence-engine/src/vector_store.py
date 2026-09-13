from pathlib import Path

import chromadb
import pandas as pd
from chromadb.utils import embedding_functions

from src.sql_analytics import run_query


BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_PATH = BASE_DIR / "data" / "chroma_db"
COLLECTION_NAME = "properties"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


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
    furnished_text = "furnished" if row["furnished"] else "unfurnished"

    return (
        f"{row['property_type']} in {row['neighborhood_name']} with "
        f"{row['bedrooms']} bedrooms, {row['bathrooms']} bathrooms and "
        f"{row['area_sqm']} square meters of area. "
        f"It has {row['parking_spaces']} parking spaces and is "
        f"{row['distance_to_center_km']} km from the city center. "
        f"The neighborhood has a school score of {row['school_score']} out of 10, "
        f"a transit score of {row['transit_score']} out of 10 and "
        f"a safety score of {row['safety_score']} out of 10. "
        f"The property is on floor {row['floor']} and is {furnished_text}."
    )


def prepare_property_documents(data):
    ids = []
    documents = []
    metadatas = []

    for _, row in data.iterrows():
        ids.append(str(row["property_id"]))
        documents.append(build_property_document(row))

        metadatas.append({
            "property_id": str(row["property_id"]),
            "property_type": str(row["property_type"]),
            "neighborhood_name": str(row["neighborhood_name"]),
            "bedrooms": int(row["bedrooms"]),
            "bathrooms": int(row["bathrooms"]),
            "area_sqm": float(row["area_sqm"]),
            "parking_spaces": int(row["parking_spaces"]),
            "sale_price": float(row["sale_price"]),
            "school_score": float(row["school_score"]),
            "transit_score": float(row["transit_score"]),
            "safety_score": float(row["safety_score"]),
        })

    return ids, documents, metadatas


def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )


def get_chroma_client():
    return chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )


def get_property_collection():
    client = get_chroma_client()

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
    )


def build_property_index():
    data = load_property_search_data()
    ids, documents, metadatas = prepare_property_documents(data)
    collection = get_property_collection()

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )

    return collection


def get_index_summary():
    collection = get_property_collection()

    sample = collection.get(
        limit=1,
        include=["documents", "metadatas"],
    )

    return collection.count(), sample


def build_search_filter(
    property_type=None,
    min_bedrooms=None,
    max_price=None,
):
    conditions = []

    if property_type is not None:
        conditions.append({
            "property_type": {"$eq": property_type}
        })

    if min_bedrooms is not None:
        conditions.append({
            "bedrooms": {"$gte": min_bedrooms}
        })

    if max_price is not None:
        conditions.append({
            "sale_price": {"$lte": max_price}
        })

    if not conditions:
        return None

    if len(conditions) == 1:
        return conditions[0]

    return {"$and": conditions}


def format_search_results(results):
    records = []

    for property_id, document, metadata, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        records.append({
            "property_id": property_id,
            "property_type": metadata["property_type"],
            "neighborhood_name": metadata["neighborhood_name"],
            "bedrooms": metadata["bedrooms"],
            "sale_price": metadata["sale_price"],
            "distance": distance,
            "document": document,
        })

    return pd.DataFrame(records)


def semantic_search(
    query,
    n_results=5,
    property_type=None,
    min_bedrooms=None,
    max_price=None,
):
    collection = get_property_collection()

    where = build_search_filter(
        property_type=property_type,
        min_bedrooms=min_bedrooms,
        max_price=max_price,
    )

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    return format_search_results(results)