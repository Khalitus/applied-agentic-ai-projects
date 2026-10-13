import sqlite3
from functools import lru_cache

import chromadb
import pandas as pd

from chromadb.utils import embedding_functions

from src.config import (
    DATABASE_PATH,
    KNOWLEDGE_DIR,
    ROOT_DIR,
    VECTOR_STORE_DIR,
)
from src.knowledge_documents import (
    build_case_records,
    build_policy_records,
)


SPLIT_PATH = ROOT_DIR / "data" / "splits" / "claim_splits.csv"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 64

@lru_cache(maxsize=1)
def get_embedding_function():
    import torch

    torch.set_num_threads(1)

    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        device="cpu",
    )

def load_source_data():
    with sqlite3.connect(DATABASE_PATH) as connection:
        claims = pd.read_sql_query(
            "SELECT * FROM claims",
            connection,
        )

    splits = pd.read_csv(SPLIT_PATH)

    return claims, splits

def rebuild_collection(client, name, records):
    if not records:
        raise ValueError(f"No records to index: {name}")

    existing = {
        item if isinstance(item, str) else item.name
        for item in client.list_collections()
    }

    if name in existing:
        client.delete_collection(name=name)

    collection = client.create_collection(
        name=name,
        embedding_function=get_embedding_function(),
    )

    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]

        collection.add(
            ids=[item["id"] for item in batch],
            documents=[item["document"] for item in batch],
            metadatas=[item["metadata"] for item in batch],
        )

    return collection

def main():
    claims, splits = load_source_data()

    case_records = build_case_records(claims, splits)
    policy_records = build_policy_records(KNOWLEDGE_DIR)

    print("\nKnowledge preparation")
    print(f"Historical cases: {len(case_records)}")
    print(f"Policy chunks: {len(policy_records)}")

    if len(case_records) != 840:
        raise ValueError(
            "Historical index must contain 840 training cases."
        )

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(VECTOR_STORE_DIR)
    )

    cases = rebuild_collection(
        client,
        "historical_claims",
        case_records,
    )

    policies = rebuild_collection(
        client,
        "claim_policies",
        policy_records,
    )

    print(f"Historical claims count: {cases.count()}")
    print(f"Claim policies count: {policies.count()}")

    if cases.count() != len(case_records) or policies.count() != len(policy_records):
        raise ValueError("Collection counts do not match record totals.")


if __name__ == "__main__":
    main()