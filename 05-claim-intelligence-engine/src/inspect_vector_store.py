import chromadb
import pandas as pd

from src.config import ROOT_DIR, VECTOR_STORE_DIR


SPLIT_PATH = ROOT_DIR / "data" / "splits" / "claim_splits.csv"


def main():
    splits = pd.read_csv(SPLIT_PATH)

    client = chromadb.PersistentClient(
        path=str(VECTOR_STORE_DIR)
    )

    cases = client.get_collection("historical_claims")
    policies = client.get_collection("claim_policies")

    print("\nVector store inspection")
    print(f"Historical cases: {cases.count()}")
    print(f"Policy chunks: {policies.count()}")

    stored = cases.get(
        include=["documents", "metadatas"]
    )

    indexed_ids = {
        metadata["claim_id"]
        for metadata in stored["metadatas"]
    }

    expected_ids = set(
        splits.loc[
            splits["split"] == "train",
            "claim_id",
        ]
    )

    if indexed_ids != expected_ids:
        raise ValueError("Indexed IDs do not match expected IDs.")

    if cases.count() != 840:
        raise ValueError("Expected exactly 840 historical claims.")

    if stored["documents"]:
        print("\nSample historical document:")
        print(stored["documents"][0])
        print("\nSample historical metadata:")
        print(stored["metadatas"][0])

    policy_stored = policies.get(
        include=["documents", "metadatas"]
    )

    if not policy_stored["documents"]:
        raise ValueError("Policy collection is empty.")

    print("\nIndexed policy sources")

    source_files = set()

    for document, metadata in zip(
        policy_stored["documents"],
        policy_stored["metadatas"],
    ):
        source_files.add(metadata["source_file"])

        print(f"\nSource: {metadata['source_file']}")
        print(f"Chunk: {metadata['chunk_index']}")
        print(document)

    for source in sorted(source_files):
        print(f"Indexed: {source}")

if __name__ == "__main__":
    main()