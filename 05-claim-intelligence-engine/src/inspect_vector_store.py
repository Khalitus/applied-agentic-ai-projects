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

    if policy_stored["documents"]:
        print("\nSample policy document 1:")
        print(policy_stored["documents"][0])
        print("\nSample policy metadata 1:")
        print(policy_stored["metadatas"][0])

        print("\nSample policy document 2:")
        print(policy_stored["documents"][1])
        print("\nSample policy metadata 2:")
        print(policy_stored["metadatas"][1])

        print("\nSample policy document 3:")
        print(policy_stored["documents"][2])
        print("\nSample policy metadata 3:")
        print(policy_stored["metadatas"][2])

if __name__ == "__main__":
    main()