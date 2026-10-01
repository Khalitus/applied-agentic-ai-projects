import pandas as pd

from config import EVAL_DIR
from retriever import retrieve

def get_sources(documents):
    return [
        doc.metadata.get("source")
        for doc in documents
    ]

def reciprocal_rank(expected_source, retrieved_sources):
    for rank, source in enumerate(retrieved_sources, start=1):
        if source == expected_source:
            return 1.0 / rank
    return 0.0


def hit_at_k(expected_source, retrieved_sources):
    return float(expected_source in retrieved_sources)


def evaluate(strategy, k=4):
    evaluation = pd.read_csv(EVAL_DIR / "retrieval_eval.csv")
    rows = []

    for item in evaluation.to_dict("records"):
        docs = retrieve(item["query"], strategy=strategy, k=k)
        sources = get_sources(docs)

        rows.append({
            "query_id": item["query_id"],
            "expected_source": item["expected_source"],
            "retrieved_sources": sources,
            "hit_at_k": hit_at_k(item["expected_source"], sources),
            "reciprocal_rank": reciprocal_rank(item["expected_source"], sources),
        })

    results = pd.DataFrame(rows)
    print(results[["query_id", "expected_source", "hit_at_k", "reciprocal_rank"]])
    print(f"\n{strategy} Recall@{k}: {results['hit_at_k'].mean():.3f}")
    print(f"{strategy} MRR: {results['reciprocal_rank'].mean():.3f}")
    return results

def compare_strategies(k=4):
    rows = []

    for strategy in ["baseline", "mmr", "parent"]:
        results = evaluate(strategy, k=k)

        rows.append({
            "strategy": strategy,
            "recall_at_k": results["hit_at_k"].mean(),
            "mrr": results["reciprocal_rank"].mean(),
        })

    comparison = pd.DataFrame(rows)

    print("\nRetrieval strategy comparison:")
    print(comparison.to_string(index=False))

    return comparison

if __name__ == "__main__":
    compare_strategies()
