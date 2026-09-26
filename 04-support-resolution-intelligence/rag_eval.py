import pandas as pd

from config import EVAL_DIR
from retriever import advanced_search


def reciprocal_rank(expected_source, retrieved_sources):
    for rank, source in enumerate(retrieved_sources, start=1):
        if source == expected_source:
            return 1.0 / rank
    return 0.0


def hit_at_k(expected_source, retrieved_sources):
    return float(expected_source in retrieved_sources)


def evaluate(mode, k=4):
    evaluation = pd.read_csv(EVAL_DIR / "retrieval_eval.csv")
    rows = []

    for item in evaluation.to_dict("records"):
        docs = advanced_search(item["query"], mode=mode, k=k)
        sources = [doc.metadata.get("source") for doc in docs]

        rows.append({
            "query_id": item["query_id"],
            "expected_source": item["expected_source"],
            "retrieved_sources": sources,
            "hit_at_k": hit_at_k(item["expected_source"], sources),
            "reciprocal_rank": reciprocal_rank(item["expected_source"], sources),
        })

    results = pd.DataFrame(rows)
    print(results[["query_id", "expected_source", "hit_at_k", "reciprocal_rank"]])
    print(f"\n{mode} Recall@{k}: {results['hit_at_k'].mean():.3f}")
    print(f"{mode} MRR: {results['reciprocal_rank'].mean():.3f}")
    return results


if __name__ == "__main__":
    # TODO Task 8: evaluate baseline, MMR, and parent-aware retrieval,
    # then compare the metrics instead of declaring one "better" by intuition.
    evaluate("baseline")
