from collections import OrderedDict

from rag_index import load_index, load_parent_documents


def baseline_search(query, k=4):
    store = load_index()
    return store.similarity_search(query, k=k)


def parent_aware_search(query, child_k=10, parent_k=4):
    """Retrieve child chunks, then collapse duplicate children back to parent documents."""
    store = load_index()
    child_hits = store.similarity_search(query, k=child_k)

    # TODO Task 7:
    # 1. collect unique parent_id values in retrieval order
    # 2. map parent_id -> full parent document from load_parent_documents()
    # 3. return the first parent_k unique parents
    raise NotImplementedError


def mmr_search(query, k=4, fetch_k=12):
    store = load_index()
    return store.max_marginal_relevance_search(
        query,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=0.6,
    )


def advanced_search(query, mode="parent", k=4):
    if mode == "baseline":
        return baseline_search(query, k=k)
    if mode == "mmr":
        return mmr_search(query, k=k)
    if mode == "parent":
        return parent_aware_search(query, parent_k=k)
    raise ValueError(f"Unknown retriever mode: {mode}")
