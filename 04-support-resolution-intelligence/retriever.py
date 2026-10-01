from collections import OrderedDict

from rag_index import load_index, load_parent_documents


def baseline_search(query, k=4):
    store = load_index()
    return store.similarity_search(query, k=k)


def parent_aware_search(query, parent_k=4, child_k=12):
    if child_k < parent_k:
        child_k = parent_k
    store = load_index()
    parents = load_parent_documents()

    parent_lookup = {
        parent.metadata["parent_id"]: parent
        for parent in parents
    }

    child_results = baseline_search(query, child_k)

    selected_parents = []
    seen_parent_ids = set()

    for child in child_results:
        parent_id = child.metadata["parent_id"]

        if parent_id in seen_parent_ids:
            continue

        parent = parent_lookup.get(parent_id)

        if parent is None:
            continue

        selected_parents.append(parent)
        seen_parent_ids.add(parent_id)

        if len(selected_parents) == parent_k:
            break

    return selected_parents


def mmr_search(query, k=4, fetch_k=12):
    store = load_index()
    return store.max_marginal_relevance_search(
        query,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=0.6,
    )

def retrieve(query, strategy="mmr", k=4):
    strategy = strategy.lower()
    if strategy == "baseline":
        return baseline_search(query, k=k)

    if strategy == "mmr":
        return mmr_search(query, k=k)

    if strategy == "parent":
        return parent_aware_search(query, parent_k=k)

    raise ValueError(
        f"Unknown retrieval strategy: {strategy}"
    )