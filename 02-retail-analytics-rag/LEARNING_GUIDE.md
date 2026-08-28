# Learning Guide — 55 / 35 / 10

## 55% — Understanding

Before changing code, explain this architecture from memory:

```text
STRUCTURED DATA
CSV -> SQLite -> SQL -> DataFrame -> exact answer / chart

UNSTRUCTURED DATA
TXT -> Load -> Split -> Embed -> Chroma -> Retrieve
    -> Context + Question -> LLM -> grounded answer
```

You must be able to answer:

1. Why should exact sales totals come from SQL instead of RAG?
2. Why is a line chart appropriate for monthly sales?
3. Why is a bar chart appropriate for category profit?
4. What does GROUP BY do?
5. Why does WHERE filter rows while HAVING filters groups?
6. What keys connect `orders` and `products`?
7. Why chunk documents before embedding?
8. Why must question and documents use compatible embeddings?
9. What does the retriever actually return?
10. What is an augmented prompt?
11. Why does the prompt contain an “I don't know” rule?
12. Why return sources?
13. Why can the LLM be swapped without changing the conceptual RAG pipeline?
14. Why are SQL and RAG separate tabs instead of using an autonomous agent?

If one answer is fuzzy, trace that part of the code.

## 35% — Hands-on Practice

### SQL
- Rebuild `high_volume_categories()` from scratch.
- Rebuild `product_profitability()` without looking.
- Add return rate by category.
- Add average delivery days by region.
- Add top five cities by net sales.
- Add categories with profit above a threshold using HAVING.

### Visualization
Build and explain:
- monthly sales line plot
- category profit bar plot
- region/category heatmap
- discount vs order-value scatter plot
- rating histogram

For every chart write:
**“What business question does this answer?”**

### RAG
- Change chunk size: 700 -> 400.
- Change top-k retrieval: 3 -> 1 -> 5.
- Inspect source documents.
- Ask an unanswerable question.
- Add a new policy file and verify retrieval.
- Ask a follow-up question that depends on memory.

## 10% — Memorization

Do not memorize full code.

Memorize anchors only:

### SQL
SELECT
FROM
WHERE
GROUP BY
HAVING
ORDER BY
COUNT / SUM / AVG
INNER JOIN ... ON

### Visualization
lineplot -> trend
barplot -> compare groups
heatmap -> patterns in a matrix
scatterplot -> relationship
histplot -> distribution

### RAG
Load -> Split -> Embed -> Store -> Retrieve -> Augment -> Generate
