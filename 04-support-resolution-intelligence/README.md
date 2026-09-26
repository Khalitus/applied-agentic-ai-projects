# Support Resolution Intelligence — Starter

Month 2 / Project 4

A learning project that combines:
- relational data + SQL
- Pandas
- preprocessing pipelines
- cross-validation
- XGBoost
- leakage prevention
- embeddings + Chroma
- baseline, MMR, and parent-aware retrieval
- retrieval evaluation
- grounded RAG generation
- Gradio

## Learning split

- 55% understanding: why each design choice exists
- 35% hands-on: you implement the important joins, preprocessing, retrieval logic, evaluation comparison, prompt wiring, and integration
- 10% memorization: only recurring patterns such as train/test separation, fit/transform discipline, retriever inputs/outputs, and basic SQL syntax

## Architecture

Structured path:
CSV -> SQLite -> SQL JOIN -> preprocessing pipeline -> XGBoost -> escalation probability

Knowledge path:
Policies + historical resolved cases -> parent docs -> child chunks -> embeddings -> Chroma -> advanced retriever -> LLM -> grounded guidance

Application:
Ticket ID + question -> ML risk + RAG guidance + source evidence

## Tasks

1. Environment and data audit
2. SQL analytics
3. Modeling dataset JOIN + leakage audit
4. Preprocessing pipeline
5. XGBoost + cross-validation + test evaluation
6. RAG ingestion and chunking
7. Advanced retrieval
8. Retrieval evaluation
9. Grounded answer generation
10. Gradio integration and final test scenarios

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python database.py
```

Then complete tasks in order. Do not build the vector index until Task 6 and do not run the app until Tasks 4–10 are complete.

## Important leakage rule

The escalation model predicts risk at ticket intake. Therefore do not train on:
- first_response_minutes
- resolution_hours
- resolution_text

These values occur after intake and would leak future information into the model.

## Synthetic data

- `customers.csv`: customer-level structured features
- `products.csv`: product catalog
- `tickets.csv`: support ticket history and escalation label
- `resolved_cases.csv`: historical cases for retrieval
- policy `.md` files: support knowledge base
- `retrieval_eval.csv`: small labeled retrieval benchmark

All data is synthetic.

## Optional extensions after the core project

Only add these after Tasks 1–10 work correctly:

1. Multi-query retrieval: generate 3 query variants, retrieve for each, deduplicate, then compare against baseline.
2. Metadata-aware retrieval: infer filters such as `issue_type` or `source_type` and apply them before/while retrieving.
3. FAISS comparison: build a second index and compare retrieval quality and operational trade-offs with Chroma.
4. Threshold tuning: choose an escalation threshold from validation data instead of assuming 0.50.
5. Error analysis: inspect the worst ML false negatives and worst RAG retrieval misses.
