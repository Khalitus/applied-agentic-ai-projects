# Campaign Intelligence Agent

SQL-backed campaign analysis, pre-contact subscription prediction, and optional LLM-assisted interpretation using the [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing).

The application combines historical campaign aggregates with a scikit-learn classifier and a LangGraph workflow. Historical analysis and model inference run locally; an optional Groq API integration produces short natural-language explanations of the returned evidence.

## Capabilities

- **Historical analytics:** ingest source records into SQLite and examine subscription counts and rates by occupation, education, or previous campaign outcome.
- **Predictive inference:** validate user-supplied customer attributes and estimate term-deposit subscription probability with a fitted Logistic Regression pipeline.
- **Workflow orchestration:** route historical analysis and prediction requests through explicit LangGraph state transitions.
- **Interpretation:** optionally summarize SQL or model outputs with Groq, with a deterministic fallback when API access is unavailable.
- **Interface:** access historical analytics and custom predictions through separate Gradio tabs.

This repository provides a baseline implementation, not a production targeting service or an independently validated multi-agent system. The LLM interpretation is not independently fact-checked by a separate verification step.

## Technical stack

Python 3.12, Pandas, SQLite, scikit-learn, Pydantic, LangGraph, LangChain-Groq, Gradio, and pytest.

## Data

**Source:** UCI Machine Learning Repository, *Bank Marketing*, dataset 222 (Moro, Rita, Cortez), DOI [10.24432/C5K306](https://doi.org/10.24432/C5K306), CC BY 4.0.

The application uses `bank-full.csv` (45,211 historical records, 16 source predictors and a binary target). The source data is downloaded from UCI rather than checked into Git.

Predictions use 11 attributes assumed to be available before a new marketing contact. Features tied to the current or completed call, including `duration`, are excluded. See [data card](docs/data_card.md) for availability assumptions, split limitations, and responsible-use considerations.

## Getting started

Use Python 3.12. From this directory:

```bash
python -m pip install -r requirements.txt
cp .env.example .env
python -m scripts.fetch_data
python -m scripts.ingest_data
python -m src.ml
python -m pytest -q
python app.py
```

Open the local URL displayed by Gradio (normally `http://127.0.0.1:7860`).

`GROQ_API_KEY` is optional. Leave it empty for local-only analytics and prediction. When configured, review your provider's current model access and rate limits; the model name can be changed through `GROQ_MODEL`.

### Primary commands

| Command | Result |
|---|---|
| `python -m scripts.fetch_data` | Downloads the official CSV to `data/raw/` |
| `python -m scripts.ingest_data` | Builds a local SQLite database from the downloaded CSV |
| `python -m src.ml` | Trains the baseline model and writes metrics to `reports/` |
| `python app.py` | Starts the Gradio interface |
| `python -m pytest -q` | Runs automated checks |
| `sqlite3 data/campaigns.sqlite3 < sql/analysis.sql` | Runs sample queries using the SQLite CLI |

The trained model and SQLite database are generated locally and are excluded from version control. Evaluation reports and verified screenshots may be committed when available.

## Architecture

```text
UCI CSV ──┬──> SQLite ──────────────> Historical SQL analysis ─┐
          └──> Model training ──────> Customer prediction ─────┤
                                                                v
Gradio input ──> LangGraph routing ──> Structured evidence ──> Optional narration
                                                                |
                                                                v
                                                    Gradio results + evidence
```

LangGraph routes explicitly selected requests to deterministic SQL or ML functions. The LLM does not calculate subscription probabilities or execute generated SQL. See [architecture](docs/architecture.md) for component boundaries.

## Evaluation and limitations

The baseline uses a source-order 70/15/15 train/validation/holdout split and a fixed 0.5 decision threshold. Preprocessing is fitted exclusively to training observations. `src.ml` calculates classification and probability-based metrics; **this README does not claim an achieved score before execution on the original dataset**.

Source-order splitting is a temporal proxy, not a verified time- and customer-independent evaluation. The dataset lacks stable customer identifiers and complete timestamps. The historical source population may differ substantially from present-day banking customers. A prediction is neither causal evidence nor a guaranteed customer outcome. No outreach action or financial decision is automated.

See the [model card](docs/model_card.md) and [data card](docs/data_card.md) for evaluation scope and known risks.

## Repository structure

```text
app.py                   Gradio user interface
src/                     Feature definitions, validation, training, inference, LangGraph
scripts/                 Dataset retrieval and SQLite ingestion
sql/                     Reproducible SQL examples
tests/                   Automated tests
docs/                    Architecture, data card, model card
data/raw/                Locally downloaded data (Git-ignored)
artifacts/               Locally generated model (Git-ignored)
reports/                 Evaluation outputs
screenshots/             Verified application screenshots
```
