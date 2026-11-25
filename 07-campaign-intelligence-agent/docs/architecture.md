# System architecture

## Components

| Component | Responsibility |
|---|---|
| Dataset ingestion | Fetch official UCI source data and import records into SQLite |
| SQL analytics | Aggregate subscription rates through allowlisted group-by dimensions |
| Model training | Fit preprocessing and Logistic Regression on training observations only |
| Inference | Validate customer input and apply the saved scikit-learn pipeline |
| LangGraph | Route requests to the SQL or prediction branch |
| Narration | Optionally explain existing SQL/ML evidence through Groq |
| Gradio | Accept user input and display structured evidence and interpretation |

## Execution flow

```text
                                  ┌──> SQL analysis ──────┐
Gradio ──> LangGraph supervisor ──┤                       ├──> Narration ──> Response
                                  └──> ML prediction ────┘

UCI CSV ──> SQLite ───────────────> SQL analysis
       └──> Training pipeline ────> Saved model ──> ML prediction
```

The two LangGraph branches are selected deterministically from the interface operation. Each invokes application code and returns structured evidence. There is no independent LLM supervisor, autonomous tool discovery, human-approval checkpoint, or separate evidence-verification agent in the current implementation.

## Data integrity and execution boundaries

- The historical SQL endpoint does not execute model-written SQL. Group-by columns come from a predefined allowlist and values use bound parameters.
- The predictor accepts an explicit Pydantic schema; unknown fields are rejected.
- Training uses 11 pre-contact features and excludes source variables tied to the completed call.
- LLM narration is an optional presentation layer. API failure falls back to the underlying evidence.
- No marketing messages, external write operations, or customer decisions are triggered by the graph.
- Model files are deserialized with joblib; only load artifacts produced by the local trusted training process.

## Operational constraints

This application operates on historical public data and local files. It is not hardened for public deployment, external authentication, sensitive customer data, or banking operations. The optional LLM output is prompt-constrained but not independently verified and should be checked against the structured evidence.
