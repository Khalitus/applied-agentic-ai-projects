# Urban Mobility Demand Intelligence

A learning-oriented applied AI project combining UCI hourly bike-sharing data,
SQLite, Pandas, XGBoost and an interactive Gradio application. The single-agent
tool-calling flow is intentionally left as an implementation exercise.

## Data attribution

Original dataset: Fanaee-T, H. (2013), *Bike Sharing*, UCI Machine Learning
Repository. https://doi.org/10.24432/C5W894 (CC BY 4.0).
Historical Capital Bikeshare hourly activity, 2011–2012.

The original CSV is downloaded from UCI, not generated. This repository does not
commit downloaded raw data, secrets, trained models or generated reports.

## Setup (from the repository root)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m scripts.download_data
python -m src.db
python -m src.train
python app.py
```

Then open the local URL printed by Gradio.

For tests: `python -m pytest -q`.

## What works in the scaffold

- Authenticated-by-source UCI downloader (no synthetic dataset)
- Data integrity checks and SQLite ingestion
- Parameterized hourly SQL query
- Chronological 80/20 XGBoost regression baseline with mean benchmark
- User-submitted scenario prediction + reference historical hourly average
- Local Gradio interface
- Unit tests for feature generation and SQLite query

## What you implement during the project

- Richer data profiling and analytical SQL tools
- Proper model error analysis, feature importance, evaluation improvements
- Agent tool schemas, LLM-driven tool selection, validated dispatch
- Grounded multi-tool answers, tests, documentation and screenshots

## Important limitations

- This is an educational historical-scenario estimator, not a current operations forecast.
- The model uses 2011–2012 data without real-time weather, inventory or station locations.
- Never use `casual` or `registered` as prediction features because `cnt` is their sum.
- Date uses month and day-of-week; it is not a trained projection for future years.
- A date labeled 'holiday' must be confirmed by the user; the app does not fetch a
  live public holiday calendar.
- A global hourly model cannot recommend actual station-level bike repositioning.
- Gradio's future AI narrative must be grounded in SQL/model outputs, not invented.

See `docs/PROJECT_BLUEPRINT.md` for the full learning plan.
