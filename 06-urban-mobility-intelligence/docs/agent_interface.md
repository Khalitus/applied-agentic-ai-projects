# Interactive Agent and Historical Analytics Interface

## Scope

The application exposes three independent Gradio tabs: a user-supplied hourly demand scenario, a tool-calling analytics assistant, and a historical hourly demand visualization. Data is from the UCI Bike Sharing dataset for 2011–2012. No live forecasting, station-level recommendations, or weather feed is provided.

## Agent interface

`app.answer_question(message, history)` delegates to `src.agent.ask_agent(message)`. **History is display-only**; previous turns are not included in the model's current context. A follow-up question must repeat any relevant conditions. Responses and failures are rendered as Markdown. Unexpected exceptions are logged server-side and converted into a generic user-facing message.

Provider order is selected with `AGENT_PRIMARY_PROVIDER=groq` or `gemini` in the local `.env` file. The active model is configured with `GROQ_MODEL` or `GEMINI_MODEL`. The alternate provider is invoked only for provider API failures. Missing primary credentials can be handled by the configured alternate. The application does not display or transmit secrets as tool arguments.

The tool boundary remains `src/agent_tools.dispatch_tool`, with six allowlisted tool schemas and fixed argument limits. Model-generated SQL, file paths, arbitrary functions, or code are not executed.

## Numerical comparison reliability

`src.analytics.compare_workdays()` returns per-group counts and historical means. When both working-day categories are present, it also adds the following fields to the working-day row:

| Field | Definition |
|---|---|
| `difference_vs_nonworking` | Working-day mean minus non-working-day mean, in rentals/hour |
| `percent_difference_vs_nonworking` | `100 × difference / non-working-day mean`, or null when the denominator is zero |

Using the observed means 193.21 and 181.41 yields **11.80 rentals/hour** and **6.50%** higher average hourly demand. This is a descriptive association, not evidence that working-day status causes the difference. The values are computed in Python and returned by the registered SQL analytics tool. The LLM is instructed to quote the supplied percentage instead of inventing one. This reduces, but cannot eliminate, errors in generated explanations.

## Historical profiles

`app.load_hourly_patterns()` calls `hourly_demand(workingday=0)` and `hourly_demand(workingday=1)` and returns a Pandas DataFrame. The chart plots mean rentals per recorded hour, with a separate line for working and non-working dates. Empty/missing database states are handled explicitly. The chart is built from SQLite results without invoking an LLM.

## Local verification

```bash
python -m pytest -q
python -m scripts.agent_smoke_test
python app.py
```

Inside Gradio, verify the demand prediction form, three independent assistant questions, and the historical profile plot. Confirm that a request mentioning only an hour returns recorded historical evidence and that a fully specified scenario invokes the optimized prediction model. Compare numeric statements against actual tool data; automated mock tests do not establish live provider correctness.

## Constraints

LLM answers can still be numerically or semantically wrong. A sample size is not a confidence interval. Provider uptime, per-project quotas, and free-tier terms can change. Model evaluation covers the historical holdout only. The interface is a local analytics prototype, not a continuously monitored production application.
