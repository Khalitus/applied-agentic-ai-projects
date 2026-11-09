# Agent Provider Fallback

## Objective

The conversational analytics agent uses Gemini as its primary provider and Groq as a secondary provider when a Gemini API request fails. Both providers use the same six allowlisted analytics/prediction tools from `src/agent_tools.py`. The underlying SQLite and XGBoost components are unchanged.

## Configuration

The local `.env` file (not tracked by Git) accepts:

```dotenv
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-3.5-flash-lite
GROQ_API_KEY=...
GROQ_MODEL=openai/gpt-oss-120b
```

Model availability and free-tier quotas depend on each provider account and can change. Neither provider is guaranteed to provide uninterrupted or free service. Do not expose credentials in source code, reports, logs or screenshots.

## Failure behavior

1. For each nonempty, valid user question, call Gemini first.
2. If Gemini returns an API/provider exception (HTTP 400/401/403/404/429/5xx, a connection failure or timeout), log only the exception class and HTTP code. Do not log credentials or arbitrary provider messages.
3. If `GROQ_API_KEY` is configured, start a new conversation using Groq and the original question. Do not reuse Gemini-specific SDK message objects.
4. Send Groq the same tool definitions. Every requested function is dispatched through `dispatch_tool()`, including all allowlist and argument validation checks.
5. On completion, return the Groq answer as an ordinary answer. If both providers fail, return concise sanitized diagnostic messages.
6. Do **not** fail over on a normal answer, invalid tool request, local tool failure response, or tool-round limit.

A fallback started after some successful Gemini tool calls may repeat those calls on Groq. This is acceptable for the current **read-only** tools. Do not apply this restart policy to future state-changing tools without idempotency safeguards.

## Execution constraints

Both providers have independent hard limits of four tool-call rounds and eight tool invocations per question. `src/agent_tools.py` alone determines which operations may run; the LLM cannot access arbitrary SQL or filesystem paths. Function results are returned to the requesting provider using its native tool-response format.

The agent is directed to ground numeric claims in actual tool results, and responses remain non-authoritative for live transportation operations. Prompt instructions alone are not a guarantee against unsupported assertions; further agent-evaluation cases remain essential.

## Diagnostics

Run the regression tests before live provider requests:

```bash
python -m pytest tests/test_agent.py tests/test_agent_fallback.py -q
python -m pytest -q
```

Use a minimal provider check when diagnosing credentials or API availability:

```bash
python -m scripts.provider_diagnostics
```

This makes one plain-text API request per configured provider and displays only access success or the exception class and HTTP status. For example, if plain Gemini requests succeed but the agent's Gemini request returns HTTP 400, inspect function declaration and request compatibility. The fallback can still work while that issue is investigated.

Run the existing live agent smoke test using Gemini-first and Groq-on-error behavior:

```bash
python -m scripts.agent_smoke_test
```

For free-tier budgets, avoid repeated `--all` smoke tests until the first call succeeds. Each tool-calling interaction can involve multiple billable/rate-limited API requests.

## Dependencies

`google-genai`, `groq` and `python-dotenv` are already declared in the project requirements. No new package is required when the existing environment satisfies them.
