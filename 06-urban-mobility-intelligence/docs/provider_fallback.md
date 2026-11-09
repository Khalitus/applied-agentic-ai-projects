# Agent Provider Fallback

## Objective

The conversational analytics agent supports **configurable provider priority**. Set `AGENT_PRIMARY_PROVIDER=groq` to use Groq first, with Gemini available as fallback, or `AGENT_PRIMARY_PROVIDER=gemini` for the reverse. Both providers use the same six allowlisted analytics/prediction tools from `src/agent_tools.py`. The underlying SQLite and XGBoost components are unchanged.

## Configuration

The local `.env` file (not tracked by Git) accepts:

```dotenv
AGENT_PRIMARY_PROVIDER=groq
GROQ_API_KEY=...
GROQ_MODEL=openai/gpt-oss-120b
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-3.5-flash-lite
```

Model availability and free-tier quotas depend on each provider account and can change. Neither provider is guaranteed to provide uninterrupted or free service. Do not expose credentials in source code, reports, logs or screenshots.

## Failure behavior

1. For each valid question, select the configured primary provider. The environment template prefers Groq because Gemini has returned HTTP 400 in the local project environment. If the setting is absent, the code retains Gemini-first for backward compatibility.
2. On a provider API exception, log only the exception class and HTTP code. Do not log credentials or provider error bodies.
3. Start a fresh conversation on the alternate configured provider, using the original question. Do not reuse SDK-specific message objects from the failed provider.
4. Both providers receive the same six tool definitions and route all requested operations through `dispatch_tool()`.
5. If both providers fail, return sanitized diagnostics. Do **not** fall back for ordinary model answers, invalid tool requests, application validation failures, or execution-limit responses.
6. A missing primary-provider key allows the alternate provider to run if configured; the application does not require both credentials.

Provider failover can repeat a previous successful tool request. All existing tools are read-only; do not reuse this policy for tools with side effects without idempotency safeguards.

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

Run the live smoke test using the preference in `.env`:

```bash
python -m scripts.agent_smoke_test
```

For free-tier budgets, avoid repeated `--all` smoke tests until the first call succeeds. Each tool-calling interaction can involve multiple billable/rate-limited API requests.

## Dependencies

`google-genai`, `groq` and `python-dotenv` are already declared in the project requirements. No new package is required when the existing environment satisfies them.
