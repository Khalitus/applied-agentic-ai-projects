# Agent Tool Contract and Controlled Execution

## Purpose

The tool boundary exposes a fixed set of historical bicycle-demand analytics and prediction functions to a future language-model orchestrator. It separates model-authored arguments from trusted Python execution. Tool availability does not mean a conversational agent is enabled.

## Registered tools

| Tool | Arguments | Result |
|---|---|---|
| `hourly_profile` | `hour`: integer, 0–23 | Historical sample size and rental statistics for that hour |
| `compare_workdays` | None | Working/non-working historical demand summaries |
| `weather_impact` | None | Historical weather-category demand summaries and sample sizes |
| `hourly_demand` | Optional `workingday`: 0 or 1 | Historical hourly group profile |
| `historical_reference` | Required `hour`; optional `workingday` and `weathersit` | Conditional historical demand aggregates |
| `estimate_scenario` | `date_text`, `hour`, `holiday`, `weather_code`, `temperature_c`, `humidity_pct`, `windspeed_kmh` | Optimized model estimate and SQL reference evidence |

`estimate_scenario` is preferred over a bare model value when the question requests both a prediction and historical context. SQL reference observations match hour, working-day status and weather category, **not** exact meteorological values.

## Request and response format

Tool calls use a registered name and a JSON object of arguments. A tool invocation can receive a JSON string as emitted by an LLM function-calling interface or an already-decoded Python dictionary.

Example:

```json
{"name":"historical_reference","arguments":{"hour":8,"workingday":1,"weathersit":1}}
```

Successful execution returns:

```json
{"ok":true,"tool":"historical_reference","data":{"hour":8,"observations":280,"avg_rentals":503.86,"min_rentals":66,"max_rentals":839}}
```

The values above reproduce one observed historical comparison from the local dataset; they are not global defaults. Errors use a stable envelope:

```json
{"ok":false,"tool":"historical_reference","error":{"code":"invalid_arguments","message":"..."}}
```

Permitted error codes are `unknown_tool`, `invalid_arguments`, `tool_unavailable`, and `execution_error`. Do not surface arbitrary database errors, filesystem paths, secrets or stack traces in tool responses.

## Validation boundary

The registry in `src/agent_tools.py` is authoritative. Requests are rejected unless the tool name is explicitly registered. Arguments are decoded as JSON objects and checked for required fields, unknown fields, exact primitive types and bounded numeric ranges. Invalid calendar dates, non-finite numeric values, duplicate JSON keys and oversized string payloads are rejected. SQL database paths and arbitrary queries are never exposed to the language model.

The approved handlers call existing read-only aggregation functions or validated XGBoost inference. The caller must not be allowed to inject function names, SQL fragments or filesystem paths through arguments.

## Implementing the dispatcher

Complete `dispatch_tool(name, arguments)` in `src/agent_tools.py`. It must follow this order:

1. Reject any name absent from `TOOL_HANDLERS` using `_failure(name, "unknown_tool", ...)`.
2. Call `validate_arguments(name, arguments)`. Convert `ToolArgumentsError` into an `invalid_arguments` failure.
3. Execute only `TOOL_HANDLERS[name](**validated)`.
4. On success return `{"ok": True, "tool": name, "data": result}`.
5. Catch `FileNotFoundError` and `sqlite3.Error` as `tool_unavailable` without leaking implementation details.
6. Catch other exceptions as `execution_error` with a generic message. Do not conceal errors in development logs; log full exceptions on the server side when an agent is integrated.

Do not use `eval`, `exec`, `globals`, Python attribute lookup derived from the model, arbitrary SQL, or a dynamically imported function.

## Execution boundaries

- The registered tool set is bounded and contains no general database mutation.
- SQL queries use bound parameters and fixed SQL fragments.
- Model requests are restricted to the existing seven-field scenario contract.
- Small sample sizes must remain visible when interpreting results.
- Historical 2011–2012 data cannot support claims about current demand or live operations.
- The language-model API connection, tool-call rounds, timeout strategy and final grounded response are separate integration concerns.

## Verification

```bash
python -m pytest tests/test_agent_tools.py -q
python -m pytest -q
```

Tests cover schemas, argument handling, denied operations, execution results, and sanitized failures. For integration, only invoke tools via `dispatch_tool`; never execute model-authored functions directly.
