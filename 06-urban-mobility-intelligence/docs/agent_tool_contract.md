# Agent Tool Contract

## Purpose

The conversational analytics service exposes a fixed set of trusted local functions to Gemini or Groq. Model-generated function requests are treated as **untrusted inputs**. An allowlisted Python dispatcher checks both function names and arguments before invoking SQLite analytics or XGBoost inference.

## Registered tools

| Tool | Inputs | Evidence returned |
|---|---|---|
| `hourly_profile` | Required `hour` (integer 0–23) | Count, mean, minimum and maximum rentals for that hour |
| `compare_workdays` | None | Two day-type summaries; a Python-computed working-day absolute and percentage difference |
| `weather_impact` | None | Historical demand summary by weather category, with sample sizes |
| `hourly_demand` | Optional `workingday` (0 or 1) | Hour-by-hour average and observation count |
| `historical_reference` | Required `hour`; optional `workingday`, `weathersit` | Summary of records matching the categorical filters |
| `estimate_scenario` | Date, hour, holiday flag, weather category, temperature, humidity, wind | Optimized model estimate and associated historical SQL reference |

All tools are **read-only**. No tool accepts arbitrary SQL, filesystem paths, Python code or a dynamic function name.

## Execution contract

The definitions in `src/agent_tools.py` use JSON-compatible input schemas. The registry is the only permitted source of handlers.

1. `dispatch_tool(name, arguments)` rejects unregistered tool names.
2. `validate_arguments()` rejects malformed JSON, duplicate keys, missing or unexpected fields, invalid types, invalid calendar dates, non-finite numbers and out-of-range values.
3. A valid request invokes only `TOOL_HANDLERS[name](**validated)`.
4. The dispatcher wraps the result in `{"ok":true,"tool":"...","data":...}`.
5. Errors are normalized to `unknown_tool`, `invalid_arguments`, `tool_unavailable` or `execution_error` without disclosing paths, SQL internals or secrets.

Example valid request:

```json
{"name":"historical_reference","arguments":{"hour":8,"workingday":1,"weathersit":1}}
```

Example success envelope, based on the verified historical database:

```json
{"ok":true,"tool":"historical_reference","data":{"hour":8,"observations":280,"avg_rentals":503.86,"min_rentals":66,"max_rentals":839}}
```

Example denied operation:

```json
{"ok":false,"tool":"run_sql","error":{"code":"unknown_tool","message":"Requested tool is not registered"}}
```

## Provider integration

`src/agent.py` translates the same six tool definitions for Gemini function declarations or Groq chat-completions tool calls. The provider cannot bypass validation. Function-call rounds and total call counts are bounded. When an API request fails, the alternate provider begins with fresh conversation state; existing tools are read-only, so duplicate retrieval is safe.

The instruction prompt requires tool-backed numerical claims, relevant sample counts and explicit distinctions between model estimates and historical SQL means. Prompt instructions mitigate—but cannot eliminate—incorrect prose or calculations. The Gradio chat processes each question independently without prior conversation context.

## Verification

```bash
python -m pytest tests/test_agent_tools.py -q
python -m pytest tests/test_agent.py tests/test_agent_fallback.py tests/test_agent_interface.py -q
python -m scripts.release_check
```

Read [agent execution](agent_execution.md), [provider routing](provider_fallback.md) and [release validation](release_validation.md) for operational checks and known limitations.
