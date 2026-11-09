"""Deterministic agent tests: no external Gemini requests or real model artifacts."""

from types import SimpleNamespace

import pytest

import src.agent as agent


class StubModels:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def generate_content(self, **kwargs):
        self.requests.append(kwargs)
        if not self.responses:
            raise AssertionError("Unexpected additional Gemini API request")
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def fake_client(*responses):
    return SimpleNamespace(models=StubModels(responses))


def model_response(*calls, text=None, content=None):
    content = content or SimpleNamespace(role="model", parts=["function-call-parts"])
    return SimpleNamespace(
        function_calls=list(calls),
        text=text,
        candidates=[SimpleNamespace(content=content)],
    )


def function_call(name, args=None):
    return SimpleNamespace(name=name, args=args)


def test_tool_schemas_are_exposed_to_gemini():
    tool = agent._gemini_tool()
    assert {declaration.name for declaration in tool.function_declarations} == {
        "hourly_profile", "compare_workdays", "weather_impact",
        "hourly_demand", "historical_reference", "estimate_scenario",
    }
    assert all(declaration.parameters_json_schema for declaration in tool.function_declarations)


def test_empty_question_requires_no_key(monkeypatch):
    monkeypatch.setattr(agent, "_client", lambda: (_ for _ in ()).throw(AssertionError))
    assert "enter a question" in agent.ask_agent("  ")


def test_long_question_is_rejected():
    result = agent.ask_agent("x" * (agent.MAX_QUESTION_LENGTH + 1))
    assert "too long" in result


def test_missing_key_returns_setup_message(monkeypatch):
    monkeypatch.setattr(agent, "_client", lambda: None)
    monkeypatch.setattr(agent, "_groq_client", lambda: None)
    result = agent.ask_agent("Compare working days", primary_provider="gemini")
    assert "GEMINI_API_KEY" in result


def test_direct_answer_without_tool_execution():
    client = fake_client(model_response(text="I only analyze historical rental patterns."))
    assert "historical rental" in agent.ask_agent("What can you do?", client=client)
    assert len(client.models.requests) == 1


def test_tool_call_is_dispatched_and_followed_by_response(monkeypatch):
    captured = []
    monkeypatch.setattr(agent, "dispatch_tool", lambda name, args: captured.append((name, args)) or {
        "ok": True, "tool": name, "data": {"avg_rentals": 359.01, "observations": 727}
    })
    client = fake_client(
        model_response(function_call("hourly_profile", {"hour": 8})),
        model_response(text="At 08:00, the mean was 359.01 rentals across 727 observations."),
    )
    answer = agent.ask_agent("What is the average at 8 AM?", client=client)
    assert "359.01" in answer
    assert captured == [("hourly_profile", {"hour": 8})]
    second_history = client.models.requests[1]["contents"]
    assert second_history[-1].role == "tool"
    assert second_history[-1].parts[0].function_response.response["result"]["ok"] is True


def test_multiple_tool_calls_in_one_round(monkeypatch):
    captured = []
    monkeypatch.setattr(agent, "dispatch_tool", lambda name, args: captured.append(name) or {
        "ok": True, "tool": name, "data": []
    })
    client = fake_client(
        model_response(function_call("compare_workdays", {}), function_call("weather_impact", {})),
        model_response(text="Both historical summaries were retrieved."),
    )
    assert "retrieved" in agent.ask_agent("Compare day and weather patterns", client=client)
    assert captured == ["compare_workdays", "weather_impact"]
    assert len(client.models.requests[1]["contents"][-1].parts) == 2


def test_unregistered_tool_is_handed_to_dispatcher(monkeypatch):
    monkeypatch.setattr(agent, "dispatch_tool", lambda name, args: {
        "ok": False, "tool": name,
        "error": {"code": "unknown_tool", "message": "Tool not registered"},
    })
    client = fake_client(
        model_response(function_call("run_sql", {"query": "DROP TABLE bike_rentals"})),
        model_response(text="That operation is not available."),
    )
    assert "not available" in agent.ask_agent("Execute raw SQL", client=client)
    envelope = client.models.requests[1]["contents"][-1].parts[0].function_response.response["result"]
    assert envelope["error"]["code"] == "unknown_tool"


def test_excess_calls_are_not_executed(monkeypatch):
    monkeypatch.setattr(agent, "dispatch_tool", lambda *args: (_ for _ in ()).throw(AssertionError))
    calls = [function_call("hourly_profile", {"hour": 8})] * (agent.MAX_TOTAL_TOOL_CALLS + 1)
    client = fake_client(model_response(*calls))
    assert "execution limit" in agent.ask_agent("Check many hours", client=client)
    assert len(client.models.requests) == 1


def test_max_tool_rounds_are_bounded(monkeypatch):
    monkeypatch.setattr(agent, "dispatch_tool", lambda name, args: {"ok": True, "tool": name, "data": []})
    responses = [model_response(function_call("compare_workdays", {})) for _ in range(agent.MAX_TOOL_ROUNDS + 1)]
    client = fake_client(*responses)
    assert "execution limit" in agent.ask_agent("Keep calling tools", client=client)
    assert len(client.models.requests) == agent.MAX_TOOL_ROUNDS + 1


def test_empty_model_response_is_handled():
    client = fake_client(model_response(text="  "))
    assert "no usable answer" in agent.ask_agent("Hello", client=client)


@pytest.mark.parametrize("code, expected", [(429, "rate limit"), (403, "access"), (404, "model")])
def test_api_errors_are_sanitized(code, expected):
    exc = RuntimeError("secret API details")
    exc.code = code
    client = fake_client(exc)
    result = agent.ask_agent("Summarize weather", client=client)
    assert expected in result
    assert "secret" not in result


def test_unknown_api_exception_does_not_leak_details():
    client = fake_client(RuntimeError("private service details"))
    result = agent.ask_agent("Check weather", client=client)
    assert "could not complete" in result
    assert "private" not in result


def test_invalid_function_call_name_is_reported():
    client = fake_client(model_response(function_call("", {})))
    assert "invalid tool request" in agent.ask_agent("Give demand", client=client)


def test_sdk_does_not_execute_tools_automatically():
    client = fake_client(model_response(text="I can summarize historical demand."))
    agent.ask_agent("What does this system do?", client=client)
    config = client.models.requests[0]["config"]
    assert config.automatic_function_calling.disable is True
    assert config.tools[0].function_declarations
