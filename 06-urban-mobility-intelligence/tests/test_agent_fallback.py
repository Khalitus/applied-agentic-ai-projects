"""Provider failover tests; no real network requests or API credentials required."""

import json
from types import SimpleNamespace

import pytest

import src.agent as agent


class StubGeminiModels:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


class StubGroqCompletions:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


def fake_gemini(*replies):
    return SimpleNamespace(models=StubGeminiModels(replies))


def fake_groq(*replies):
    return SimpleNamespace(chat=SimpleNamespace(completions=StubGroqCompletions(replies)))


def gemini_text(text):
    return SimpleNamespace(function_calls=[], text=text)


def groq_text(text):
    message = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def groq_call(name, args, tool_id="call-1"):
    return SimpleNamespace(
        id=tool_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def groq_tools(*calls):
    message = SimpleNamespace(content=None, tool_calls=list(calls))
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def api_error(code):
    error = RuntimeError("internal token or private details")
    error.code = code
    return error


def test_gemini_success_does_not_use_groq():
    gemini = fake_gemini(gemini_text("Historical analysis only."))
    groq = fake_groq()
    result = agent.ask_agent("What can you do?", client=gemini, groq_client=groq)
    assert result == "Historical analysis only."
    assert not groq.chat.completions.calls


@pytest.mark.parametrize("status", [400, 401, 403, 404, 429, 500])
def test_gemini_api_error_triggers_groq(status):
    gemini = fake_gemini(api_error(status))
    groq = fake_groq(groq_text("Answered by Groq."))
    result = agent.ask_agent("Compare weather patterns", client=gemini, groq_client=groq)
    assert result == "Answered by Groq."
    assert len(groq.chat.completions.calls) == 1


def test_groq_calls_validated_dispatch_and_returns_tool_id(monkeypatch):
    invoked = []
    monkeypatch.setattr(agent, "dispatch_tool", lambda name, args: (
        invoked.append((name, args)) or
        {"ok": True, "tool": name, "data": {"observations": 727, "avg_rentals": 359.01}}
    ))
    groq = fake_groq(
        groq_tools(groq_call("hourly_profile", {"hour": 8}, "call-8")),
        groq_text("At 8 AM, average demand is 359.01 rentals across 727 hours."),
    )
    result = agent.ask_agent("What about 8 AM?", client=fake_gemini(api_error(429)), groq_client=groq)
    assert "359.01" in result
    assert invoked == [("hourly_profile", '{"hour": 8}')]
    second_request = groq.chat.completions.calls[1]["messages"]
    assert second_request[-1]["role"] == "tool"
    assert second_request[-1]["tool_call_id"] == "call-8"
    assert json.loads(second_request[-1]["content"])["data"]["observations"] == 727
    assert second_request[-2]["tool_calls"][0]["id"] == "call-8"


def test_groq_multiple_tool_calls(monkeypatch):
    called = []
    monkeypatch.setattr(agent, "dispatch_tool", lambda name, args: called.append(name) or {
        "ok": True, "tool": name, "data": []
    })
    groq = fake_groq(
        groq_tools(groq_call("compare_workdays", {}, "a"), groq_call("weather_impact", {}, "b")),
        groq_text("Both comparisons retrieved."),
    )
    result = agent._run_groq("Compare weather and days", groq, agent.DEFAULT_GROQ_MODEL)
    assert "retrieved" in result
    assert called == ["compare_workdays", "weather_impact"]
    sent = groq.chat.completions.calls[1]["messages"]
    assert sent[-2]["tool_call_id"] == "a"
    assert sent[-1]["tool_call_id"] == "b"


def test_unknown_groq_function_passes_only_through_dispatch(monkeypatch):
    monkeypatch.setattr(agent, "dispatch_tool", lambda name, args: {
        "ok": False, "tool": name,
        "error": {"code": "unknown_tool", "message": "Not registered"},
    })
    groq = fake_groq(
        groq_tools(groq_call("run_sql", {"sql": "DROP TABLE bike_rentals"})),
        groq_text("This operation is not supported."),
    )
    assert "not supported" in agent._run_groq("Do SQL", groq, agent.DEFAULT_GROQ_MODEL)
    sent = groq.chat.completions.calls[1]["messages"][-1]
    assert json.loads(sent["content"])["error"]["code"] == "unknown_tool"


def test_groq_limits_prevent_execution(monkeypatch):
    monkeypatch.setattr(agent, "dispatch_tool", lambda *args: (_ for _ in ()).throw(AssertionError))
    calls = [groq_call("hourly_profile", {"hour": 8}, f"id-{i}")
             for i in range(agent.MAX_TOTAL_TOOL_CALLS + 1)]
    groq = fake_groq(groq_tools(*calls))
    result = agent._run_groq("Many calls", groq, agent.DEFAULT_GROQ_MODEL)
    assert "execution limit" in result
    assert len(groq.chat.completions.calls) == 1


def test_groq_round_cap(monkeypatch):
    monkeypatch.setattr(agent, "dispatch_tool", lambda name, args: {"ok": True, "tool": name, "data": []})
    replies = [groq_tools(groq_call("compare_workdays", {}, f"id-{i}"))
               for i in range(agent.MAX_TOOL_ROUNDS + 1)]
    groq = fake_groq(*replies)
    assert "execution limit" in agent._run_groq("Keep calling", groq, agent.DEFAULT_GROQ_MODEL)
    assert len(groq.chat.completions.calls) == agent.MAX_TOOL_ROUNDS + 1


def test_groq_invalid_call_id_is_not_dispatched(monkeypatch):
    monkeypatch.setattr(agent, "dispatch_tool", lambda *args: (_ for _ in ()).throw(AssertionError))
    groq = fake_groq(groq_tools(groq_call("hourly_profile", {"hour": 8}, "")))
    assert "invalid tool request" in agent._run_groq("Demand", groq, agent.DEFAULT_GROQ_MODEL)


def test_both_provider_failures_are_sanitized():
    gemini = fake_gemini(api_error(429))
    groq = fake_groq(api_error(503))
    result = agent.ask_agent("Compare demands", client=gemini, groq_client=groq)
    assert "Groq fallback also failed" in result
    assert "internal token" not in result
    assert "rate limit" in result


def test_without_groq_key_returns_diagnostic(monkeypatch):
    gemini = fake_gemini(api_error(400))
    result = agent.ask_agent("Compare demands", client=gemini)
    assert "HTTP 400" in result
    assert "GROQ_API_KEY" in result


def test_groq_options_and_tools_passed():
    groq = fake_groq(groq_text("Done."))
    agent._run_groq("What can you do?", groq, agent.DEFAULT_GROQ_MODEL)
    args = groq.chat.completions.calls[0]
    assert args["model"] == "openai/gpt-oss-120b"
    assert args["tool_choice"] == "auto"
    assert {item["function"]["name"] for item in args["tools"]} == {
        "hourly_profile", "compare_workdays", "weather_impact",
        "hourly_demand", "historical_reference", "estimate_scenario",
    }
