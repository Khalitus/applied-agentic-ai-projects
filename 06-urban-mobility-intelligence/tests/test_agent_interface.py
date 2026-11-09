"""UI integration, provider order and deterministic comparison checks."""

from types import SimpleNamespace

import pandas as pd
import pytest

import app
import src.agent as agent
import src.analytics as analytics


def test_chat_delegates_independent_question(monkeypatch):
    calls = []
    monkeypatch.setattr(app, "ask_agent", lambda question: calls.append(question) or "Evidence found.")
    result = app.answer_question("Compare workdays", [{"role": "user", "content": "Ignore"}])
    assert result == "Evidence found."
    assert calls == ["Compare workdays"]


@pytest.mark.parametrize("message", ["", "  ", None, 4])
def test_chat_rejects_invalid_input_without_provider(monkeypatch, message):
    monkeypatch.setattr(app, "ask_agent", lambda question: (_ for _ in ()).throw(AssertionError))
    assert "Enter a question" in app.answer_question(message, [])


def test_chat_hides_unexpected_internal_exception(monkeypatch):
    monkeypatch.setattr(app, "ask_agent", lambda question: (_ for _ in ()).throw(RuntimeError("secret key")))
    result = app.answer_question("What is demand?", [])
    assert "unavailable" in result
    assert "secret" not in result


def test_profiles_missing_database(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "DB_PATH", tmp_path / "missing.db")
    frame, message = app.load_hourly_patterns()
    assert frame.empty
    assert "python -m src.db" in message


def test_profiles_are_sourced_by_day_type(monkeypatch, tmp_path):
    file = tmp_path / "data.db"
    file.touch()
    monkeypatch.setattr(app, "DB_PATH", file)
    requests = []

    def query(workingday):
        requests.append(workingday)
        return [
            {"hr": 8, "avg_rentals": 20 + workingday * 40, "observations": 11},
            {"hr": 17, "avg_rentals": 30 + workingday * 50, "observations": 12},
        ]

    monkeypatch.setattr(app, "hourly_demand", query)
    frame, message = app.load_hourly_patterns()
    assert requests == [0, 1]
    assert len(frame) == 4
    assert set(frame.columns) == set(app.CHART_COLUMNS)
    assert set(frame["day_type"]) == {"Non-working day", "Working day"}
    assert frame.loc[frame["day_type"].eq("Working day"), "avg_rentals"].tolist() == [60, 80]
    assert "2011–2012" in message


def test_profiles_empty_database(monkeypatch, tmp_path):
    file = tmp_path / "data.db"
    file.touch()
    monkeypatch.setattr(app, "DB_PATH", file)
    monkeypatch.setattr(app, "hourly_demand", lambda workingday: [])
    frame, message = app.load_hourly_patterns()
    assert frame.empty
    assert "No historical" in message


def test_profiles_fail_without_leaking_database_errors(monkeypatch, tmp_path):
    file = tmp_path / "data.db"
    file.touch()
    monkeypatch.setattr(app, "DB_PATH", file)
    monkeypatch.setattr(app, "hourly_demand", lambda workingday: (_ for _ in ()).throw(ValueError("sensitive schema")))
    frame, message = app.load_hourly_patterns()
    assert frame.empty
    assert "sensitive" not in message
    assert "Data error" in message


def test_groq_preferred_success_does_not_call_gemini(monkeypatch):
    calls = []
    monkeypatch.setattr(agent, "_run_groq", lambda question, client, model: calls.append("groq") or "Groq answer")
    monkeypatch.setattr(agent, "_run_gemini", lambda *args: (_ for _ in ()).throw(AssertionError))
    answer = agent.ask_agent("Compare hours", client=object(), groq_client=object(), primary_provider="groq")
    assert answer == "Groq answer"
    assert calls == ["groq"]


def test_groq_failure_falls_back_to_gemini(monkeypatch):
    calls = []
    def fail(*args):
        calls.append("groq")
        return agent._ProviderFailure("Groq", "Groq rate limited", 429)
    def pass_gemini(*args):
        calls.append("gemini")
        return "Gemini answer"
    monkeypatch.setattr(agent, "_run_groq", fail)
    monkeypatch.setattr(agent, "_run_gemini", pass_gemini)
    answer = agent.ask_agent("Compare hours", client=object(), groq_client=object(), primary_provider="groq")
    assert answer == "Gemini answer"
    assert calls == ["groq", "gemini"]


def test_groq_preferred_missing_key_uses_gemini(monkeypatch):
    monkeypatch.setattr(agent, "_groq_client", lambda: None)
    monkeypatch.setattr(agent, "_run_gemini", lambda *args: "Gemini answer")
    assert agent.ask_agent("Compare", client=object(), primary_provider="groq") == "Gemini answer"


def test_groq_preferred_without_both_keys(monkeypatch):
    monkeypatch.setattr(agent, "_groq_client", lambda: None)
    monkeypatch.setattr(agent, "_client", lambda: None)
    result = agent.ask_agent("Compare", primary_provider="groq")
    assert "GROQ_API_KEY" in result
    assert "GEMINI_API_KEY" in result


def test_invalid_provider_rejected_without_any_request(monkeypatch):
    monkeypatch.setattr(agent, "_run_groq", lambda *args: (_ for _ in ()).throw(AssertionError))
    monkeypatch.setattr(agent, "_run_gemini", lambda *args: (_ for _ in ()).throw(AssertionError))
    result = agent.ask_agent("Compare", primary_provider="unknown")
    assert "Invalid AGENT_PRIMARY_PROVIDER" in result


def test_groq_failure_and_gemini_failure_are_sanitized(monkeypatch):
    monkeypatch.setattr(agent, "_run_groq", lambda *args: agent._ProviderFailure("Groq", "quota exhausted", 429))
    monkeypatch.setattr(agent, "_run_gemini", lambda *args: agent._ProviderFailure("Gemini", "access denied", 403))
    text = agent.ask_agent("Compare", client=object(), groq_client=object(), primary_provider="groq")
    assert "Gemini fallback also failed" in text
    assert "quota exhausted" in text


def test_comparison_percent_uses_python_not_llm(monkeypatch):
    class Connection:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def execute(self, sql):
            return SimpleNamespace(fetchall=lambda: [
                {"workingday": 0, "observations": 5514, "avg_rentals": 181.41, "min_rentals": 1, "max_rentals": 783},
                {"workingday": 1, "observations": 11865, "avg_rentals": 193.21, "min_rentals": 1, "max_rentals": 977},
            ])
    monkeypatch.setattr(analytics, "connect", lambda path: Connection())
    result = analytics.compare_workdays()
    assert result[1]["difference_vs_nonworking"] == 11.8
    assert result[1]["percent_difference_vs_nonworking"] == 6.5


def test_comparison_zero_reference_has_no_percentage(monkeypatch):
    class Connection:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def execute(self, sql):
            return SimpleNamespace(fetchall=lambda: [
                {"workingday": 0, "avg_rentals": 0.0},
                {"workingday": 1, "avg_rentals": 60.0},
            ])
    monkeypatch.setattr(analytics, "connect", lambda path: Connection())
    result = analytics.compare_workdays()
    assert result[1]["percent_difference_vs_nonworking"] is None
