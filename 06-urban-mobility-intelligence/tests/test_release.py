"""Release checks are deterministic and never contact a model provider."""

import warnings

import pytest

import app
from scripts import evaluate_agent, release_check


def test_gradio_chatbot_uses_non_deprecated_messages_format():
    with warnings.catch_warnings(record=True) as issued:
        warnings.simplefilter("always")
        widget = app.build_chatbot()
    assert widget is not None
    assert not any("tuples" in str(item.message).lower() for item in issued)


def test_release_tool_registry_blocks_unapproved_sql():
    result = release_check.verify_tool_boundary()
    assert result == {"registered_tools": 6, "unknown_tool_denied": True}


def test_missing_source_data_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        release_check.verify_source(tmp_path / "missing.csv")


def test_missing_database_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        release_check.verify_database(tmp_path / "missing.sqlite")


def test_missing_model_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        release_check.verify_model(tmp_path / "missing.joblib")


def test_release_report_sanitizes_internal_errors(monkeypatch):
    monkeypatch.setattr(release_check, "verify_source", lambda: (_ for _ in ()).throw(
        RuntimeError("/private/secret/key")))
    result = release_check.run_checks()
    assert result["checks"]["source_data"] == {
        "status": "failed", "reason": "RuntimeError"
    }
    assert "/private/secret/key" not in str(result)
    assert result["passed"] < result["total"]


def test_live_evaluator_number_parser_handles_thousands_and_decimals():
    answer = "5,514 records, mean 181.41; working 11,865 and 193.21; difference 6.5%."
    assert evaluate_agent.contains_numbers(answer, [
        (5514, 0), (181.41, 0.01), (11865, 0), (193.21, 0.01), (6.5, 0.01)
    ])


def test_evaluator_rejects_missing_percentage():
    answer = "Working mean 193.21; nonworking mean 181.41; difference approximately 12%."
    assert not evaluate_agent.contains_numbers(answer, [(6.5, 0.1)])


def test_evaluator_uses_tolerance_for_rounded_means():
    assert evaluate_agent.contains_numbers("Mean 525.3 rentals/hour.", [(525.32, 0.1)])


def test_evaluator_no_api_calls_without_live_flag(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["evaluate_agent"])
    monkeypatch.setattr(evaluate_agent, "ask_agent", lambda _: (_ for _ in ()).throw(AssertionError))
    assert evaluate_agent.main() == 0
    assert "No API requests made" in capsys.readouterr().out


def test_agent_evaluation_cases_are_based_on_local_analytics(monkeypatch):
    monkeypatch.setattr(evaluate_agent, "compare_workdays", lambda: [
        {"workingday": 0, "observations": 5514, "avg_rentals": 181.41},
        {"workingday": 1, "observations": 11865, "avg_rentals": 193.21,
         "percent_difference_vs_nonworking": 6.5},
    ])
    monkeypatch.setattr(evaluate_agent, "hourly_demand", lambda workingday: [
        {"hr": hour, "avg_rentals": avg} for hour, avg in ((8, 477.0), (17, 525.3), (18, 492.2))
    ])
    monkeypatch.setattr(evaluate_agent, "hourly_profile", lambda hour: {
        "observations": 727, "avg_rentals": 359.01
    })
    monkeypatch.setattr(evaluate_agent, "weather_impact", lambda: [
        {"weathersit": 4, "observations": 3}
    ])
    cases = evaluate_agent.evaluation_cases()
    assert len(cases) == 4
    assert cases[0]["expected"][-1] == (6.5, 0.11)
    assert cases[1]["expected"][0:3] == [(17, 0), (18, 0), (8, 0)]
    assert cases[3]["expected"] == [(3, 0)]
