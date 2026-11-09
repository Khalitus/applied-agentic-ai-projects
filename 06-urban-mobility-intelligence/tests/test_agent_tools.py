import json

import pytest

import src.agent_tools as tools


SCENARIO = {
    "date_text": "2012-08-15",
    "hour": 8,
    "holiday": False,
    "weather_code": 1,
    "temperature_c": 22,
    "humidity_pct": 60,
    "windspeed_kmh": 12,
}


def test_schema_and_handler_names_match():
    names = set(tools.TOOL_HANDLERS)
    assert names == set(tools.TOOL_SCHEMAS)
    assert names == {item["function"]["name"] for item in tools.TOOL_DEFINITIONS}
    assert all(not item["function"]["parameters"]["additionalProperties"]
               for item in tools.TOOL_DEFINITIONS)


def test_no_tool_exposes_database_paths():
    assert all("path" not in item["function"]["parameters"]["properties"]
               for item in tools.TOOL_DEFINITIONS)


def test_parser_rejects_duplicate_json_keys():
    with pytest.raises(tools.ToolArgumentsError, match="Duplicate"):
        tools.parse_arguments('{"hour":8,"hour":17}')


@pytest.mark.parametrize("bad", ['[]', 'null', '{"hour":8', 3, [], "x" * 4097])
def test_parser_rejects_invalid_argument_payloads(bad):
    with pytest.raises(tools.ToolArgumentsError):
        tools.parse_arguments(bad)


@pytest.mark.parametrize("payload", [
    {"hour": True}, {"hour": 8.5}, {"hour": -1}, {"hour": 24},
    {"hour": 8, "path": "/tmp/other.sqlite3"},
])
def test_rejects_invalid_hourly_profile_arguments(payload):
    with pytest.raises(tools.ToolArgumentsError):
        tools.validate_arguments("hourly_profile", payload)


@pytest.mark.parametrize("payload", [
    {**SCENARIO, "date_text": "2012-02-30"},
    {**SCENARIO, "date_text": "20120815"},
    {**SCENARIO, "holiday": 1},
    {**SCENARIO, "temperature_c": -1},
    {**SCENARIO, "humidity_pct": float("nan")},
    {**SCENARIO, "windspeed_kmh": 70},
])
def test_rejects_invalid_prediction_scenarios(payload):
    with pytest.raises(tools.ToolArgumentsError):
        tools.validate_arguments("estimate_scenario", payload)


def test_required_scenario_arguments_and_no_arg_tools():
    with pytest.raises(tools.ToolArgumentsError, match="Missing"):
        tools.validate_arguments("estimate_scenario", {"hour": 8})
    assert tools.validate_arguments("compare_workdays", "{}") == {}
    with pytest.raises(tools.ToolArgumentsError, match="Unknown"):
        tools.validate_arguments("weather_impact", {"sql": "DROP TABLE bike_rentals"})


def test_unregistered_tool_is_rejected():
    with pytest.raises(tools.ToolNotAllowedError):
        tools.validate_arguments("run_sql", {"query": "SELECT 1"})


def test_dispatch_success_returns_structured_data(monkeypatch):
    monkeypatch.setitem(tools.TOOL_HANDLERS, "hourly_profile", lambda hour: {"hr": hour, "observations": 5})
    result = tools.dispatch_tool("hourly_profile", json.dumps({"hour": 8}))
    assert result == {
        "ok": True,
        "tool": "hourly_profile",
        "data": {"hr": 8, "observations": 5},
    }


def test_dispatch_can_call_zero_argument_tool(monkeypatch):
    monkeypatch.setitem(tools.TOOL_HANDLERS, "compare_workdays", lambda: [{"workingday": 0}])
    result = tools.dispatch_tool("compare_workdays", {})
    assert result["ok"] is True
    assert result["data"] == [{"workingday": 0}]


def test_dispatch_rejects_unauthorized_tool_before_execution():
    result = tools.dispatch_tool("run_sql", {"query": "DELETE FROM bike_rentals"})
    assert result["ok"] is False
    assert result["error"]["code"] == "unknown_tool"


def test_dispatch_rejects_unknown_fields_without_calling_handler(monkeypatch):
    def forbidden(**kwargs):
        raise AssertionError("The handler must not execute")
    monkeypatch.setitem(tools.TOOL_HANDLERS, "hourly_profile", forbidden)
    result = tools.dispatch_tool("hourly_profile", {"hour": 8, "path": "/tmp/foo"})
    assert result["ok"] is False
    assert result["error"]["code"] == "invalid_arguments"


def test_dispatch_rejects_missing_required_fields():
    result = tools.dispatch_tool("estimate_scenario", {"hour": 8})
    assert result["ok"] is False
    assert result["error"]["code"] == "invalid_arguments"


def test_dispatch_catches_missing_model_or_database(monkeypatch):
    def missing(**kwargs):
        raise FileNotFoundError("Optimized model unavailable")
    monkeypatch.setitem(tools.TOOL_HANDLERS, "estimate_scenario", missing)
    result = tools.dispatch_tool("estimate_scenario", SCENARIO)
    assert result["ok"] is False
    assert result["error"]["code"] == "tool_unavailable"


def test_dispatch_handles_unexpected_errors_without_leaking_details(monkeypatch):
    def broken(**kwargs):
        raise RuntimeError("secret internal file path")
    monkeypatch.setitem(tools.TOOL_HANDLERS, "hourly_profile", broken)
    result = tools.dispatch_tool("hourly_profile", {"hour": 8})
    assert result["ok"] is False
    assert result["error"]["code"] == "execution_error"
    assert "secret" not in result["error"]["message"]


def test_dispatch_reports_database_errors_without_leaking_details(monkeypatch):
    def broken(**kwargs):
        raise __import__("sqlite3").OperationalError("no such table: secret")
    monkeypatch.setitem(tools.TOOL_HANDLERS, "weather_impact", broken)
    result = tools.dispatch_tool("weather_impact", "{}")
    assert result["ok"] is False
    assert result["error"]["code"] == "tool_unavailable"
    assert "secret" not in result["error"]["message"]
