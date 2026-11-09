"""Allowlisted analytics tools, JSON schemas, and validated execution boundary."""

import json
import math
import re
import sqlite3
from datetime import date
from typing import Any

from src.analytics import (
    compare_workdays,
    historical_reference,
    hourly_demand,
    hourly_profile,
    weather_impact,
)
from src.config import DB_PATH
from src.predict import estimate_scenario


MAX_ARGUMENT_LENGTH = 4096


class ToolArgumentsError(ValueError):
    """Raised when a tool request violates its declared argument contract."""


class ToolNotAllowedError(ValueError):
    """Raised when a requested tool is not registered."""


def _integer(minimum: int, maximum: int) -> dict:
    return {"type": "integer", "minimum": minimum, "maximum": maximum}


def _number(minimum: float, maximum: float) -> dict:
    return {"type": "number", "minimum": minimum, "maximum": maximum}


def _definition(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


TOOL_DEFINITIONS = [
    _definition(
        "hourly_profile",
        "Summarize recorded 2011–2012 bike rentals at one hour across all dates and weather.",
        {"hour": _integer(0, 23)},
        ["hour"],
    ),
    _definition(
        "compare_workdays",
        "Compare historical hourly rental counts for working and non-working days.",
        {},
        [],
    ),
    _definition(
        "weather_impact",
        "Summarize hourly rental counts by recorded weather category, including sample sizes.",
        {},
        [],
    ),
    _definition(
        "hourly_demand",
        "Get a historical profile grouped by hour, optionally limited to a working-day category.",
        {"workingday": _integer(0, 1)},
        [],
    ),
    _definition(
        "historical_reference",
        "Aggregate historical rentals matching an hour and optionally day type and weather category.",
        {
            "hour": _integer(0, 23),
            "workingday": _integer(0, 1),
            "weathersit": _integer(1, 4),
        },
        ["hour"],
    ),
    _definition(
        "estimate_scenario",
        "Estimate total hourly rentals from historical patterns and return matched SQL evidence. Not a live forecast.",
        {
            "date_text": {"type": "string", "format": "date"},
            "hour": _integer(0, 23),
            "holiday": {"type": "boolean"},
            "weather_code": _integer(1, 4),
            "temperature_c": _number(0, 41),
            "humidity_pct": _number(0, 100),
            "windspeed_kmh": _number(0, 67),
        },
        [
            "date_text", "hour", "holiday", "weather_code",
            "temperature_c", "humidity_pct", "windspeed_kmh",
        ],
    ),
]

TOOL_SCHEMAS = {
    item["function"]["name"]: item["function"]["parameters"]
    for item in TOOL_DEFINITIONS
}


def _historical_handler(func):
    def call(**kwargs):
        if not DB_PATH.is_file():
            raise FileNotFoundError("Historical database unavailable. Run: python -m src.db")
        return func(**kwargs)
    return call


TOOL_HANDLERS = {
    "hourly_profile": _historical_handler(hourly_profile),
    "compare_workdays": _historical_handler(compare_workdays),
    "weather_impact": _historical_handler(weather_impact),
    "hourly_demand": _historical_handler(hourly_demand),
    "historical_reference": _historical_handler(historical_reference),
    "estimate_scenario": estimate_scenario,
}


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ToolArgumentsError(f"Duplicate argument: {key}")
        result[key] = value
    return result


def parse_arguments(arguments: str | dict) -> dict:
    """Decode a JSON object or accept an already-decoded plain dictionary."""
    if isinstance(arguments, str):
        if len(arguments) > MAX_ARGUMENT_LENGTH:
            raise ToolArgumentsError("Tool arguments exceed the size limit")
        try:
            arguments = json.loads(arguments, object_pairs_hook=_unique_keys)
        except json.JSONDecodeError as exc:
            raise ToolArgumentsError("Tool arguments must be valid JSON") from exc
    if type(arguments) is not dict:
        raise ToolArgumentsError("Tool arguments must be a JSON object")
    return arguments


def validate_arguments(name: str, arguments: str | dict) -> dict:
    """Reject unknown, missing, extra, incorrectly typed, or out-of-range fields."""
    if type(name) is not str or name not in TOOL_SCHEMAS:
        raise ToolNotAllowedError("Requested tool is not allowed")
    values = parse_arguments(arguments)
    schema = TOOL_SCHEMAS[name]
    properties = schema["properties"]

    missing = set(schema["required"]) - values.keys()
    if missing:
        raise ToolArgumentsError(f"Missing required arguments: {', '.join(sorted(missing))}")
    unknown = values.keys() - properties.keys()
    if unknown:
        raise ToolArgumentsError(f"Unknown arguments: {', '.join(sorted(unknown))}")

    for key, value in values.items():
        spec = properties[key]
        kind = spec["type"]
        if kind == "integer":
            valid_type = type(value) is int
        elif kind == "number":
            valid_type = type(value) in (int, float) and math.isfinite(value)
        elif kind == "boolean":
            valid_type = type(value) is bool
        else:
            valid_type = type(value) is str
        if not valid_type:
            raise ToolArgumentsError(f"Invalid type for {key}: expected {kind}")

        if kind in ("integer", "number"):
            if not spec["minimum"] <= value <= spec["maximum"]:
                raise ToolArgumentsError(f"{key} must be between {spec['minimum']} and {spec['maximum']}")
        if spec.get("format") == "date":
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                raise ToolArgumentsError("date_text must use YYYY-MM-DD")
            try:
                date.fromisoformat(value)
            except ValueError as exc:
                raise ToolArgumentsError("date_text is not a valid calendar date") from exc
    return values


def _failure(name: str, code: str, message: str) -> dict:
    return {"ok": False, "tool": name, "error": {"code": code, "message": message}}


def dispatch_tool(name: str, arguments: str | dict) -> dict:
    """Execute an approved tool and return a structured success or failure result."""
    if type(name) is not str or name not in TOOL_HANDLERS:
        return _failure(name, "unknown_tool", "Requested tool is not registered")

    try:
        validated = validate_arguments(name, arguments)
    except ToolArgumentsError as exc:
        return _failure(name, "invalid_arguments", str(exc))

    try:
        result = TOOL_HANDLERS[name](**validated)
    except (FileNotFoundError, sqlite3.Error):
        return _failure(name, "tool_unavailable", "Tool dependencies are unavailable")
    except Exception:
        return _failure(name, "execution_error", "Tool execution failed")

    return {"ok": True, "tool": name, "data": result}