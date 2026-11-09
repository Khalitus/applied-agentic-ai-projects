"""Interactive historical demand estimation, analytics and bounded agent access."""

import inspect
import logging
import math
import sqlite3

import gradio as gr
import pandas as pd

from src.agent import ask_agent
from src.analytics import hourly_demand
from src.config import DB_PATH
from src.predict import estimate_scenario


LOGGER = logging.getLogger(__name__)
CHART_COLUMNS = ["hour", "avg_rentals", "day_type", "observations"]


def build_chatbot() -> gr.Chatbot:
    """Use message histories across supported Gradio major versions."""
    options = {"type": "messages"} if "type" in inspect.signature(gr.Chatbot).parameters else {}
    return gr.Chatbot(**options)


def _integer_input(value, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{label} must be an integer")
    return int(number)


def format_prediction(result: dict) -> str:
    prediction = result["estimated_rentals"]
    scenario = result["scenario"]
    reference = result["historical_reference"]
    day_type = "working day" if scenario["workingday"] else "non-working day"
    conditions = (
        f"{scenario['hour']:02d}:00, {day_type}, "
        f"weather category {scenario['weather_code']}"
    )

    if result["reference_sample"] == "no_matches":
        historical = (
            f"No historical observations matched **{conditions}**. "
            "No historical average can be reported for this combination."
        )
    else:
        count = reference["observations"]
        historical = (
            f"For **{conditions}**, the historical average was "
            f"**{reference['avg_rentals']:,.2f} rentals/hour** "
            f"across **{count:,} observations** "
            f"(range: {reference['min_rentals']:,}–{reference['max_rentals']:,})."
        )
        if result["reference_sample"] == "small_sample":
            historical += " **Small sample:** treat this comparison cautiously."

    return (
        f"### Estimated demand: {prediction:,.1f} rentals/hour\n\n"
        "**Model:** Optimized XGBoost\n\n"
        f"**Historical reference (2011–2012):** {historical}\n\n"
        "The historical comparison matches hour, working-day status and "
        "weather category—not exact temperature, humidity, wind speed or date. "
        "The model estimate and historical average are different measures.\n\n"
        "*Scenario estimate based on 2011–2012 patterns; not a live forecast "
        "or station-level operational recommendation.*"
    )


def run_scenario(date_text, hour, holiday, weather, temperature, humidity, wind):
    try:
        result = estimate_scenario(
            date_text=date_text,
            hour=_integer_input(hour, "Hour"),
            holiday=holiday,
            weather_code=_integer_input(weather, "Weather category"),
            temperature_c=float(temperature),
            humidity_pct=float(humidity),
            windspeed_kmh=float(wind),
        )
        return format_prediction(result)
    except (TypeError, ValueError, FileNotFoundError, RuntimeError) as exc:
        return f"**Input or setup error:** {exc}"


def answer_question(message: str, history: list | None = None) -> str:
    """Answer one independent question; chat history is display-only."""
    if not isinstance(message, str) or not message.strip():
        return "Enter a question about recorded rentals or a demand scenario."
    try:
        return ask_agent(message)
    except Exception:
        LOGGER.exception("Agent request failed")
        return "The analysis service is unavailable. Check the local server logs."


def load_hourly_patterns() -> tuple[pd.DataFrame, str]:
    """Plot means by hour and day type from read-only SQLite aggregates."""
    empty = pd.DataFrame(columns=CHART_COLUMNS)
    if not DB_PATH.is_file():
        return empty, "**Setup error:** Historical database missing. Run `python -m src.db`."

    records = []
    try:
        for day_type, label in ((0, "Non-working day"), (1, "Working day")):
            for row in hourly_demand(workingday=day_type):
                if row["avg_rentals"] is None:
                    continue
                records.append({
                    "hour": int(row["hr"]),
                    "avg_rentals": float(row["avg_rentals"]),
                    "day_type": label,
                    "observations": int(row["observations"]),
                })
    except (sqlite3.Error, OSError, ValueError):
        LOGGER.exception("Historical profile query failed")
        return empty, "**Data error:** Historical profiles could not be loaded."

    if not records:
        return empty, "No historical hourly records were found."

    chart = pd.DataFrame(records, columns=CHART_COLUMNS).sort_values(
        ["day_type", "hour"], kind="stable"
    )
    return chart, (
        "Each point is the mean rentals per recorded hour in 2011–2012. "
        "Working days exclude weekends and recorded holidays. "
        "These are descriptive averages, not predictions or current demand."
    )


with gr.Blocks(title="Urban Mobility Demand Intelligence") as demo:
    gr.Markdown("# Urban Mobility Demand Intelligence")
    gr.Markdown(
        "Historical bicycle-rental demand analysis based on Capital Bikeshare "
        "records from 2011–2012. No live data or station-level forecasts."
    )

    with gr.Tab("Demand estimate"):
        gr.Markdown("Estimate total hourly rentals for a specified calendar and weather scenario.")
        with gr.Row():
            date_text = gr.Textbox(
                label="Scenario date (YYYY-MM-DD)", value="2012-08-15"
            )
            hour = gr.Slider(0, 23, step=1, value=8, label="Hour of day")
            holiday = gr.Checkbox(label="Holiday", value=False)
        with gr.Row():
            weather = gr.Dropdown(
                choices=[
                    ("Clear", 1),
                    ("Mist / cloudy", 2),
                    ("Light rain / snow", 3),
                    ("Heavy precipitation", 4),
                ],
                value=1,
                label="Weather category",
            )
            temperature = gr.Slider(0, 41, value=22, label="Temperature (°C)")
            humidity = gr.Slider(0, 100, value=60, label="Humidity (%)")
            wind = gr.Slider(0, 67, value=12, label="Wind speed (km/h)")

        predict_button = gr.Button("Estimate demand", variant="primary")
        forecast = gr.Markdown()
        predict_button.click(
            fn=run_scenario,
            inputs=[date_text, hour, holiday, weather, temperature, humidity, wind],
            outputs=forecast,
        )

    with gr.Tab("Ask the agent"):
        gr.Markdown(
            "Ask about recorded hourly demand, working-day patterns, weather, "
            "or a fully specified rental scenario. "
            "**Each question is processed independently; follow-ups must repeat context.** "
            "Responses can contain model-generated errors, so verify critical figures."
        )
        gr.ChatInterface(
            fn=answer_question,
            chatbot=build_chatbot(),
            examples=[
                "Compare average bike rentals on working and non-working days, including sample sizes.",
                "Which three hours have the highest average rentals on working days?",
                "How does rental demand vary by weather category?",
            ],
        )

    with gr.Tab("Historical patterns"):
        gr.Markdown(
            "Compare average recorded hourly rentals on working and non-working days. "
            "The plot is loaded from SQLite; it is not generated by the LLM."
        )
        chart_button = gr.Button("Load hourly profiles")
        chart = gr.LinePlot(
            x="hour",
            y="avg_rentals",
            color="day_type",
            title="Average hourly rentals by day type",
            x_title="Hour of day (0–23)",
            y_title="Average rentals per hour",
            height=350,
        )
        chart_note = gr.Markdown()
        chart_button.click(
            fn=load_hourly_patterns,
            inputs=[],
            outputs=[chart, chart_note],
        )

    gr.Markdown(
        "Source: UCI Bike Sharing Dataset (Fanaee-T, 2013), CC BY 4.0. "
        "Analytical findings reflect historical observations, not present-day activity."
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch()
