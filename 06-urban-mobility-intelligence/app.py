"""Interactive historical-pattern hourly bike-demand estimation."""

import math

import gradio as gr

from src.predict import estimate_scenario


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


with gr.Blocks(title="Urban Mobility Demand Intelligence") as demo:
    gr.Markdown("# Urban Mobility Demand Intelligence")
    gr.Markdown(
        "Estimate hourly bicycle rentals from historical Capital Bikeshare "
        "patterns (2011–2012). Results are illustrative, not real-time forecasts."
    )
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
    gr.Markdown(
        "Source: UCI Bike Sharing Dataset (Fanaee-T, 2013), CC BY 4.0. "
        "The reference sample size indicates data availability, not prediction certainty."
    )


if __name__ == "__main__":
    demo.launch()
