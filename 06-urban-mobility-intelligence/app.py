"""Minimal application: a real interactive ML baseline and an agent exercise."""
import gradio as gr

from src.agent import ask_agent
from src.analytics import hourly_profile
from src.predict import predict_demand


def run_scenario(date_text, hour, holiday, weather, temperature, humidity, wind):
    try:
        hour = int(hour)
        result = predict_demand(
            date_text=date_text,
            hour=hour,
            holiday=bool(holiday),
            weather_code=int(weather),
            temperature_c=float(temperature),
            humidity_pct=float(humidity),
            windspeed_kmh=float(wind),
        )
        past = hourly_profile(hour)
        reference = past.get("avg_rentals")
        context = (
            f"Historical all-weather average at {hour:02d}:00: "
            f"**{reference} rentals/hour** across "
            f"{past.get('observations', 0)} observations."
            if reference is not None else "No historical reference available."
        )
        return (
            f"### Estimated hourly demand: {result:,.0f} rentals\n\n"
            f"{context}\n\n"
            "*Educational scenario using historical 2011–2012 patterns. "
            "Not a live forecast. Weather conditions differ across the historical average.*"
        )
    except (ValueError, FileNotFoundError) as exc:
        return f"**Input/setup error:** {exc}"


with gr.Blocks(title="Urban Mobility Demand Intelligence") as demo:
    gr.Markdown("# Urban Mobility Demand Intelligence")
    gr.Markdown("Historical bike rental analysis, predictive ML, and a tool-using agent (next task).")
    with gr.Tab("Demand Prediction"):
        with gr.Row():
            date_text = gr.Textbox(label="Scenario date (YYYY-MM-DD)", value="2012-08-15")
            hour = gr.Slider(0, 23, step=1, value=8, label="Hour of day")
            holiday = gr.Checkbox(label="Holiday", value=False)
        with gr.Row():
            weather = gr.Dropdown(
                choices=[("Clear", 1), ("Mist/cloudy", 2),
                         ("Light rain/snow", 3), ("Heavy precipitation", 4)],
                value=1, label="Weather category",
            )
            temperature = gr.Slider(0, 41, value=22, label="Approx. temperature (°C)")
            humidity = gr.Slider(0, 100, value=60, label="Humidity (%)")
            wind = gr.Slider(0, 67, value=12, label="Wind speed (km/h)")
        predict_button = gr.Button("Predict hourly demand", variant="primary")
        forecast = gr.Markdown()
        predict_button.click(
            run_scenario,
            [date_text, hour, holiday, weather, temperature, humidity, wind],
            forecast,
        )
    with gr.Tab("Ask the Agent"):
        question = gr.Textbox(label="Question", lines=3,
                              placeholder="How busy are weekday mornings?")
        ask_button = gr.Button("Ask")
        answer = gr.Markdown()
        ask_button.click(ask_agent, question, answer)
    gr.Markdown("Source: UCI Bike Sharing Dataset (Fanaee-T, 2013), CC BY 4.0.")


if __name__ == "__main__":
    demo.launch()
