"""Two-mode Gradio interface: historical SQL exploration + user-input ML inference."""

import gradio as gr

from src.agent import graph
from src.features import (
    EDUCATION_CHOICES, JOB_CHOICES, MARITAL_CHOICES, POUTCOME_CHOICES
)


def run_history(segment, question):
    try:
        state = graph.invoke({"mode": "history", "segment": segment, "question": question})
        return state["explanation"], state["evidence"]
    except Exception as exc:
        return f"Error: {exc}", {}


def run_prediction(age, job, marital, education, default, balance, housing,
                   loan, pdays, previous, poutcome):
    payload = dict(age=age, job=job, marital=marital, education=education,
                   default=default, balance=balance, housing=housing,
                   loan=loan, pdays=pdays, previous=previous, poutcome=poutcome)
    try:
        state = graph.invoke({"mode": "predict", "customer": payload,
                              "question": "Explain the estimated probability and its limitations."})
        result = state["evidence"]
        summary = (f"### Estimated subscription likelihood: {result['probability']:.1%}\n\n"
                   f"Default 50% classification threshold: "
                   f"{'above' if result['above_default_threshold'] else 'below'}\n\n"
                   f"**Interpretation:**\n\n{state['explanation']}\n\n"
                   "*Model estimates are not guarantees. No customer decisions are automated.*")
        return summary, result
    except Exception as exc:
        return f"Error: {exc}", {}


with gr.Blocks(title="Campaign Intelligence Agent") as demo:
    gr.Markdown("# Campaign Intelligence Agent")
    gr.Markdown("UCI Bank Marketing | SQLite analytics | ML inference | LangGraph narration")
    with gr.Tab("Historical analytics"):
        segment = gr.Dropdown(["job", "education", "poutcome"], value="job", label="Group by")
        question = gr.Textbox(value="What do these historical subscription rates suggest?",
                              label="Analyst question", max_lines=2)
        analyze = gr.Button("Analyze historical data", variant="primary")
        history_text = gr.Markdown()
        history_json = gr.JSON(label="SQL evidence")
        analyze.click(run_history, [segment, question], [history_text, history_json])
    with gr.Tab("New customer prediction"):
        gr.Markdown("Enter information available **before** the next marketing contact.")
        with gr.Row():
            age = gr.Number(value=38, precision=0, label="Age")
            balance = gr.Number(value=2400, precision=0, label="Average balance (EUR)")
            job = gr.Dropdown(JOB_CHOICES, value="technician", label="Occupation")
        with gr.Row():
            marital = gr.Dropdown(MARITAL_CHOICES, value="married", label="Marital status")
            education = gr.Dropdown(EDUCATION_CHOICES, value="secondary", label="Education")
            default = gr.Dropdown(["no", "yes"], value="no", label="Credit default")
        with gr.Row():
            housing = gr.Dropdown(["no", "yes"], value="yes", label="Housing loan")
            loan = gr.Dropdown(["no", "yes"], value="no", label="Personal loan")
            pdays = gr.Number(value=-1, precision=0, label="Days since prior campaign (-1 = none)")
        with gr.Row():
            previous = gr.Number(value=0, precision=0, label="Previous contacts")
            poutcome = gr.Dropdown(POUTCOME_CHOICES, value="unknown", label="Previous outcome")
        predict = gr.Button("Predict subscription likelihood", variant="primary")
        prediction_text = gr.Markdown()
        prediction_json = gr.JSON(label="Structured model result")
        predict.click(
            run_prediction,
            [age, job, marital, education, default, balance, housing,
             loan, pdays, previous, poutcome],
            [prediction_text, prediction_json],
        )


if __name__ == "__main__":
    demo.launch()
