import gradio as gr

from database import get_ticket
from ml_model import CATEGORICAL_FEATURES, NUMERIC_FEATURES, predict_escalation
from rag_answer import answer_question


def analyze(ticket_id, question, retriever_mode):
    ticket = get_ticket(ticket_id.strip())
    if ticket is None:
        return "Ticket not found.", "", ""

    model_features = {
        key: ticket.get(key)
        for key in NUMERIC_FEATURES + CATEGORICAL_FEATURES
    }
    risk = predict_escalation(model_features)

    # TODO Task 10:
    # Enrich the user's question with useful ticket context,
    # call answer_question(), and return:
    # 1. escalation prediction
    # 2. generated guidance
    # 3. retrieved source list
    raise NotImplementedError


with gr.Blocks(title="Support Resolution Intelligence") as demo:
    gr.Markdown("# Support Resolution Intelligence")
    gr.Markdown("Escalation-risk prediction + evidence-grounded support guidance")

    ticket_id = gr.Textbox(label="Ticket ID", value="T00001")
    question = gr.Textbox(
        label="Support question",
        placeholder="What policy applies and should this case be escalated?",
        lines=3,
    )
    retriever_mode = gr.Radio(
        ["baseline", "mmr", "parent"],
        value="parent",
        label="Retriever",
    )
    run = gr.Button("Analyze")

    risk_output = gr.Textbox(label="ML escalation assessment")
    answer_output = gr.Textbox(label="RAG guidance", lines=10)
    source_output = gr.Textbox(label="Sources")

    run.click(
        analyze,
        inputs=[ticket_id, question, retriever_mode],
        outputs=[risk_output, answer_output, source_output],
    )


if __name__ == "__main__":
    demo.launch()
