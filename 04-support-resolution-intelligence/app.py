import gradio as gr

from database import get_ticket
from ml_model import CATEGORICAL_FEATURES, NUMERIC_FEATURES, predict_escalation
from rag_answer import generate_answer

def build_ticket_context(ticket):
    fields = [
        "issue_type",
        "priority",
        "channel",
        "customer_tier",
        "category",
        "region",
        "sentiment_score",
        "previous_tickets_90d",
        "days_since_purchase",
    ]

    lines = []

    for field in fields:
        value = ticket

        if value is not None:
            lines.append(
                f"{field}: {value}"
            )

    return "\n".join(lines)

def analyze(ticket_id, question, retriever_mode):
    ticket = get_ticket(ticket_id.strip())

    if ticket is None:
        return "Ticket not found.", "", ""

    model_features = {
        key: ticket.get(key)
        for key in NUMERIC_FEATURES + CATEGORICAL_FEATURES
    }

    risk = predict_escalation(model_features)

    risk_label = (
        "Escalation likely"
        if risk["prediction"]
        else "Escalation not predicted"
    )

    risk_text = (
        f"{risk_label}\n"
        f"Probability: {risk['probability']:.1%}"
    )

    ticket_context = build_ticket_context(ticket)

    rag_question = (
        f"Ticket context:\n"
        f"{ticket_context}\n\n"
        f"Support question:\n"
        f"{question.strip()}"
    )

    rag_result = generate_answer(
        rag_question,
        strategy=retriever_mode,
        k=4,
    )

    source_lines = []

    for source in rag_result["sources"]:
        source_lines.append(
            f"{source["source_type"]}: {source["source"]}"

        )

    source_text = "\n".join(source_lines)

    return (
        risk_text,
        rag_result["answer"],
        source_text,
    )


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
