from llm import get_llm
from retriever import retrieve


SYSTEM_RULES = """You are a customer-support decision-support assistant.
Answer only from the supplied context.
If the context is insufficient, say so clearly.
Do not invent policy.
Separate policy guidance from historical-case examples.
"""


def format_context(documents):
    blocks = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        source_type = doc.metadata.get("source_type", "unknown")
        blocks.append(
            f"[Source {i} | {source_type} | {source}]\n{doc.page_content}"
        )
    return "\n\n".join(blocks)


def answer_question(question, mode="mmr"):
    documents = retrieve(question, mode=mode, k=4)
    context = format_context(documents)

    # TODO Task 9:
    # Build messages for the LLM using SYSTEM_RULES, context, and question.
    # Return both answer text and source names.
    raise NotImplementedError
