"""LangGraph routing over SQL/ML services with optional LLM narration."""

import json
import os
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

from src.predict import predict_subscription
from src.sql_queries import query_history

load_dotenv()


class AgentState(TypedDict, total=False):
    mode: Literal["history", "predict"]
    question: str
    segment: str
    customer: dict
    evidence: dict
    explanation: str


def supervisor(state: AgentState):
    # Intentionally deterministic: the UI has already selected an explicit mode.
    return {}


def route(state: AgentState):
    if state["mode"] not in ("history", "predict"):
        raise ValueError("Unknown operation")
    return state["mode"]


def historical_analyst(state: AgentState):
    return {"evidence": query_history(segment=state.get("segment", "job"))}


def prediction_analyst(state: AgentState):
    return {"evidence": predict_subscription(state["customer"])}


def narrator(state: AgentState):
    evidence = json.dumps(state["evidence"], ensure_ascii=False)
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        return {"explanation": "Offline evidence (no API required):\n" + evidence}

    # The LLM describes supplied tool outputs; it does not calculate predictions.
    try:
        from langchain_groq import ChatGroq
        model = ChatGroq(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            temperature=0,
            max_retries=1,
            timeout=25,
        )
        message = model.invoke([
            ("system", "You explain historical bank-marketing analytics to an analyst. "
             "Use ONLY the supplied evidence. Do not make up numbers, causal claims, "
             "individual guarantees, fairness claims or financial advice. "
             "Historical evidence may not generalize. Suggest human review, not automated outreach. "
             "Keep answer under 130 words."),
            ("human", f"Question: {state.get('question', 'Summarize the results')}\n"
             f"Structured evidence: {evidence}"),
        ])
        return {"explanation": str(message.content)}
    except Exception as exc:
        return {"explanation": f"LLM narration unavailable ({type(exc).__name__}).\nEvidence: {evidence}"}


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("supervisor", supervisor)
    builder.add_node("historical_analyst", historical_analyst)
    builder.add_node("prediction_analyst", prediction_analyst)
    builder.add_node("narrator", narrator)
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor", route,
        {"history": "historical_analyst", "predict": "prediction_analyst"}
    )
    builder.add_edge("historical_analyst", "narrator")
    builder.add_edge("prediction_analyst", "narrator")
    builder.add_edge("narrator", END)
    return builder.compile()


graph = build_graph()
