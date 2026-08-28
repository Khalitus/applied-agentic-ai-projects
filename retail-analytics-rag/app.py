import gradio as gr

from src.sql_queries import (
    recent_south_orders,
    sales_by_region,
    high_volume_categories,
    product_profitability,
    data_quality_summary,
)
from src.charts import (
    monthly_sales_chart,
    category_profit_chart,
    region_category_heatmap,
    discount_vs_order_value_scatter,
)

_rag_chain = None

def run_business_question(choice):
    mapping = {
        "Recent South-region orders": recent_south_orders,
        "Sales and profit by region": sales_by_region,
        "High-volume categories": high_volume_categories,
        "Product profitability": product_profitability,
        "Data quality summary": data_quality_summary,
    }
    return mapping[choice]()

def make_chart(choice):
    mapping = {
        "Monthly sales trend": monthly_sales_chart,
        "Profit by category": category_profit_chart,
        "Region/category heatmap": region_category_heatmap,
        "Discount vs order value": discount_vs_order_value_scatter,
    }
    return mapping[choice]()

def ask_policy(question):
    global _rag_chain

    try:
        if _rag_chain is None:
            from src.rag_pipeline import build_rag_chain
            _rag_chain = build_rag_chain()

        from src.rag_pipeline import answer_question
        result = answer_question(_rag_chain, question)

        source_text = ", ".join(result["sources"]) if result["sources"] else "No source returned"
        return result["answer"], source_text

    except Exception as exc:
        return (
            f"RAG is not ready yet: {exc}",
            "SQL and visualization tabs still work without an LLM key."
        )

with gr.Blocks(title="Retail Analytics & Policy Assistant") as demo:
    gr.Markdown("""
    # Retail Analytics & Policy Assistant
    Month-1 capstone: **SQL + Visualization + RAG**

    **Structured facts → SQL · Patterns → charts · Policy knowledge → RAG**
    """)

    with gr.Tab("SQL Analytics"):
        question = gr.Dropdown(
            choices=[
                "Recent South-region orders",
                "Sales and profit by region",
                "High-volume categories",
                "Product profitability",
                "Data quality summary",
            ],
            value="Sales and profit by region",
            label="Choose a business question"
        )
        run_btn = gr.Button("Run SQL")
        table = gr.Dataframe(label="Query result")
        run_btn.click(run_business_question, inputs=question, outputs=table)

    with gr.Tab("Visualizations"):
        chart_choice = gr.Dropdown(
            choices=[
                "Monthly sales trend",
                "Profit by category",
                "Region/category heatmap",
                "Discount vs order value",
            ],
            value="Monthly sales trend",
            label="Choose a chart"
        )
        chart_btn = gr.Button("Create chart")
        plot = gr.Plot(label="Visualization")
        chart_btn.click(make_chart, inputs=chart_choice, outputs=plot)

    with gr.Tab("Policy RAG"):
        policy_question = gr.Textbox(
            label="Ask a policy question",
            placeholder="Example: Who must approve a 12% discount?"
        )
        ask_btn = gr.Button("Ask")
        answer = gr.Textbox(label="Answer")
        sources = gr.Textbox(label="Sources")
        ask_btn.click(ask_policy, inputs=policy_question, outputs=[answer, sources])

if __name__ == "__main__":
    demo.launch()
