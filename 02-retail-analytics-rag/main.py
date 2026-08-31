import gradio as gr

from rag_llm import (
    VECTOR_DB_PATH,
    add_chunk_metadata,
    ask_policy,
    create_vector_store,
    format_policy_result,
    load_policy,
    split_policy,
)
from sql_analysis import (
    category_profitability,
    channel_performance,
    city_order_summary,
    clean_data,
    create_database,
    get_overall_kpis,
    high_discount_orders,
    largest_orders,
    load_data,
    low_rated_orders,
    monthly_sales,
    product_profitability,
    recent_orders,
    region_performance,
    return_rate_by_category,
    sales_by_region,
    south_online_orders,
    supplier_summary,
    validate_data,
)
from visualization import (
    plot_category_profitability,
    plot_discount_vs_order_value,
    plot_monthly_sales,
    plot_rating_distribution,
    plot_region_category_heatmap,
    plot_return_rate_by_category,
    plot_sales_by_region,
)


ANALYSIS_FUNCTIONS = {
    "Region performance": region_performance,
    "Channel performance": channel_performance,
    "Monthly sales": monthly_sales,
    "Product profitability": product_profitability,
    "Category profitability": category_profitability,
    "Supplier summary": supplier_summary,
    "Return rate by category": return_rate_by_category,
    "City order summary": city_order_summary,
    "Recent orders": recent_orders,
    "South online orders": south_online_orders,
    "Largest orders": largest_orders,
    "High discount orders": high_discount_orders,
    "Low rated orders": low_rated_orders,
}


def initialize_project():
    """Prepare the datasets, database and RAG index."""
    orders, products = load_data()
    orders, products = clean_data(orders, products)

    validation = validate_data(orders, products)
    if any(validation.values()):
        raise ValueError("Data validation failed.")

    create_database(orders, products)

    if not VECTOR_DB_PATH.exists() or not any(VECTOR_DB_PATH.iterdir()):
        documents = load_policy()
        chunks = split_policy(documents)
        chunks = add_chunk_metadata(chunks)
        create_vector_store(chunks)

    return orders, products


def run_analysis(analysis_name):
    """Run the selected business analysis."""
    return ANALYSIS_FUNCTIONS[analysis_name]()


def answer_policy_question(question):
    """Answer a company policy question."""
    try:
        return format_policy_result(ask_policy(question))
    except Exception as error:
        print(f"Policy assistant error: {error}")
        return "The policy assistant is temporarily unavailable."


def build_app(orders, products):
    """Build the Gradio interface."""
    chart_functions = {
        "Sales by region": lambda: plot_sales_by_region(sales_by_region()),
        "Monthly sales": lambda: plot_monthly_sales(monthly_sales()),
        "Category profitability": lambda: plot_category_profitability(
            category_profitability()
        ),
        "Discount vs order value": lambda: plot_discount_vs_order_value(orders),
        "Rating distribution": lambda: plot_rating_distribution(orders),
        "Region-category heatmap": lambda: plot_region_category_heatmap(
            orders, products
        ),
        "Return rate by category": lambda: plot_return_rate_by_category(
            return_rate_by_category()
        ),
    }

    def generate_chart(chart_name):
        return chart_functions[chart_name]()

    kpis = get_overall_kpis().iloc[0]

    with gr.Blocks(title="Retail analytics + RAG assistant") as app:
        gr.Markdown(
            """
# Retail analytics + RAG assistant

Explore retail performance, visualize business trends, and ask grounded questions about company policies.
"""
        )

        with gr.Row():
            gr.Number(
                value=int(kpis["total_orders"]),
                label="Total orders",
                interactive=False,
            )
            gr.Number(
                value=int(kpis["units_sold"]),
                label="Units sold",
                interactive=False,
            )
            gr.Number(
                value=float(kpis["total_sales"]),
                label="Net sales",
                interactive=False,
            )
            gr.Number(
                value=float(kpis["average_order_value"]),
                label="Average order value",
                interactive=False,
            )

        with gr.Tab("Analytics"):
            analysis_choice = gr.Dropdown(
                choices=list(ANALYSIS_FUNCTIONS),
                value="Region performance",
                label="Business analysis",
            )
            analysis_button = gr.Button("Run analysis")
            analysis_output = gr.Dataframe(
                label="Results",
                interactive=False,
            )

            analysis_button.click(
                fn=run_analysis,
                inputs=analysis_choice,
                outputs=analysis_output,
            )

        with gr.Tab("Visualizations"):
            chart_choice = gr.Dropdown(
                choices=list(chart_functions),
                value="Sales by region",
                label="Visualization",
            )
            chart_button = gr.Button("Generate chart")
            chart_output = gr.Plot(label="Chart")

            chart_button.click(
                fn=generate_chart,
                inputs=chart_choice,
                outputs=chart_output,
            )

        with gr.Tab("Policy assistant"):
            gr.Markdown(
                "Ask questions about returns, shipping, discount approvals, and customer service policies."
            )

            policy_question = gr.Textbox(
                label="Policy question",
                placeholder="Who approves a 12% discount?",
                lines=2,
            )
            policy_button = gr.Button("Ask policy assistant")
            policy_answer = gr.Markdown()

            gr.Examples(
                examples=[
                    ["Who approves a 12 percent discount?"],
                    ["How long does delivery take in the West region?"],
                    ["Can an opened non-defective product be returned?"],
                    ["What happens when a customer gives a rating of 2?"],
                    ["When does a delayed shipment need to be escalated?"],
                ],
                inputs=policy_question,
            )

            policy_button.click(
                fn=answer_policy_question,
                inputs=policy_question,
                outputs=policy_answer,
            )
            policy_question.submit(
                fn=answer_policy_question,
                inputs=policy_question,
                outputs=policy_answer,
            )

    return app


def main():
    orders, products = initialize_project()
    app = build_app(orders, products)
    app.launch(share=True)


if __name__ == "__main__":
    main()
