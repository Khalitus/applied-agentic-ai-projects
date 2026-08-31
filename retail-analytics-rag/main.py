import gradio as gr

from sql_analysis import (
    load_data,
    clean_data,
    validate_data,
    create_database,
    get_overall_kpis,
    sales_by_region,
    recent_orders,
    high_discount_orders,
    largest_orders,
    low_rated_orders,
    region_performance,
    channel_performance,
    city_order_summary,
    monthly_sales,
    product_profitability,
    category_profitability,
    supplier_summary,
    return_rate_by_category
)

from visualization import (
    plot_sales_by_region,
    plot_monthly_sales,
    plot_category_profitability,
    plot_discount_vs_order_value,
    plot_rating_distribution,
    plot_region_category_heatmap,
    plot_return_rate_by_category
)

from rag_llm import (
    VECTOR_DB_PATH,
    load_policy,
    split_policy,
    add_chunk_metadata,
    create_vector_store,
    ask_policy,
    format_policy_result
)


def initialize_project():
    """Prepare the datasets, database and RAG index."""

    orders, products = load_data()
    orders, products = clean_data(
        orders,
        products
    )

    validation = validate_data(
        orders,
        products
    )

    if any(validation.values()):
        raise ValueError(
            "Data validation failed."
        )

    create_database(
        orders,
        products
    )

    if (
        not VECTOR_DB_PATH.exists()
        or not any(VECTOR_DB_PATH.iterdir())
    ):
        documents = load_policy()

        chunks = split_policy(
            documents
        )

        chunks = add_chunk_metadata(
            chunks
        )

        create_vector_store(
            chunks
        )

    return orders, products

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
    "Largest orders": largest_orders,
    "High discount orders": high_discount_orders,
    "Low rated orders": low_rated_orders
}

def run_analysis(analysis_name):
    """Run the selected business analysis."""
    return ANALYSIS_FUNCTIONS[
        analysis_name
    ]()

def generate_chart(chart_name):
    """Generate the selected visualization."""

    if chart_name == "Sales by region":
        return plot_sales_by_region(
            sales_by_region()
        )

    if chart_name == "Monthly sales":
        return plot_monthly_sales(
            monthly_sales()
        )

    if chart_name == "Category profitability":
        return plot_category_profitability(
            category_profitability()
        )

    if chart_name == "Discount vs order value":
        return plot_discount_vs_order_value(
            orders
        )

    if chart_name == "Rating distribution":
        return plot_rating_distribution(
            orders
        )

    if chart_name == "Region-category heatmap":
        return plot_region_category_heatmap(
            orders,
            products
        )

    if chart_name == "Return rate by category":
        return plot_return_rate_by_category(
            return_rate_by_category()
        )

def answer_policy_question(question):
    """Answer a company policy question."""

    try:
        result = ask_policy(
            question
        )

        return format_policy_result(
            result
        )

    except Exception as error:
        print(
            f"Policy assistant error: {error}"
        )

        return (
            "The policy assistant is "
            "temporarily unavailable."
        )

def build_app():
    """Build the Gradio interface."""

    kpis = get_overall_kpis().iloc[0]

    with gr.Blocks(
        title="Retail analytics + RAG assistant"
    ) as app:

        gr.Markdown(
            """
# Retail analytics + RAG assistant

Explore retail performance, visualize business trends,
and ask grounded questions about company policies.
"""
        )

        with gr.Row():
            gr.Number(
                value=int(kpis["total_orders"]),
                label="Total orders",
                interactive=False
            )

            gr.Number(
                value=int(kpis["units_sold"]),
                label="Units sold",
                interactive=False
            )

            gr.Number(
                value=float(kpis["total_sales"]),
                label="Net sales",
                interactive=False
            )

            gr.Number(
                value=float(
                    kpis["average_order_value"]
                ),
                label="Average order value",
                interactive=False
            )
        with gr.Tab("Analytics"):

            analysis_choice = gr.Dropdown(
                choices=list(
                    ANALYSIS_FUNCTIONS.keys()
                ),
                value="Region performance",
                label="Business analysis"
            )

            analysis_button = gr.Button(
                "Run analysis"
            )

            analysis_output = gr.Dataframe(
                label="Results",
                interactive=False
            )

            analysis_button.click(
                fn=run_analysis,
                inputs=analysis_choice,
                outputs=analysis_output
            )
        with gr.Tab("Visualizations"):

            chart_choice = gr.Dropdown(
                choices=[
                    "Sales by region",
                    "Monthly sales",
                    "Category profitability",
                    "Discount vs order value",
                    "Rating distribution",
                    "Region-category heatmap",
                    "Return rate by category"
                ],
                value="Sales by region",
                label="Visualization"
            )

            chart_button = gr.Button(
                "Generate chart"
            )

            chart_output = gr.Plot(
                label="Chart"
            )

            chart_button.click(
                fn=generate_chart,
                inputs=chart_choice,
                outputs=chart_output
            )
        with gr.Tab("Policy assistant"):

            gr.Markdown(
                """
Ask questions about returns, shipping,
discount approvals, and customer service policies.
"""
            )

            policy_question = gr.Textbox(
                label="Policy question",
                placeholder=(
                    "Who approves a 12% discount?"
                ),
                lines=2
            )

            policy_button = gr.Button(
                "Ask policy assistant"
            )

            gr.Examples(
                examples=[
                    [
                        "Who approves a 12 percent discount?"
                    ],
                    [
                        "How long does delivery take in the West region?"
                    ],
                    [
                        "Can an opened non-defective product be returned?"
                    ],
                    [
                        "What happens when a customer gives a rating of 2?"
                    ]
                ],
                inputs=policy_question
            )

            policy_answer = gr.Markdown()

            policy_button.click(
                fn=answer_policy_question,
                inputs=policy_question,
                outputs=policy_answer
            )

            policy_question.submit(
                fn=answer_policy_question,
                inputs=policy_question,
                outputs=policy_answer
            )
# MAIN PROGRAM

def main():
    print(
        "\n=== RETAIL ANALYTICS + RAG ===\n"
    )

# RUN PROGRAM

if __name__ == "__main__":
    main()