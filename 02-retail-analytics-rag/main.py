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
# MAIN PROGRAM

def main():
    print(
        "\n=== RETAIL ANALYTICS + RAG ===\n"
    )

# RUN PROGRAM

if __name__ == "__main__":
    main()