from sql_analysis import (
    load_data,
    inspect_data,
    clean_data,
    validate_data,
    create_database,
    verify_database,
    get_overall_kpis,
    sales_by_region,
    recent_orders,
    south_online_orders,
    high_discount_orders,
    largest_orders,
    low_rated_orders,
    region_performance,
    channel_performance,
    large_customer_segments,
    slow_regions,
    city_order_summary,
    joined_order_sample,
    product_profitability,
    category_profitability,
    supplier_summary,
    monthly_sales,
    return_rate_by_category
)

from visualization import (
    plot_sales_by_region,
    plot_monthly_sales,
    plot_category_profitability,
    plot_discount_vs_order_value,
    plot_rating_distribution,
    plot_region_category_heatmap,
    plot_return_rate_by_category,
    save_chart
)

from rag_llm import (
    load_policy,
    split_policy,
    add_chunk_metadata,
    create_embeddings,
    create_vector_store,
    load_vector_store,
    create_retriever,
    retrieve_policy_context,
    inspect_retrieval
)


# MAIN PROGRAM

def main():
    print(
        "\n=== RETAIL ANALYTICS + RAG ===\n"
    )

    # LOAD

    orders, products = load_data()

    # INSPECT RAW DATA

    inspect_data(
        orders,
        products
    )

    # CLEAN

    orders, products = clean_data(
        orders,
        products
    )

    # VALIDATE

    validate_data(
        orders,
        products
    )



    # CREATE DATABASE


    create_database(
        orders,
        products
    )



    # VERIFY DATABASE

    verify_database()

    # KPI ANALYSIS

    kpis = get_overall_kpis()

    print("\nOverall KPIs:")
    print(kpis)




    # REGION ANALYSIS

    region_sales = sales_by_region()

    print("\nSales by region:")
    print(region_sales)


    # VISUALIZATION

    figure = plot_sales_by_region(
        region_sales
    )

    chart_path = save_chart(
        figure,
        "sales_by_region.png"
    )

    print(
        "\nChart saved:",
        chart_path
    )

    return_data = return_rate_by_category()

    return_figure = plot_return_rate_by_category(return_data)

    save_chart(
        return_figure,
        "return_rate_by_category.png"
    )
    # RAG PREPROCESSING

    documents = load_policy()

    chunks = split_policy(documents)

    chunks = add_chunk_metadata(chunks)

    vector_store = load_vector_store()

    retriever = create_retriever(
        vector_store,
        k=3
    )

    question = (
        "Who approves a 12 percent discount?"
    )

    documents = retrieve_policy_context(
        retriever,
        question
    )

    inspect_retrieval(
        documents
    )
# RUN PROGRAM

if __name__ == "__main__":
    main()