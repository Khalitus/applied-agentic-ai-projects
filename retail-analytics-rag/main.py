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
    product_profitability
)

from visualization import (
    plot_sales_by_region,
    save_chart
)

from rag_llm import (
    load_policy,
    split_policy
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


    # RAG PREPROCESSING

    documents = load_policy()

    chunks = split_policy(
        documents
    )

    print(
        "\nPolicy chunks created:",
        len(chunks)
    )

    print(
        "\nFirst chunk preview:"
    )

    print(
        chunks[0].page_content[:300]
    )


# RUN PROGRAM

if __name__ == "__main__":
    main()