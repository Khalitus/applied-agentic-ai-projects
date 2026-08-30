from sql_analysis import (
    load_data,
    clean_data,
    create_database,
    get_overall_kpis,
    sales_by_region
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

    print("\n=== RETAIL ANALYTICS + RAG ===\n")


    # LOAD DATA

    orders, products = load_data()

    print("Orders loaded:", len(orders))
    print("Products loaded:", len(products))


    # CLEAN DATA

    orders, products = clean_data(
        orders,
        products
    )

    print("Data cleaning complete.")


    # CREATE SQL DATABASE

    create_database(
        orders,
        products
    )

    print("SQLite database created.")


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