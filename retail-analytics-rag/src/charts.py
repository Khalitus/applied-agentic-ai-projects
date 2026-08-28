import matplotlib.pyplot as plt
import seaborn as sns

from src.database import run_query

def monthly_sales_chart():
    df = run_query("""
        SELECT
            substr(order_date, 1, 7) AS month,
            ROUND(SUM(quantity * unit_price * (1 - discount_pct)), 2) AS net_sales
        FROM orders
        GROUP BY month
        ORDER BY month
    """)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=df, x="month", y="net_sales", marker="o", ax=ax)
    ax.set_title("Monthly Net Sales")
    ax.set_xlabel("Month")
    ax.set_ylabel("Net Sales")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig

def category_profit_chart():
    df = run_query("""
        SELECT
            p.category,
            ROUND(SUM(
                o.quantity * o.unit_price * (1 - o.discount_pct)
                - o.quantity * p.unit_cost
            ), 2) AS profit
        FROM orders AS o
        INNER JOIN products AS p
            ON o.product_id = p.product_id
        GROUP BY p.category
        ORDER BY profit DESC
    """)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=df, x="category", y="profit", ax=ax)
    ax.set_title("Profit by Product Category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Profit")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig

def region_category_heatmap():
    df = run_query("""
        SELECT
            o.region,
            p.category,
            ROUND(SUM(o.quantity * o.unit_price * (1 - o.discount_pct)), 2) AS net_sales
        FROM orders AS o
        INNER JOIN products AS p
            ON o.product_id = p.product_id
        GROUP BY o.region, p.category
    """)

    pivot = df.pivot(index="region", columns="category", values="net_sales")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt=".0f", ax=ax)
    ax.set_title("Net Sales by Region and Category")
    fig.tight_layout()
    return fig

def discount_vs_order_value_scatter():
    df = run_query("""
        SELECT
            p.category,
            o.discount_pct,
            (o.quantity * o.unit_price * (1 - o.discount_pct)) AS order_value
        FROM orders AS o
        INNER JOIN products AS p
            ON o.product_id = p.product_id
    """)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(
        data=df,
        x="discount_pct",
        y="order_value",
        hue="category",
        alpha=0.65,
        ax=ax
    )
    ax.set_title("Discount vs Order Value")
    ax.set_xlabel("Discount %")
    ax.set_ylabel("Order Value")
    fig.tight_layout()
    return fig
