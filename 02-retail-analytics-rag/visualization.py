from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

# OUTPUT FOLDER

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(exist_ok=True)

# SAVE CHART

def save_chart(figure, filename):

    filepath = OUTPUT_DIR / filename

    figure.savefig(
        filepath,
        bbox_inches="tight"
    )

    return filepath



# BAR CHART

def plot_sales_by_region(df):

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    sns.barplot(
        data=df,
        x="region",
        y="sales",
        ax=ax
    )

    ax.set_title(
        "Net Sales by Region"
    )

    ax.set_xlabel(
        "Region"
    )

    ax.set_ylabel(
        "Sales"
    )

    fig.tight_layout()

    return fig

def plot_monthly_sales(df):
    """Plot monthly sales over time."""

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.lineplot(
        data=df,
        x="month",
        y="total_sales",
        marker="o",
        ax=ax
    )

    ax.set_title("Monthly net sales")
    ax.set_xlabel("Month")
    ax.set_ylabel("Net sales")
    ax.tick_params(axis="x", rotation=45)
    
    fig.tight_layout()
    return fig

def plot_category_profitability(df):
    """Plot total profit by product category."""

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.barplot(
        data=df,
        x="category",
        y="total_profit",
        ax=ax
    )

    ax.set_title(
        "Profit by product category"
    )

    ax.set_xlabel(
        "Category"
    )

    ax.set_ylabel(
        "Total profit"
    )

    fig.tight_layout()

    return fig

def plot_discount_vs_order_value(df):
    # scatter plot
    pass


def plot_rating_distribution(df):
    # histogram
    pass


def plot_region_category_heatmap(df):
    # heatmap
    pass