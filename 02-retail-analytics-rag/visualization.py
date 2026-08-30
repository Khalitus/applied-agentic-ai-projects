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
    # line chart
    pass


def plot_discount_vs_order_value(df):
    # scatter plot
    pass


def plot_rating_distribution(df):
    # histogram
    pass


def plot_region_category_heatmap(df):
    # heatmap
    pass