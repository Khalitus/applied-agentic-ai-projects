import matplotlib.pyplot as plt
import seaborn as sns


def sales_by_region_chart(df):

    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=df,
        x="region",
        y="sales"
    )

    plt.title("Sales by Region")

    return plt.gcf()