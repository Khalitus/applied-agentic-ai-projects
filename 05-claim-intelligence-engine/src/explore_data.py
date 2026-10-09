from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import PROCESSED_DATA_DIR, ROOT_DIR


DATA_PATH = PROCESSED_DATA_DIR / "claims_processed.csv"
FIGURE_DIR = ROOT_DIR / "reports" / "figures"


def analyze_target(df: pd.DataFrame):
    target = df["claim_amount"]

    print("\nClaim amount statistics")
    print(target.describe())

    q1 = target.quantile(0.25)
    q3 = target.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = (
        (target < lower_bound)
        | (target > upper_bound)
    )

    print("\nOutlier analysis")
    print(f"Lower bound: {lower_bound:.2f}")
    print(f"Upper bound: {upper_bound:.2f}")
    print(f"Potential outliers: {outliers.sum()}")


def plot_claim_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        df["claim_amount"],
        bins=30,
        edgecolor="black",
    )

    ax.set_title("Claim amount distribution")
    ax.set_xlabel("Claim amount ($)")
    ax.set_ylabel("Frequency")

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "claim_distribution.png",
        dpi=150,
    )

    plt.close(fig)

def plot_average_claim_by_severity(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))

    severity_order = ["minor", "moderate", "severe"]

    averages = (
        df.groupby("damage_severity")["claim_amount"]
        .mean()
        .reindex(severity_order)
    )

    ax.bar(
        [severity.title() for severity in averages.index],
        averages.values,
        edgecolor="black",
        linewidth=1.2,
        width=0.55,
    )

    ax.set_title("Average claim by severity")
    ax.set_xlabel("Severity")
    ax.set_ylabel("Average claim")

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "average_claim_by_severity.png",
        dpi=150,
    )

    plt.close(fig)

def plot_repair_vs_claim(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(df["repair_estimate"], df["claim_amount"])

    ax.set_title("Repair estimate vs claim amount")
    ax.set_xlabel("Repair estimate")
    ax.set_ylabel("Claim amount")

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "repair_vs_claim.png",
        dpi=150,
    )

    plt.close(fig)

def plot_incident_frequency(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        ["Collision", "Glass", "Hail", "Vandalism", "Theft"], 
        df["incident_type"].value_counts(), 
        edgecolor='black', 
        linewidth=1.2, 
        width=0.55
    )

    ax.set_title("Incident type frequency")
    ax.set_xlabel("Type of Incident")
    ax.set_ylabel("Frequency")

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "Incident_frequency.png",
        dpi=150,
    )

    plt.close(fig)

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Run python -m src.clean_data first."
        )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    claims = pd.read_csv(DATA_PATH)

    analyze_target(claims)
    plot_claim_distribution(claims)

    plot_average_claim_by_severity(claims)
    plot_repair_vs_claim(claims)
    plot_incident_frequency(claims)

    print("\nEDA complete")


if __name__ == "__main__":
    main()