
"""Explore and audit historical bike-sharing data."""

import pandas as pd

from src.data import load_hourly


def main() -> None:
    df = load_hourly()

    print("\nDataset overview")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {df.shape[1]}")
    print(f"Unique dates: {df['dteday'].nunique()}")
    print(f"Date range: {df['dteday'].min()} to {df['dteday'].max()}")

    print("\nColumn types")
    print(df.dtypes.to_string())

    print("\nData quality")
    print(f"Missing values: {int(df.isna().sum().sum())}")
    print(f"Duplicate records: {int(df.duplicated().sum())}")
    print(f"Duplicate IDs: {int(df['instant'].duplicated().sum())}")

    timestamps = (
        pd.to_datetime(df["dteday"])
        + pd.to_timedelta(df["hr"], unit="h")
    )

    print(f"Duplicate timestamps: {int(timestamps.duplicated().sum())}")

    print("\nRental demand statistics")
    print(
        df["cnt"]
        .describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.95])
        .round(2)
        .to_string()
    )

    print("\nHourly demand profile")

    hourly = (
        df.groupby("hr", as_index=False)
        .agg(
            observations=("cnt", "size"),
            mean_rentals=("cnt", "mean"),
            median_rentals=("cnt", "median"),
            max_rentals=("cnt", "max"),
        )
        .round(2)
    )

    print(hourly.to_string(index=False))

    busiest = hourly.loc[hourly["mean_rentals"].idxmax()]
    quietest = hourly.loc[hourly["mean_rentals"].idxmin()]

    print(f"\nBusiest average hour: {int(busiest['hr']):02d}:00")
    print(f"Quietest average hour: {int(quietest['hr']):02d}:00")

    top_3_hours = hourly.nlargest(3, "mean_rentals")    

    print("\nTop 3 hours by average rentals:")
    for _, row in top_3_hours.iterrows():
        print(f"{int(row['hr']):02d}:00 - {row['mean_rentals']:.2f} rentals on average")

    
    print("\nDemand by weather condition")

    weather = (
        df.groupby("weathersit", as_index=False)
        .agg(
            observations=("cnt", "size"),
            mean_rentals=("cnt", "mean"),
            median_rentals=("cnt", "median"),
        )
        .round(2)
    )

    percentage_of_records = (weather["observations"] / len(df) * 100).round(2)
    weather["percentage_of_records"] = percentage_of_records

    print(weather.to_string(index=False))

    print("\nDemand by working day")
    working_day = (
        df.groupby("workingday", as_index=False)
        .agg(
            observations=("cnt", "size"),
            mean_rentals=("cnt", "mean"),
            median_rentals=("cnt", "median"),
        )
        .round(2)
    )

    percentage_of_records = (working_day["observations"] / len(df) * 100).round(2)
    working_day["percentage_of_records"] = percentage_of_records

    print(working_day.to_string(index=False))

    
    print("\nDemand by year")

    yearly = (
        df.groupby("yr", as_index=False)
        .agg(
            observations=("cnt", "size"),
            total_rentals=("cnt", "sum"),
            mean_rentals=("cnt", "mean"),
            median_rentals=("cnt", "median"),
        )
        .round(2)
    )

    yearly["percentage_of_records"] = (
        yearly["observations"] / len(df) * 100
    ).round(2)
    
    print(yearly.to_string(index=False))


    print("\nDaily observation completeness")
    daily_counts = df.groupby("dteday").size()

    print(daily_counts.describe().round(2).to_string())
    print(f"Days with fewer than 24 records: {(daily_counts < 24).sum()}")
    print(f"Days with exactly 24 records: {(daily_counts == 24).sum()}")


if __name__ == "__main__":
    main()
