from src.database import setup_database, show_database_summary
from src.sql_analytics import (
    get_property_catalog,
    get_filtered_properties,
    get_recent_sales,
    get_neighborhood_summary,
    get_latest_property_sales,
    get_neighborhood_price_analytics,
    get_sale_history_growth
)
from src.ml_model import (
    load_modeling_dataset,
    prepare_features_target,
    train_baseline_model
)

def main():
    print("\nPROPERTY INTELLIGENCE ENGINE")
    print("-" * 32)
    setup_database()
    show_database_summary()

    print("\nModeling dataset")
    print("-" * 32)

    data = load_modeling_dataset()

    print(data.head().to_string(index=False))
    print(f"\nRows: {len(data)}")
    print(f"Columns: {len(data.columns)}")

    X, y = prepare_features_target(data)

    print("\nFeatures")
    print("-" * 32)
    print(X.head().to_string(index=False))

    print("\nTarget")
    print("-" * 32)
    print(y.head().to_string(index=False))

    print(X.shape)
    print(y.shape)

    print(X.isna().sum())
    print(y.isna().sum())

    print(X.dtypes)

    model, X, y = train_baseline_model()

    print("\nBaseline model")
    print("-" * 32)
    print(model)
  

if __name__ == "__main__":
    main()
