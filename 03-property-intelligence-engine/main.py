from src.database import setup_database, show_database_summary
from src.sql_analytics import (
    get_property_catalog,
    get_filtered_properties,
    get_recent_sales,
    get_neighborhood_summary,
    get_latest_property_sales,
    get_neighborhood_price_analytics,
    get_sale_history_growth,
)
from src.ml_model import (
    load_modeling_dataset,
    prepare_features_target,
    train_baseline_model,
    make_sample_predictions,
    evaluate_model,
    split_modeling_data,
    evaluate_baseline_split,
    compare_tree_sizes,
    train_tuned_tree,
    train_random_forest,
    better_model,
    forest_improvement
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

    # model, X, y = train_baseline_model()

    # sample, predictions = make_sample_predictions(
    #     model,
    #     X,
    #     count=5,
    # )

    # print("\nSample properties")
    # print("-" * 32)
    # print(sample.to_string(index=False))

    # print("\nPredicted prices")
    # print("-" * 32)
    # print(predictions)

    # mae = evaluate_training_error(
    #     model,
    #     X,
    #     y,
    # )

    # print("\nTraining MAE")
    # print("-" * 32)
    # print(f"{mae:,.2f}")

    train_X, val_X, train_y, val_y = split_modeling_data(X, y)

    # print("\nTraining and validation split")
    # print("-" * 32)
    # print(f"Training features: {train_X.shape}")
    # print(f"Validation features: {val_X.shape}")
    # print(f"Training target: {train_y.shape}")
    # print(f"Validation target: {val_y.shape}")

    # train_mae, val_mae = evaluate_baseline_split(
    #     train_X,
    #     val_X,
    #     train_y,
    #     val_y,
    # )

    # print("\nUnrestricted decision tree")
    # print("-" * 32)
    # print(f"Training MAE: {train_mae:,.2f}")
    # print(f"Validation MAE: {val_mae:,.2f}")

    results = compare_tree_sizes(
        train_X,
        val_X,
        train_y,
        val_y,
    )

    print("\nDecision tree tuning")
    print("-" * 32)

    for leaves, mae in results.items():
        print(
            f"Max leaf nodes: {leaves:<3} "
            f"Validation MAE: {mae:,.2f}"
        )

    best_leaf_nodes = min(results, key = results.get)

    best_model = train_tuned_tree(best_leaf_nodes, train_X, train_y)
    best_mae = evaluate_model(best_model, val_X, val_y)

    print(f"\nBest max leaf nodes: {best_leaf_nodes}")
    print(f"Best MAE: {best_mae:,.2f}")

    forest_model = train_random_forest(train_X, train_y)
    forest_mae = evaluate_model(forest_model, val_X, val_y)

    print(f"\nForest MAE: {forest_mae:,.2f}")

    good_model = better_model(best_mae, forest_mae)

    print(f"Better model: {good_model}")

    improvement = forest_improvement(best_mae, forest_mae)
    print(f"Random Forest improvement: {improvement:,.2f}%")

if __name__ == "__main__":
    main()
