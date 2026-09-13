from src.database import setup_database
from src.ml_model import (
    better_model,
    compare_tree_sizes,
    evaluate_model,
    forest_improvement,
    load_modeling_dataset,
    prepare_features_target,
    split_modeling_data,
    train_random_forest,
    train_tuned_tree,
)
from src.sql_analytics import (
    get_neighborhood_price_analytics,
    get_neighborhood_summary,
    get_property_catalog,
    get_sale_history_growth,
)
from src.vector_store import (
    build_property_index,
    get_index_summary,
    semantic_search,
)

def show_property_analytics():
    catalog = get_property_catalog(limit=5)
    neighborhoods = get_neighborhood_summary()
    prices = get_neighborhood_price_analytics(limit=10)

    print("\nProperty catalog")
    print(catalog.to_string(index=False))

    print("\nNeighborhood summary")
    print(neighborhoods.to_string(index=False))

    print("\nNeighborhood price analytics")
    print(prices.to_string(index=False))

def show_price_history():
    property_id = input("\nProperty ID: ").strip().upper()

    history = get_sale_history_growth(limit=1000)

    property_history = history[property_id]

    if property_history.empty:
        print("\nNo sale history found for that property.")
        return

    print()
    print(property_history.to_string(index=False))

def evaluate_valuation_models():
    data = load_modeling_dataset()
    X, y = prepare_features_target(data)

    train_X, val_X, train_y, val_y = split_modeling_data(X, y)

    results = compare_tree_sizes(
        train_X,
        val_X,
        train_y,
        val_y,
    )

    best_leaf_nodes = results.max()

    tree_model = train_tuned_tree(
        best_leaf_nodes,
        train_X,
        train_y,
    )

    tree_mae = evaluate_model(
        tree_model,
        val_X,
        val_y,
    )

    forest_model = train_random_forest(
        train_X,
        train_y
    )

    forest_mae = evaluate_model(
        forest_model,
        val_X,
        val_y
    )

    winner = better_model(tree_mae, forest_mae)
    improvement = forest_improvement(tree_mae, forest_mae)

    print("\nProperty valuation models")
    print(f"Decision Tree leaf nodes: {best_leaf_nodes}")
    print(f"Decision Tree MAE: {tree_mae:,.2f}")
    print(f"Random Forest MAE: {forest_mae:,.2f}")
    print(f"Better model: {winner}")
    print(f"Random Forest improvement: {improvement:,.2f}%")

def show_menu():
    print("\nProperty Intelligence Engine")
    print("1. View property analytics")
    print("2. Analyze price history")
    print("3. Evaluate property valuation models")
    print("4. Search properties semantically")
    print("5. Exit")

def main():
    setup_database()

    while True:
        show_menu()
        choice = input("\nChoose an option: ").strip()

        if choice == "1":
            show_property_analytics()
        elif choice == "2":
            show_price_history()
        elif choice == "3":
            evaluate_valuation_models()
        elif choice == "4":
            pass
        elif choice == "5":
            print("\nGoodbye.")
            break
        else:
            print("\nInvalid option. Choose 1-5.")


if __name__ == "__main__":
    main()