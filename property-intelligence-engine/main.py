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
            pass
        elif choice == "2":
            pass
        elif choice == "3":
            pass
        elif choice == "4":
            pass
        elif choice == "5":
            print("\nGoodbye.")
            break
        else:
            print("\nInvalid option. Choose 1-5.")


if __name__ == "__main__":
    main()