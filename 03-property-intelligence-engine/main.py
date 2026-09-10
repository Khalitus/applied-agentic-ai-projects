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

def main():
    print("\nPROPERTY INTELLIGENCE ENGINE")
    print("-" * 32)
    setup_database()
    show_database_summary()

    print("\nPROPERTY CATALOG")
    history = get_sale_history_growth(limit=30)

    print(history.to_string(index=False))

if __name__ == "__main__":
    main()
