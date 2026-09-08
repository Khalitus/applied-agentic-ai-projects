from src.database import setup_database, show_database_summary
from src.sql_analytics import (
    get_property_catalog,
    get_filtered_properties,
    get_recent_sales,
    get_neighborhood_summary,
)

def main():
    print("\nPROPERTY INTELLIGENCE ENGINE")
    print("-" * 32)
    setup_database()
    show_database_summary()

    print("\nPROPERTY CATALOG")
    summary = get_neighborhood_summary()

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
