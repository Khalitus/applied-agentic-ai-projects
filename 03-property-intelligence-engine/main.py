from src.database import setup_database, show_database_summary
from src.sql_analytics import (
    get_property_catalog,
    get_filtered_properties,
)

def main():
    print("\nPROPERTY INTELLIGENCE ENGINE")
    print("-" * 32)
    setup_database()
    show_database_summary()

    print("\nPROPERTY CATALOG")
    filtered = get_filtered_properties(
        property_type="House",
        min_bedrooms=3,
        limit=5,
    )

    print(filtered.to_string(index=False))


if __name__ == "__main__":
    main()
