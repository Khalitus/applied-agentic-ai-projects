from src.database import setup_database, show_database_summary
from src.sql_analytics import (
    get_property_catalog,
    get_filtered_properties,
    get_recent_sales,
)

def main():
    print("\nPROPERTY INTELLIGENCE ENGINE")
    print("-" * 32)
    setup_database()
    show_database_summary()

    print("\nPROPERTY CATALOG")
    recent_sales = get_recent_sales(limit=5)

    print(recent_sales.to_string(index=False))    


if __name__ == "__main__":
    main()
