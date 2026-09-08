from src.database import setup_database, show_database_summary
from src.sql_analytics import get_property_catalog

def main():
    print("\nPROPERTY INTELLIGENCE ENGINE")
    print("-" * 32)
    setup_database()
    show_database_summary()

    print("\nPROPERTY CATALOG")
    catalog = get_property_catalog(limit=5)
    print(catalog.to_string(index=False))

    
if __name__ == "__main__":
    main()
