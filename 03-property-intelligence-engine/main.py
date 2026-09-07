from src.database import setup_database, show_database_summary

def main():
    print("\nPROPERTY INTELLIGENCE ENGINE")
    print("-" * 32)
    setup_database()
    show_database_summary()

if __name__ == "__main__":
    main()
