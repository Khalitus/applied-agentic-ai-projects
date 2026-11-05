"""Verify the SQLite bike-sharing database."""

from src.db import connect


def main() -> None:
    with connect() as connection:
        total = connection.execute(
            "SELECT COUNT(*) FROM bike_rentals"
        ).fetchone()[0]

        unique_dates = connection.execute(
            "SELECT COUNT(DISTINCT dteday) FROM bike_rentals"
        ).fetchone()[0]

        print(f"Total records: {total:,}")
        print(f"Unique dates: {unique_dates}")

        min_rentals = connection.execute(
            "SELECT MIN(cnt) FROM bike_rentals"
        ).fetchone()[0]
        max_rentals = connection.execute(
            "SELECT MAX(cnt) FROM bike_rentals"
        ).fetchone()[0]

        print(f"Minimum rentals: {min_rentals}")
        print(f"Maximum rentals: {max_rentals}")

        invalid_relationships = connection.execute(
            "SELECT * FROM bike_rentals WHERE casual + registered != cnt"
        ).fetchall()
        print(f"Invalid rental-count relationships: {len(invalid_relationships)}")

        duplicate_combinations = connection.execute(
            "SELECT dteday, hr, COUNT(*) FROM bike_rentals GROUP BY dteday, hr HAVING COUNT(*) > 1"
        ).fetchall()
        print(f"Duplicate date-hour combinations: {len(duplicate_combinations)}")

        primary_key_info = connection.execute(
            "PRAGMA table_info(bike_rentals)"
        ).fetchall()
        print("Primary Key Information:")
        for column in primary_key_info:
            if column["pk"]:
                print(f"  {column['name']}: PRIMARY KEY")

        index_info = connection.execute(
            "SELECT * FROM sqlite_master WHERE type = 'index'"
        ).fetchall()
        print("Index Information:")
        for index in index_info:
            print(f"  {index['name']}: {index['sql']}")

    # Assertions moved inside main() to access local scope variables
    assert total == 17379
    assert unique_dates == 731
    assert min_rentals == 1
    assert max_rentals == 977
    assert len(invalid_relationships) == 0
    assert len(duplicate_combinations) == 0

    assert any(
        column["name"] == "instant" and column["pk"] == 1
        for column in primary_key_info
    )

    index_names = {index["name"] for index in index_info}

    assert {
        "idx_bike_date_hour",
        "idx_bike_hr",
    }.issubset(index_names)

    print("All database assertions passed successfully!")


if __name__ == "__main__":
    main()