import psycopg

from src.database import get_database_settings


def main() -> None:
    settings = get_database_settings()

    print("=" * 60)
    print("POSTGRESQL CONNECTION TEST")
    print("=" * 60)

    try:
        with psycopg.connect(**settings) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        current_database(),
                        current_user,
                        current_setting('port'),
                        version();
                    """
                )
                database_name, user_name, database_port, database_version = cursor.fetchone()
    except psycopg.Error as database_error:
        raise RuntimeError(
            "Could not connect to PostgreSQL. Check the .env values and database service."
        ) from database_error

    print("Connection status: SUCCESS")
    print(f"Database: {database_name}")
    print(f"User: {user_name}")
    print(f"Port: {database_port}")
    print(f"Version: {database_version}")
    print("=" * 60)


if __name__ == "__main__":
    main()
