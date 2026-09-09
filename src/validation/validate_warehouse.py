from pathlib import Path

import pandas as pd
import psycopg

from src.database import get_database_settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_DATA_FILE = PROJECT_ROOT / "data" / "processed" / "flights_2024_01_clean.parquet"
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "interim"
VALIDATION_RESULTS_FILE = OUTPUT_DIRECTORY / "warehouse_validation_results.csv"
VALIDATION_SUMMARY_FILE = OUTPUT_DIRECTORY / "warehouse_validation_summary.txt"


def add_result(results, rule_name, expected_value, actual_value, description):
    results.append(
        {
            "rule_name": rule_name,
            "status": "PASS" if expected_value == actual_value else "FAIL",
            "expected_value": int(expected_value),
            "actual_value": int(actual_value),
            "description": description,
        }
    )


def fetch_single_value(cursor, query: str) -> int:
    cursor.execute(query)
    return int(cursor.fetchone()[0])


def validate_warehouse(clean_data: pd.DataFrame) -> pd.DataFrame:
    expected = {
        "date_dimension_count": int(clean_data["flight_date"].nunique()),
        "airline_dimension_count": int(clean_data["reporting_airline_id"].nunique()),
        "airport_dimension_count": len(
            set(clean_data["origin_airport_id"].dropna().astype(int))
            | set(clean_data["destination_airport_id"].dropna().astype(int))
        ),
        "fact_flight_count": len(clean_data),
        "completed_flight_count": int((clean_data["flight_status"] == "Completed").sum()),
        "cancelled_flight_count": int((clean_data["flight_status"] == "Cancelled").sum()),
        "diverted_flight_count": int((clean_data["flight_status"] == "Diverted").sum()),
        "on_time_arrival_count": int((clean_data["arrival_on_time"] == True).sum()),
        "delayed_arrival_count": int((clean_data["arrival_on_time"] == False).sum()),
        "unknown_arrival_count": int(clean_data["arrival_on_time"].isna().sum()),
    }

    settings = get_database_settings(require_env_file=False)
    results = []

    with psycopg.connect(**settings) as connection:
        with connection.cursor() as cursor:
            actual = {
                "date_dimension_count": fetch_single_value(cursor, "SELECT COUNT(*) FROM warehouse.dim_date;"),
                "airline_dimension_count": fetch_single_value(cursor, "SELECT COUNT(*) FROM warehouse.dim_airline;"),
                "airport_dimension_count": fetch_single_value(cursor, "SELECT COUNT(*) FROM warehouse.dim_airport;"),
                "fact_flight_count": fetch_single_value(cursor, "SELECT COUNT(*) FROM warehouse.fact_flight;"),
                "completed_flight_count": fetch_single_value(cursor, "SELECT COUNT(*) FROM warehouse.fact_flight WHERE flight_status = 'Completed';"),
                "cancelled_flight_count": fetch_single_value(cursor, "SELECT COUNT(*) FROM warehouse.fact_flight WHERE flight_status = 'Cancelled';"),
                "diverted_flight_count": fetch_single_value(cursor, "SELECT COUNT(*) FROM warehouse.fact_flight WHERE flight_status = 'Diverted';"),
                "on_time_arrival_count": fetch_single_value(cursor, "SELECT COUNT(*) FROM warehouse.fact_flight WHERE arrival_on_time = TRUE;"),
                "delayed_arrival_count": fetch_single_value(cursor, "SELECT COUNT(*) FROM warehouse.fact_flight WHERE arrival_on_time = FALSE;"),
                "unknown_arrival_count": fetch_single_value(cursor, "SELECT COUNT(*) FROM warehouse.fact_flight WHERE arrival_on_time IS NULL;"),
            }

            orphan_queries = {
                "no_orphan_date_keys": """
                    SELECT COUNT(*) FROM warehouse.fact_flight f
                    LEFT JOIN warehouse.dim_date d ON f.date_key = d.date_key
                    WHERE d.date_key IS NULL;
                """,
                "no_orphan_airline_keys": """
                    SELECT COUNT(*) FROM warehouse.fact_flight f
                    LEFT JOIN warehouse.dim_airline d ON f.airline_key = d.airline_key
                    WHERE d.airline_key IS NULL;
                """,
                "no_orphan_origin_keys": """
                    SELECT COUNT(*) FROM warehouse.fact_flight f
                    LEFT JOIN warehouse.dim_airport d ON f.origin_airport_key = d.airport_key
                    WHERE d.airport_key IS NULL;
                """,
                "no_orphan_destination_keys": """
                    SELECT COUNT(*) FROM warehouse.fact_flight f
                    LEFT JOIN warehouse.dim_airport d ON f.destination_airport_key = d.airport_key
                    WHERE d.airport_key IS NULL;
                """,
            }
            orphan_counts = {
                name: fetch_single_value(cursor, query)
                for name, query in orphan_queries.items()
            }

            duplicate_fact_keys = fetch_single_value(
                cursor,
                """
                SELECT COUNT(*) FROM (
                    SELECT date_key, airline_key, flight_number,
                           origin_airport_key, destination_airport_key,
                           scheduled_departure_time
                    FROM warehouse.fact_flight
                    GROUP BY 1,2,3,4,5,6
                    HAVING COUNT(*) > 1
                ) d;
                """,
            )

            cursor.execute(
                "SELECT COALESCE(ROUND(SUM(total_reported_delay_minutes)::numeric, 2), 0) FROM warehouse.fact_flight;"
            )
            warehouse_delay_total = float(cursor.fetchone()[0])

    descriptions = {
        "date_dimension_count": "Date dimension count must match unique source dates.",
        "airline_dimension_count": "Airline dimension count must match unique source airlines.",
        "airport_dimension_count": "Airport dimension count must match unique source airports.",
        "fact_flight_count": "Fact row count must match the clean source dataset.",
        "completed_flight_count": "Completed flight count must match clean data.",
        "cancelled_flight_count": "Cancelled flight count must match clean data.",
        "diverted_flight_count": "Diverted flight count must match clean data.",
        "on_time_arrival_count": "On-time arrival count must match clean data.",
        "delayed_arrival_count": "Delayed arrival count must match clean data.",
        "unknown_arrival_count": "Unknown arrival count must match clean data.",
    }
    for name, expected_value in expected.items():
        add_result(results, name, expected_value, actual[name], descriptions[name])

    for name, count in orphan_counts.items():
        add_result(results, name, 0, count, "Warehouse foreign keys must resolve to their dimensions.")

    add_result(
        results,
        "fact_natural_key_is_unique",
        0,
        duplicate_fact_keys,
        "The scheduled-segment natural key must remain unique in the fact table.",
    )

    expected_delay_total = round(float(clean_data["total_reported_delay_minutes"].sum()), 2)
    results.append(
        {
            "rule_name": "reported_delay_minutes_reconcile",
            "status": "PASS" if abs(expected_delay_total - warehouse_delay_total) <= 0.01 else "FAIL",
            "expected_value": expected_delay_total,
            "actual_value": warehouse_delay_total,
            "description": "Reported delay minutes must reconcile between Parquet and PostgreSQL.",
        }
    )

    return pd.DataFrame(results)


def main() -> None:
    if not CLEAN_DATA_FILE.exists():
        raise FileNotFoundError(f"Clean data file was not found:\n{CLEAN_DATA_FILE}")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    clean_data = pd.read_parquet(CLEAN_DATA_FILE)
    results = validate_warehouse(clean_data)
    results.to_csv(VALIDATION_RESULTS_FILE, index=False)
    failed = int((results["status"] == "FAIL").sum())
    overall_status = "PASS" if failed == 0 else "FAIL"

    summary_lines = [
        "POSTGRESQL WAREHOUSE VALIDATION SUMMARY",
        "=" * 70,
        f"Validation rules checked: {len(results)}",
        f"Failed rules: {failed}",
        f"Overall status: {overall_status}",
        "",
        results.to_string(index=False),
    ]
    VALIDATION_SUMMARY_FILE.write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))

    if failed:
        raise RuntimeError(f"Warehouse validation failed with {failed} failed rule(s).")


if __name__ == "__main__":
    main()
