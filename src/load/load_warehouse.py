import argparse
from pathlib import Path

import pandas as pd

from src.contracts import WAREHOUSE_REQUIRED_CLEAN_COLUMNS
from src.database import get_database_settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLEAN_DATA_FILE = PROJECT_ROOT / "data" / "processed" / "flights_2024_01_clean.parquet"
INTERIM_DIRECTORY = PROJECT_ROOT / "data" / "interim"
FACT_LOAD_FILE = INTERIM_DIRECTORY / "fact_flight_load.csv"
LOAD_SUMMARY_FILE = INTERIM_DIRECTORY / "warehouse_load_summary.txt"

FACT_COLUMNS = [
    "date_key",
    "airline_key",
    "origin_airport_key",
    "destination_airport_key",
    "flight_number",
    "tail_number",
    "route_code",
    "scheduled_departure_time",
    "actual_departure_time",
    "scheduled_departure_hour",
    "departure_time_block",
    "scheduled_arrival_time",
    "actual_arrival_time",
    "scheduled_arrival_hour",
    "arrival_time_block",
    "departure_delay_minutes_signed",
    "departure_delay_minutes",
    "departure_delayed_15",
    "arrival_delay_minutes_signed",
    "arrival_delay_minutes",
    "arrival_delayed_15",
    "arrival_on_time",
    "taxi_out_minutes",
    "taxi_in_minutes",
    "scheduled_elapsed_minutes",
    "actual_elapsed_minutes",
    "air_time_minutes",
    "flight_count",
    "distance_miles",
    "distance_group",
    "cancelled",
    "cancellation_code",
    "diverted",
    "flight_status",
    "delay_cause_reported",
    "carrier_delay_minutes",
    "weather_delay_minutes",
    "national_air_system_delay_minutes",
    "security_delay_minutes",
    "late_aircraft_delay_minutes",
    "total_reported_delay_minutes",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load cleaned flight data into PostgreSQL.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_CLEAN_DATA_FILE,
        help="Clean Parquet file to load.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing warehouse data before loading.",
    )
    return parser.parse_args()


def validate_load_input(flight_data: pd.DataFrame) -> None:
    missing_columns = sorted(set(WAREHOUSE_REQUIRED_CLEAN_COLUMNS) - set(flight_data.columns))
    if missing_columns:
        raise ValueError(f"Clean data is missing warehouse columns: {missing_columns}")

    missing_required = {
        column: int(flight_data[column].isna().sum())
        for column in WAREHOUSE_REQUIRED_CLEAN_COLUMNS
        if flight_data[column].isna().any()
    }
    if missing_required:
        details = ", ".join(
            f"{column}={count:,}" for column, count in sorted(missing_required.items())
        )
        raise ValueError(f"Warehouse load stopped because required values are missing: {details}")


def create_date_dimension(flight_data: pd.DataFrame) -> pd.DataFrame:
    dates = pd.DataFrame({"full_date": flight_data["flight_date"].drop_duplicates()})
    dates["date_key"] = dates["full_date"].dt.strftime("%Y%m%d").astype(int)
    dates["year_number"] = dates["full_date"].dt.year
    dates["quarter_number"] = dates["full_date"].dt.quarter
    dates["month_number"] = dates["full_date"].dt.month
    dates["month_name"] = dates["full_date"].dt.month_name()
    dates["day_of_month"] = dates["full_date"].dt.day
    dates["day_of_week_number"] = dates["full_date"].dt.dayofweek + 1
    dates["day_name"] = dates["full_date"].dt.day_name()
    dates["is_weekend"] = dates["full_date"].dt.dayofweek >= 5
    return dates[
        [
            "date_key",
            "full_date",
            "year_number",
            "quarter_number",
            "month_number",
            "month_name",
            "day_of_month",
            "day_of_week_number",
            "day_name",
            "is_weekend",
        ]
    ].sort_values("date_key")


def create_airline_dimension(flight_data: pd.DataFrame) -> pd.DataFrame:
    dimension = (
        flight_data[["reporting_airline_id", "reporting_airline_code"]]
        .drop_duplicates()
        .sort_values("reporting_airline_id")
        .reset_index(drop=True)
    )
    conflicts = dimension.groupby("reporting_airline_id").size()
    conflicts = conflicts[conflicts > 1].index.tolist()
    if conflicts:
        raise ValueError(f"Conflicting airline attributes were found for IDs: {conflicts}")
    if dimension["reporting_airline_code"].isna().any():
        raise ValueError("Airline dimension contains missing airline codes.")
    return dimension


def create_airport_dimension(flight_data: pd.DataFrame) -> pd.DataFrame:
    origin = flight_data[
        [
            "origin_airport_id",
            "origin_airport_code",
            "origin_city_name",
            "origin_state_code",
            "origin_state_name",
        ]
    ].rename(
        columns={
            "origin_airport_id": "airport_id",
            "origin_airport_code": "airport_code",
            "origin_city_name": "city_name",
            "origin_state_code": "state_code",
            "origin_state_name": "state_name",
        }
    )
    destination = flight_data[
        [
            "destination_airport_id",
            "destination_airport_code",
            "destination_city_name",
            "destination_state_code",
            "destination_state_name",
        ]
    ].rename(
        columns={
            "destination_airport_id": "airport_id",
            "destination_airport_code": "airport_code",
            "destination_city_name": "city_name",
            "destination_state_code": "state_code",
            "destination_state_name": "state_name",
        }
    )
    dimension = (
        pd.concat([origin, destination], ignore_index=True)
        .drop_duplicates()
        .sort_values("airport_id")
        .reset_index(drop=True)
    )
    if dimension[["airport_id", "airport_code"]].isna().any(axis=None):
        raise ValueError("Airport dimension contains missing IDs or airport codes.")
    conflicts = dimension.groupby("airport_id").size()
    conflicts = conflicts[conflicts > 1].index.tolist()
    if conflicts:
        raise ValueError(f"Conflicting airport attributes were found for IDs: {conflicts[:20]}")
    return dimension


def load_date_dimension(cursor, date_dimension):
    query = """
        INSERT INTO warehouse.dim_date (
            date_key, full_date, year_number, quarter_number, month_number,
            month_name, day_of_month, day_of_week_number, day_name, is_weekend
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (date_key) DO UPDATE SET
            full_date = EXCLUDED.full_date,
            year_number = EXCLUDED.year_number,
            quarter_number = EXCLUDED.quarter_number,
            month_number = EXCLUDED.month_number,
            month_name = EXCLUDED.month_name,
            day_of_month = EXCLUDED.day_of_month,
            day_of_week_number = EXCLUDED.day_of_week_number,
            day_name = EXCLUDED.day_name,
            is_weekend = EXCLUDED.is_weekend;
    """
    rows = [
        (
            int(row.date_key),
            row.full_date.date(),
            int(row.year_number),
            int(row.quarter_number),
            int(row.month_number),
            str(row.month_name),
            int(row.day_of_month),
            int(row.day_of_week_number),
            str(row.day_name),
            bool(row.is_weekend),
        )
        for row in date_dimension.itertuples(index=False)
    ]
    cursor.executemany(query, rows)


def load_airline_dimension(cursor, airline_dimension):
    query = """
        INSERT INTO warehouse.dim_airline (reporting_airline_id, reporting_airline_code)
        VALUES (%s, %s)
        ON CONFLICT (reporting_airline_id) DO UPDATE SET
            reporting_airline_code = EXCLUDED.reporting_airline_code;
    """
    cursor.executemany(
        query,
        [
            (int(row.reporting_airline_id), str(row.reporting_airline_code))
            for row in airline_dimension.itertuples(index=False)
        ],
    )


def load_airport_dimension(cursor, airport_dimension):
    query = """
        INSERT INTO warehouse.dim_airport (
            airport_id, airport_code, city_name, state_code, state_name
        ) VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (airport_id) DO UPDATE SET
            airport_code = EXCLUDED.airport_code,
            city_name = EXCLUDED.city_name,
            state_code = EXCLUDED.state_code,
            state_name = EXCLUDED.state_name;
    """
    rows = []
    for row in airport_dimension.itertuples(index=False):
        rows.append(
            (
                int(row.airport_id),
                str(row.airport_code),
                None if pd.isna(row.city_name) else str(row.city_name),
                None if pd.isna(row.state_code) else str(row.state_code),
                None if pd.isna(row.state_name) else str(row.state_name),
            )
        )
    cursor.executemany(query, rows)


def get_airline_key_mapping(cursor):
    cursor.execute("SELECT reporting_airline_id, airline_key FROM warehouse.dim_airline;")
    return dict(cursor.fetchall())


def get_airport_key_mapping(cursor):
    cursor.execute("SELECT airport_id, airport_key FROM warehouse.dim_airport;")
    return dict(cursor.fetchall())


def create_fact_load_data(flight_data, airline_key_mapping, airport_key_mapping):
    fact_data = flight_data.copy()
    fact_data["date_key"] = fact_data["flight_date"].dt.strftime("%Y%m%d").astype(int)
    fact_data["airline_key"] = fact_data["reporting_airline_id"].map(airline_key_mapping)
    fact_data["origin_airport_key"] = fact_data["origin_airport_id"].map(airport_key_mapping)
    fact_data["destination_airport_key"] = fact_data["destination_airport_id"].map(airport_key_mapping)

    if fact_data[["airline_key", "origin_airport_key", "destination_airport_key"]].isna().any(axis=None):
        raise ValueError("One or more fact rows could not be matched to warehouse dimensions.")

    for column in [
        "departure_delayed_15",
        "arrival_delayed_15",
        "arrival_on_time",
        "cancelled",
        "diverted",
        "delay_cause_reported",
    ]:
        fact_data[column] = fact_data[column].astype("boolean")

    return fact_data[FACT_COLUMNS].copy()


def copy_fact_data(cursor, fact_data):
    INTERIM_DIRECTORY.mkdir(parents=True, exist_ok=True)
    fact_data.to_csv(FACT_LOAD_FILE, index=False, na_rep="")
    columns = ", ".join(FACT_COLUMNS)
    query = f"""
        COPY warehouse.fact_flight ({columns})
        FROM STDIN WITH (FORMAT CSV, HEADER TRUE, NULL '');
    """
    try:
        with FACT_LOAD_FILE.open("rb") as input_file:
            with cursor.copy(query) as copy_process:
                while chunk := input_file.read(1024 * 1024):
                    copy_process.write(chunk)
    finally:
        FACT_LOAD_FILE.unlink(missing_ok=True)


def load_warehouse(flight_data: pd.DataFrame, replace: bool = False) -> dict:
    import psycopg

    validate_load_input(flight_data)
    settings = get_database_settings(require_env_file=False)
    source_row_count = len(flight_data)

    date_dimension = create_date_dimension(flight_data)
    airline_dimension = create_airline_dimension(flight_data)
    airport_dimension = create_airport_dimension(flight_data)

    with psycopg.connect(**settings) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM warehouse.fact_flight;")
            existing_fact_count = cursor.fetchone()[0]
            if existing_fact_count:
                if not replace:
                    raise RuntimeError(
                        f"The fact table already contains {existing_fact_count:,} rows. "
                        "Use --replace for a full reload."
                    )
                cursor.execute(
                    """
                    TRUNCATE TABLE
                        warehouse.fact_flight,
                        warehouse.dim_date,
                        warehouse.dim_airline,
                        warehouse.dim_airport
                    RESTART IDENTITY CASCADE;
                    """
                )

            load_date_dimension(cursor, date_dimension)
            load_airline_dimension(cursor, airline_dimension)
            load_airport_dimension(cursor, airport_dimension)

            fact_data = create_fact_load_data(
                flight_data,
                get_airline_key_mapping(cursor),
                get_airport_key_mapping(cursor),
            )
            copy_fact_data(cursor, fact_data)

            counts = {}
            for name, table in {
                "dim_date": "warehouse.dim_date",
                "dim_airline": "warehouse.dim_airline",
                "dim_airport": "warehouse.dim_airport",
                "fact_flight": "warehouse.fact_flight",
            }.items():
                cursor.execute(f"SELECT COUNT(*) FROM {table};")
                counts[name] = cursor.fetchone()[0]

            if counts["fact_flight"] != source_row_count:
                raise RuntimeError(
                    "Fact row count does not match source data. "
                    f"Source: {source_row_count:,}, warehouse: {counts['fact_flight']:,}."
                )

            cursor.execute("ANALYZE warehouse.fact_flight;")

    return counts


def main() -> None:
    arguments = parse_arguments()
    input_file = arguments.input.resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"Clean Parquet file was not found:\n{input_file}")

    flight_data = pd.read_parquet(input_file)
    counts = load_warehouse(flight_data, replace=arguments.replace)

    summary_lines = [
        "POSTGRESQL DATA WAREHOUSE LOAD SUMMARY",
        "=" * 70,
        f"Source file: {input_file.name}",
        f"Source rows: {len(flight_data):,}",
        "",
        "LOADED TABLES",
        "-" * 70,
        f"warehouse.dim_date: {counts['dim_date']:,}",
        f"warehouse.dim_airline: {counts['dim_airline']:,}",
        f"warehouse.dim_airport: {counts['dim_airport']:,}",
        f"warehouse.fact_flight: {counts['fact_flight']:,}",
        "",
        "STATUS",
        "-" * 70,
        "Warehouse load completed successfully.",
    ]
    INTERIM_DIRECTORY.mkdir(parents=True, exist_ok=True)
    LOAD_SUMMARY_FILE.write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
