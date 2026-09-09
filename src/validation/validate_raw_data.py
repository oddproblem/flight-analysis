from pathlib import Path

import pandas as pd

from src.contracts import SOURCE_COLUMNS, SOURCE_FLIGHT_KEY_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_FILE = PROJECT_ROOT / "data" / "raw" / "flights_2024_01.csv"
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "interim"
VALIDATION_RESULTS_FILE = OUTPUT_DIRECTORY / "flights_2024_01_validation_results.csv"
VALIDATION_SUMMARY_FILE = OUTPUT_DIRECTORY / "flights_2024_01_validation_summary.txt"

NON_NEGATIVE_COLUMNS = [
    "DEP_DELAY_NEW",
    "ARR_DELAY_NEW",
    "TAXI_OUT",
    "TAXI_IN",
    "CRS_ELAPSED_TIME",
    "ACTUAL_ELAPSED_TIME",
    "AIR_TIME",
    "DISTANCE",
    "CARRIER_DELAY",
    "WEATHER_DELAY",
    "NAS_DELAY",
    "SECURITY_DELAY",
    "LATE_AIRCRAFT_DELAY",
]


def add_validation_result(results, rule_name, severity, failed_row_count, description):
    if failed_row_count == 0:
        status = "PASS"
    elif severity == "warning":
        status = "WARNING"
    else:
        status = "FAIL"
    results.append(
        {
            "rule_name": rule_name,
            "severity": severity,
            "status": status,
            "failed_row_count": int(failed_row_count),
            "description": description,
        }
    )


def validate_raw_dataframe(flight_data: pd.DataFrame) -> pd.DataFrame:
    results = []

    missing_columns = sorted(set(SOURCE_COLUMNS) - set(flight_data.columns))
    add_validation_result(
        results,
        "required_columns_present",
        "critical",
        len(missing_columns),
        f"All selected BTS source columns must exist. Missing: {missing_columns}",
    )
    if missing_columns:
        return pd.DataFrame(results)

    flight_dates = pd.to_datetime(
        flight_data["FL_DATE"],
        format="%m/%d/%Y %I:%M:%S %p",
        errors="coerce",
    )
    add_validation_result(
        results,
        "valid_flight_dates",
        "critical",
        int(flight_dates.isna().sum()),
        "FL_DATE must contain a valid calendar date.",
    )

    comparisons = {
        "year_matches_flight_date": flight_dates.dt.year != flight_data["YEAR"],
        "quarter_matches_flight_date": flight_dates.dt.quarter != flight_data["QUARTER"],
        "month_matches_flight_date": flight_dates.dt.month != flight_data["MONTH"],
        "day_matches_flight_date": flight_dates.dt.day != flight_data["DAY_OF_MONTH"],
        "weekday_matches_flight_date": (flight_dates.dt.dayofweek + 1)
        != flight_data["DAY_OF_WEEK"],
    }
    for rule_name, mismatch in comparisons.items():
        add_validation_result(
            results,
            rule_name,
            "critical",
            int((flight_dates.notna() & mismatch).sum()),
            "Source calendar fields must agree with FL_DATE.",
        )

    add_validation_result(
        results,
        "selected_period_is_january_2024",
        "critical",
        int(((flight_data["YEAR"] != 2024) | (flight_data["MONTH"] != 1)).sum()),
        "The pilot dataset must contain only January 2024.",
    )

    add_validation_result(
        results,
        "flight_key_not_missing",
        "critical",
        int(flight_data[SOURCE_FLIGHT_KEY_COLUMNS].isna().any(axis=1).sum()),
        "Flight-key fields must not be missing.",
    )
    add_validation_result(
        results,
        "no_exact_duplicate_rows",
        "critical",
        int(flight_data.duplicated().sum()),
        "The raw extract must not contain exact duplicate rows.",
    )
    add_validation_result(
        results,
        "unique_flight_keys",
        "critical",
        int(
            flight_data.duplicated(
                subset=SOURCE_FLIGHT_KEY_COLUMNS, keep=False
            ).sum()
        ),
        "Each scheduled segment must have a unique flight key.",
    )

    for column_name, rule_name in [
        ("CANCELLED", "valid_cancelled_indicator"),
        ("DIVERTED", "valid_diverted_indicator"),
    ]:
        invalid = flight_data[column_name].isna() | ~flight_data[column_name].isin([0, 1])
        add_validation_result(
            results,
            rule_name,
            "critical",
            int(invalid.sum()),
            f"{column_name} must be present and contain only 0 or 1.",
        )

    add_validation_result(
        results,
        "flight_not_cancelled_and_diverted",
        "critical",
        int(((flight_data["CANCELLED"] == 1) & (flight_data["DIVERTED"] == 1)).sum()),
        "A row must not be marked as both cancelled and diverted.",
    )

    for column_name in [
        "OP_UNIQUE_CARRIER",
        "ORIGIN",
        "DEST",
        "CRS_DEP_TIME",
        "CRS_ARR_TIME",
        "FLIGHTS",
        "DISTANCE",
    ]:
        add_validation_result(
            results,
            f"{column_name.lower()}_not_missing",
            "critical",
            int(flight_data[column_name].isna().sum()),
            f"{column_name} is required by the cleaned dataset or warehouse.",
        )

    negative_mask = pd.DataFrame(
        {column: flight_data[column] < 0 for column in NON_NEGATIVE_COLUMNS}
    )
    add_validation_result(
        results,
        "non_negative_measurements",
        "critical",
        int(negative_mask.any(axis=1).sum()),
        "Unsigned delays, taxi times, elapsed times, distance and delay causes must not be negative.",
    )

    for value_column, flag_column, rule_name in [
        ("DEP_DELAY_NEW", "DEP_DEL15", "departure_delay_indicator_consistent"),
        ("ARR_DELAY_NEW", "ARR_DEL15", "arrival_delay_indicator_consistent"),
    ]:
        comparable = flight_data[value_column].notna() & flight_data[flag_column].notna()
        expected = (flight_data.loc[comparable, value_column] >= 15).astype(int)
        actual = flight_data.loc[comparable, flag_column].astype(int)
        add_validation_result(
            results,
            rule_name,
            "critical",
            int((expected != actual).sum()),
            f"{flag_column} must agree with {value_column}.",
        )

    add_validation_result(
        results,
        "cancelled_flights_have_reason",
        "warning",
        int(
            ((flight_data["CANCELLED"] == 1) & flight_data["CANCELLATION_CODE"].isna()).sum()
        ),
        "Cancelled flights normally include a cancellation reason code.",
    )
    add_validation_result(
        results,
        "active_flights_have_no_cancellation_code",
        "warning",
        int(
            ((flight_data["CANCELLED"] == 0) & flight_data["CANCELLATION_CODE"].notna()).sum()
        ),
        "Non-cancelled flights normally have no cancellation reason code.",
    )
    add_validation_result(
        results,
        "origin_differs_from_destination",
        "warning",
        int((flight_data["ORIGIN_AIRPORT_ID"] == flight_data["DEST_AIRPORT_ID"]).sum()),
        "Origin and destination airport IDs normally differ.",
    )

    return pd.DataFrame(results)


def main() -> None:
    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(f"Raw data file was not found:\n{RAW_DATA_FILE}")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    flight_data = pd.read_csv(RAW_DATA_FILE, low_memory=False)
    results = validate_raw_dataframe(flight_data)
    results.to_csv(VALIDATION_RESULTS_FILE, index=False)

    critical_failures = int(
        ((results["severity"] == "critical") & (results["status"] == "FAIL")).sum()
    )
    warnings = int((results["status"] == "WARNING").sum())
    overall_status = (
        "FAIL" if critical_failures else "PASS WITH WARNINGS" if warnings else "PASS"
    )

    summary_lines = [
        "BTS RAW DATA QUALITY VALIDATION SUMMARY",
        "=" * 70,
        f"Source file: {RAW_DATA_FILE.name}",
        f"Rows checked: {len(flight_data):,}",
        f"Validation rules checked: {len(results)}",
        f"Critical rule failures: {critical_failures}",
        f"Warnings: {warnings}",
        f"Overall status: {overall_status}",
        "",
        results.to_string(index=False),
    ]
    VALIDATION_SUMMARY_FILE.write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))

    if critical_failures:
        raise RuntimeError(
            f"Raw data validation failed with {critical_failures} critical rule failure(s)."
        )


if __name__ == "__main__":
    main()
