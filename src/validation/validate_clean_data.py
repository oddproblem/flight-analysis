from pathlib import Path

import pandas as pd

from src.contracts import (
    CLEAN_FLIGHT_KEY_COLUMNS,
    DELAY_CAUSE_COLUMNS,
    WAREHOUSE_REQUIRED_CLEAN_COLUMNS,
)
from src.transform.clean_flight_data import extract_hour_from_hhmm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_DATA_FILE = PROJECT_ROOT / "data" / "processed" / "flights_2024_01_clean.parquet"
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "interim"
VALIDATION_RESULTS_FILE = OUTPUT_DIRECTORY / "flights_2024_01_clean_validation_results.csv"
VALIDATION_SUMMARY_FILE = OUTPUT_DIRECTORY / "flights_2024_01_clean_validation_summary.txt"

NON_NEGATIVE_COLUMNS = [
    "departure_delay_minutes",
    "arrival_delay_minutes",
    "taxi_out_minutes",
    "taxi_in_minutes",
    "scheduled_elapsed_minutes",
    "actual_elapsed_minutes",
    "air_time_minutes",
    "distance_miles",
    *DELAY_CAUSE_COLUMNS,
    "total_reported_delay_minutes",
]


def add_result(results, rule_name, failed_row_count, description):
    results.append(
        {
            "rule_name": rule_name,
            "status": "PASS" if failed_row_count == 0 else "FAIL",
            "failed_row_count": int(failed_row_count),
            "description": description,
        }
    )


def validate_clean_dataframe(flight_data: pd.DataFrame) -> pd.DataFrame:
    results = []
    required_columns = sorted(
        set(WAREHOUSE_REQUIRED_CLEAN_COLUMNS)
        | {
            "year",
            "quarter",
            "month",
            "day_of_month",
            "day_of_week",
            "arrival_delayed_15",
            "arrival_on_time",
            "is_weekend",
            "scheduled_departure_hour",
            "scheduled_arrival_hour",
        }
    )
    missing_columns = sorted(set(required_columns) - set(flight_data.columns))
    add_result(
        results,
        "required_columns_present",
        len(missing_columns),
        f"Required cleaned columns must exist. Missing: {missing_columns}",
    )
    if missing_columns:
        return pd.DataFrame(results)

    add_result(results, "dataset_is_not_empty", int(len(flight_data) == 0), "Dataset must contain rows.")
    add_result(
        results,
        "flight_key_not_missing",
        int(flight_data[CLEAN_FLIGHT_KEY_COLUMNS].isna().any(axis=1).sum()),
        "Flight-key fields must be present.",
    )
    add_result(
        results,
        "flight_key_is_unique",
        int(flight_data.duplicated(subset=CLEAN_FLIGHT_KEY_COLUMNS, keep=False).sum()),
        "Flight keys must be unique.",
    )

    add_result(
        results,
        "warehouse_required_values_not_missing",
        int(flight_data[WAREHOUSE_REQUIRED_CLEAN_COLUMNS].isna().any(axis=1).sum()),
        "Columns mapped to required warehouse fields must not be missing.",
    )

    add_result(
        results,
        "date_period_is_january_2024",
        int(
            ((flight_data["flight_date"].dt.year != 2024) | (flight_data["flight_date"].dt.month != 1)).sum()
        ),
        "The pilot dataset must contain only January 2024.",
    )

    calendar_mismatch = (
        (flight_data["year"] != flight_data["flight_date"].dt.year)
        | (flight_data["quarter"] != flight_data["flight_date"].dt.quarter)
        | (flight_data["month"] != flight_data["flight_date"].dt.month)
        | (flight_data["day_of_month"] != flight_data["flight_date"].dt.day)
        | (flight_data["day_of_week"] != flight_data["flight_date"].dt.dayofweek + 1)
    )
    add_result(
        results,
        "calendar_fields_match_flight_date",
        int(calendar_mismatch.sum()),
        "Derived calendar fields must match flight_date.",
    )

    add_result(
        results,
        "flight_status_is_valid",
        int((~flight_data["flight_status"].isin(["Completed", "Cancelled", "Diverted"])).sum()),
        "flight_status must be Completed, Cancelled or Diverted.",
    )
    add_result(
        results,
        "flight_not_cancelled_and_diverted",
        int(((flight_data["cancelled"] == 1) & (flight_data["diverted"] == 1)).sum()),
        "A flight cannot be both cancelled and diverted.",
    )

    expected_status = pd.Series("Completed", index=flight_data.index, dtype="string")
    expected_status.loc[flight_data["diverted"] == 1] = "Diverted"
    expected_status.loc[flight_data["cancelled"] == 1] = "Cancelled"
    add_result(
        results,
        "flight_status_matches_indicators",
        int((flight_data["flight_status"].astype("string") != expected_status).sum()),
        "flight_status must agree with cancelled and diverted flags.",
    )

    expected_route = (
        flight_data["origin_airport_code"].astype("string")
        + "-"
        + flight_data["destination_airport_code"].astype("string")
    )
    add_result(
        results,
        "route_code_is_consistent",
        int((flight_data["route_code"].astype("string") != expected_route).sum()),
        "route_code must match origin and destination airport codes.",
    )

    expected_weekend = flight_data["flight_date"].dt.dayofweek >= 5
    add_result(
        results,
        "weekend_indicator_is_consistent",
        int((flight_data["is_weekend"].astype(bool) != expected_weekend).sum()),
        "is_weekend must be derived from flight_date.",
    )

    expected_dep_hour = extract_hour_from_hhmm(flight_data["scheduled_departure_time"])
    expected_arr_hour = extract_hour_from_hhmm(flight_data["scheduled_arrival_time"])
    add_result(
        results,
        "scheduled_departure_hour_is_valid",
        int((flight_data["scheduled_departure_hour"].astype("Int8") != expected_dep_hour).fillna(True).sum()),
        "scheduled_departure_hour must match a valid HHMM departure time.",
    )
    add_result(
        results,
        "scheduled_arrival_hour_is_valid",
        int((flight_data["scheduled_arrival_hour"].astype("Int8") != expected_arr_hour).fillna(True).sum()),
        "scheduled_arrival_hour must match a valid HHMM arrival time.",
    )

    add_result(
        results,
        "measurements_are_non_negative",
        int((flight_data[NON_NEGATIVE_COLUMNS] < 0).any(axis=1).sum()),
        "Unsigned measurements must not be negative.",
    )

    expected_total_delay = flight_data[DELAY_CAUSE_COLUMNS].sum(axis=1)
    add_result(
        results,
        "total_reported_delay_is_consistent",
        int(((flight_data["total_reported_delay_minutes"] - expected_total_delay).abs() > 0.01).sum()),
        "Total reported delay must equal the sum of the five cause columns.",
    )

    expected_arrival_on_time = pd.Series(pd.NA, index=flight_data.index, dtype="boolean")
    eligible = (flight_data["flight_status"] == "Completed") & flight_data["arrival_delayed_15"].notna()
    expected_arrival_on_time.loc[eligible] = flight_data.loc[eligible, "arrival_delayed_15"] == 0
    actual = flight_data["arrival_on_time"].astype("string").fillna("<NA>")
    expected = expected_arrival_on_time.astype("string").fillna("<NA>")
    add_result(
        results,
        "arrival_on_time_is_consistent",
        int((actual != expected).sum()),
        "arrival_on_time must agree with flight status and ARR_DEL15.",
    )

    return pd.DataFrame(results)


def main() -> None:
    if not CLEAN_DATA_FILE.exists():
        raise FileNotFoundError(f"Clean data file was not found:\n{CLEAN_DATA_FILE}")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    flight_data = pd.read_parquet(CLEAN_DATA_FILE)
    results = validate_clean_dataframe(flight_data)
    results.to_csv(VALIDATION_RESULTS_FILE, index=False)
    failed = int((results["status"] == "FAIL").sum())
    overall_status = "PASS" if failed == 0 else "FAIL"

    summary_lines = [
        "BTS CLEAN FLIGHT DATA VALIDATION SUMMARY",
        "=" * 70,
        f"Source file: {CLEAN_DATA_FILE.name}",
        f"Rows checked: {len(flight_data):,}",
        f"Columns checked: {len(flight_data.columns)}",
        f"Validation rules checked: {len(results)}",
        f"Failed rules: {failed}",
        f"Overall status: {overall_status}",
        "",
        results.to_string(index=False),
    ]
    VALIDATION_SUMMARY_FILE.write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))

    if failed:
        raise RuntimeError(f"Clean data validation failed with {failed} failed rule(s).")


if __name__ == "__main__":
    main()
