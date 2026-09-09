from pathlib import Path

import pandas as pd

from src.contracts import SOURCE_COLUMNS, SOURCE_FLIGHT_KEY_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_FILE = PROJECT_ROOT / "data" / "raw" / "flights_2024_01.csv"
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "interim"
PROFILE_REPORT_FILE = OUTPUT_DIRECTORY / "flights_2024_01_profile.txt"
MISSING_VALUES_FILE = OUTPUT_DIRECTORY / "flights_2024_01_missing_values.csv"
DATA_TYPES_FILE = OUTPUT_DIRECTORY / "flights_2024_01_data_types.csv"


def main() -> None:
    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(f"Raw data file was not found:\n{RAW_DATA_FILE}")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    flight_data = pd.read_csv(RAW_DATA_FILE, low_memory=False)
    actual_columns = flight_data.columns.tolist()
    missing_columns = sorted(set(SOURCE_COLUMNS) - set(actual_columns))
    unexpected_columns = sorted(set(actual_columns) - set(SOURCE_COLUMNS))

    missing_summary = pd.DataFrame(
        {
            "column_name": actual_columns,
            "missing_count": flight_data.isna().sum().values,
        }
    )
    missing_summary["missing_percentage"] = (
        missing_summary["missing_count"] / max(len(flight_data), 1) * 100
    ).round(2)
    missing_summary.sort_values(
        by=["missing_percentage", "column_name"],
        ascending=[False, True],
    ).to_csv(MISSING_VALUES_FILE, index=False)

    pd.DataFrame(
        {
            "column_name": actual_columns,
            "data_type": [str(dtype) for dtype in flight_data.dtypes],
        }
    ).to_csv(DATA_TYPES_FILE, index=False)

    report_lines = [
        "BTS RAW FLIGHT DATA PROFILE",
        "=" * 60,
        f"Source file: {RAW_DATA_FILE.name}",
        f"Number of rows: {len(flight_data):,}",
        f"Number of columns: {len(actual_columns)}",
        "",
        "SCHEMA",
        "-" * 60,
        f"Missing selected columns: {missing_columns}",
        f"Unexpected columns: {unexpected_columns}",
    ]

    if not missing_columns:
        dates = pd.to_datetime(
            flight_data["FL_DATE"],
            format="%m/%d/%Y %I:%M:%S %p",
            errors="coerce",
        )
        report_lines.extend(
            [
                "",
                "DATE COVERAGE",
                "-" * 60,
                f"Minimum flight date: {dates.min().date() if dates.notna().any() else 'N/A'}",
                f"Maximum flight date: {dates.max().date() if dates.notna().any() else 'N/A'}",
                f"Invalid flight dates: {int(dates.isna().sum()):,}",
                "",
                "DUPLICATES",
                "-" * 60,
                f"Exact duplicate rows: {int(flight_data.duplicated().sum()):,}",
                f"Rows with duplicate flight keys: {int(flight_data.duplicated(subset=SOURCE_FLIGHT_KEY_COLUMNS, keep=False).sum()):,}",
                "",
                "FLIGHT SUMMARY",
                "-" * 60,
                f"Unique airlines: {flight_data['OP_CARRIER_AIRLINE_ID'].nunique():,}",
                f"Unique origin airports: {flight_data['ORIGIN_AIRPORT_ID'].nunique():,}",
                f"Unique destination airports: {flight_data['DEST_AIRPORT_ID'].nunique():,}",
                f"Cancelled flights: {int((flight_data['CANCELLED'] == 1).sum()):,}",
                f"Diverted flights: {int((flight_data['DIVERTED'] == 1).sum()):,}",
                f"Flights arriving at least 15 minutes late: {int((flight_data['ARR_DEL15'] == 1).sum()):,}",
            ]
        )

    PROFILE_REPORT_FILE.write_text("\n".join(report_lines), encoding="utf-8")
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
