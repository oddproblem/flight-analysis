"""
AeroPulse Real-World Flight Data Cleaning Pipeline.

Supports:
  1. Kaggle US DOT 2015 Flight Dataset (data/raw/flights.csv)
  2. BTS TranStats Reporting Carrier On-Time Performance extracts (data/raw/flights_*.csv)

Processes raw flight operational records into normalized, production-grade Parquet
and writes cleaning audit summaries to data/interim/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.contracts import (
    CLEAN_FLIGHT_KEY_COLUMNS,
    DELAY_CAUSE_COLUMNS,
    SOURCE_COLUMN_RENAME_MAP,
    SOURCE_COLUMNS,
    WAREHOUSE_REQUIRED_CLEAN_COLUMNS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"

DEFAULT_OUTPUT_PARQUET = PROCESSED_DATA_DIR / "flights_clean.parquet"
COMPAT_OUTPUT_PARQUET = PROCESSED_DATA_DIR / "flights_2024_01_clean.parquet"
CLEANING_SUMMARY_FILE = INTERIM_DATA_DIR / "cleaning_summary.txt"

# ─── Reference Dictionaries ───────────────────────────────────────────────────

AIRLINE_REFERENCE = {
    "AA": ("American Airlines Inc.", 19805),
    "AS": ("Alaska Airlines Inc.", 19930),
    "B6": ("JetBlue Airways", 20409),
    "DL": ("Delta Air Lines Inc.", 19790),
    "EV": ("ExpressJet Airlines Inc.", 20366),
    "F9": ("Frontier Airlines Inc.", 20436),
    "HA": ("Hawaiian Airlines Inc.", 19690),
    "MQ": ("Envoy Air", 20398),
    "NK": ("Spirit Air Lines", 20416),
    "OO": ("SkyWest Airlines Inc.", 20304),
    "UA": ("United Air Lines Inc.", 19977),
    "US": ("US Airways Inc.", 20355),
    "VX": ("Virgin America", 21171),
    "WN": ("Southwest Airlines Co.", 19393),
}

# Major Hub & Regional Airport Directory
AIRPORT_REFERENCE = {
    "ATL": ("Atlanta, GA", "GA", "Georgia", 10397),
    "ORD": ("Chicago, IL", "IL", "Illinois", 13930),
    "DFW": ("Dallas/Fort Worth, TX", "TX", "Texas", 11298),
    "DEN": ("Denver, CO", "CO", "Colorado", 11292),
    "LAX": ("Los Angeles, CA", "CA", "California", 12892),
    "IAH": ("Houston, TX", "TX", "Texas", 12266),
    "PHX": ("Phoenix, AZ", "AZ", "Arizona", 14107),
    "SFO": ("San Francisco, CA", "CA", "California", 14771),
    "LAS": ("Las Vegas, NV", "NV", "Nevada", 12889),
    "MCO": ("Orlando, FL", "FL", "Florida", 13204),
    "CLT": ("Charlotte, NC", "NC", "North Carolina", 11057),
    "SEA": ("Seattle, WA", "WA", "Washington", 14747),
    "MSP": ("Minneapolis, MN", "MN", "Minnesota", 13487),
    "BOS": ("Boston, MA", "MA", "Massachusetts", 10721),
    "DTW": ("Detroit, MI", "MI", "Michigan", 11433),
    "PHL": ("Philadelphia, PA", "PA", "Pennsylvania", 14100),
    "LGA": ("New York, NY", "NY", "New York", 12953),
    "FLL": ("Fort Lauderdale, FL", "FL", "Florida", 11697),
    "BWI": ("Baltimore, MD", "MD", "Maryland", 10821),
    "SLC": ("Salt Lake City, UT", "UT", "Utah", 14869),
    "SAN": ("San Diego, CA", "CA", "California", 14679),
    "IAD": ("Washington, DC", "VA", "Virginia", 12264),
    "DCA": ("Washington, DC", "VA", "Virginia", 11278),
    "MDW": ("Chicago, IL", "IL", "Illinois", 13232),
    "TPA": ("Tampa, FL", "FL", "Florida", 15167),
    "PDX": ("Portland, OR", "OR", "Oregon", 14057),
    "HNL": ("Honolulu, HI", "HI", "Hawaii", 12173),
    "JFK": ("New York, NY", "NY", "New York", 12478),
    "EWR": ("Newark, NJ", "NJ", "New Jersey", 11618),
    "MIA": ("Miami, FL", "FL", "Florida", 13303),
    "AUS": ("Austin, TX", "TX", "Texas", 10423),
    "BNA": ("Nashville, TN", "TN", "Tennessee", 10693),
    "DAL": ("Dallas, TX", "TX", "Texas", 11259),
    "STL": ("St. Louis, MO", "MO", "Missouri", 15016),
    "RDU": ("Raleigh/Durham, NC", "NC", "North Carolina", 14492),
    "HOU": ("Houston, TX", "TX", "Texas", 12191),
    "SMF": ("Sacramento, CA", "CA", "California", 14893),
    "MKE": ("Milwaukee, WI", "WI", "Wisconsin", 13342),
    "OAK": ("Oakland, CA", "CA", "California", 13796),
    "MCI": ("Kansas City, MO", "MO", "Missouri", 13198),
    "SJC": ("San Jose, CA", "CA", "California", 14831),
    "SNA": ("Santa Ana, CA", "CA", "California", 14908),
    "IND": ("Indianapolis, IN", "IN", "Indiana", 12339),
    "SAT": ("San Antonio, TX", "TX", "Texas", 14683),
    "RSW": ("Fort Myers, FL", "FL", "Florida", 14635),
    "PIT": ("Pittsburgh, PA", "PA", "Pennsylvania", 14122),
    "CLE": ("Cleveland, OH", "OH", "Ohio", 11042),
    "CMH": ("Columbus, OH", "OH", "Ohio", 11193),
    "CVG": ("Cincinnati, OH", "KY", "Kentucky", 11109),
    "OGG": ("Kahului, HI", "HI", "Hawaii", 13830),
    "ANC": ("Anchorage, AK", "AK", "Alaska", 10299),
}


def extract_hour_from_hhmm(time_values: pd.Series) -> pd.Series:
    """Return the hour (0-23) from valid HHMM integers/strings."""
    numeric = pd.to_numeric(time_values, errors="coerce").astype("Int64")
    hours = numeric // 100
    minutes = numeric % 100

    valid = (
        ((hours.between(0, 23)) & (minutes.between(0, 59)))
        | (numeric == 2400)
    )

    result = pd.Series(pd.NA, index=time_values.index, dtype="Int8")
    result.loc[valid] = (hours.loc[valid] % 24).astype("Int8")
    return result


def _clean_kaggle_data(raw_df: pd.DataFrame, month_filter: int | None = 1) -> pd.DataFrame:
    """Clean Kaggle US DOT flights dataset."""
    df = raw_df.copy()

    if month_filter is not None and "MONTH" in df.columns:
        df = df[df["MONTH"] == month_filter].copy()

    # Dates
    df["flight_date"] = pd.to_datetime(
        df[["YEAR", "MONTH", "DAY"]].rename(
            columns={"YEAR": "year", "MONTH": "month", "DAY": "day"}
        )
    )

    # Airlines
    df["reporting_airline_code"] = df["AIRLINE"].astype(str).str.strip()
    df["reporting_airline_name"] = df["reporting_airline_code"].map(
        lambda c: AIRLINE_REFERENCE.get(c, (c, 99999))[0]
    ).astype("string")
    df["reporting_airline_id"] = df["reporting_airline_code"].map(
        lambda c: AIRLINE_REFERENCE.get(c, (c, 10000 + abs(hash(c)) % 90000))[1]
    ).astype("int32")

    # Flights & Aircraft
    df["flight_number"] = pd.to_numeric(df["FLIGHT_NUMBER"], errors="coerce").fillna(0).astype("int32")
    df["tail_number"] = df["TAIL_NUMBER"].astype("string").str.strip().replace({"nan": pd.NA, "": pd.NA})

    # Airports
    df["origin_airport_code"] = df["ORIGIN_AIRPORT"].astype(str).str.strip()
    df["destination_airport_code"] = df["DESTINATION_AIRPORT"].astype(str).str.strip()

    def _airport_id(code: str) -> int:
        ref = AIRPORT_REFERENCE.get(code)
        if ref:
            return ref[3]
        return 10000 + abs(hash(code)) % 90000

    def _airport_city(code: str) -> str:
        ref = AIRPORT_REFERENCE.get(code)
        return ref[0] if ref else f"{code}, US"

    def _airport_state_code(code: str) -> str:
        ref = AIRPORT_REFERENCE.get(code)
        return ref[1] if ref else "US"

    def _airport_state_name(code: str) -> str:
        ref = AIRPORT_REFERENCE.get(code)
        return ref[2] if ref else "United States"

    df["origin_airport_id"] = df["origin_airport_code"].map(_airport_id).astype("int32")
    df["destination_airport_id"] = df["destination_airport_code"].map(_airport_id).astype("int32")

    df["origin_city_name"] = df["origin_airport_code"].map(_airport_city).astype("string")
    df["origin_state_code"] = df["origin_airport_code"].map(_airport_state_code).astype("string")
    df["origin_state_name"] = df["origin_airport_code"].map(_airport_state_name).astype("string")

    df["destination_city_name"] = df["destination_airport_code"].map(_airport_city).astype("string")
    df["destination_state_code"] = df["destination_airport_code"].map(_airport_state_code).astype("string")
    df["destination_state_name"] = df["destination_airport_code"].map(_airport_state_name).astype("string")

    # Scheduled & Actual Times
    df["scheduled_departure_time"] = pd.to_numeric(df["SCHEDULED_DEPARTURE"], errors="coerce").fillna(0).astype("int16")
    df["actual_departure_time"] = pd.to_numeric(df["DEPARTURE_TIME"], errors="coerce").astype("Int16")
    df["scheduled_arrival_time"] = pd.to_numeric(df["SCHEDULED_ARRIVAL"], errors="coerce").fillna(0).astype("int16")
    df["actual_arrival_time"] = pd.to_numeric(df["ARRIVAL_TIME"], errors="coerce").astype("Int16")

    df["scheduled_departure_hour"] = extract_hour_from_hhmm(df["scheduled_departure_time"])
    df["scheduled_arrival_hour"] = extract_hour_from_hhmm(df["scheduled_arrival_time"])

    df["departure_time_block"] = df["scheduled_departure_hour"].map(
        lambda h: f"{h:02d}00-{h:02d}59" if pd.notna(h) else pd.NA
    ).astype("string")
    df["arrival_time_block"] = df["scheduled_arrival_hour"].map(
        lambda h: f"{h:02d}00-{h:02d}59" if pd.notna(h) else pd.NA
    ).astype("string")

    # Delays
    df["departure_delay_minutes_signed"] = pd.to_numeric(df["DEPARTURE_DELAY"], errors="coerce").astype("Float32")
    df["departure_delay_minutes"] = df["departure_delay_minutes_signed"].clip(lower=0).astype("Float32")
    df["departure_delayed_15"] = (df["departure_delay_minutes_signed"] >= 15).astype("Int8")

    df["arrival_delay_minutes_signed"] = pd.to_numeric(df["ARRIVAL_DELAY"], errors="coerce").astype("Float32")
    df["arrival_delay_minutes"] = df["arrival_delay_minutes_signed"].clip(lower=0).astype("Float32")
    df["arrival_delayed_15"] = (df["arrival_delay_minutes_signed"] >= 15).astype("Int8")

    # Operational Status
    df["cancelled"] = pd.to_numeric(df["CANCELLED"], errors="coerce").fillna(0).astype("int8")
    df["cancellation_code"] = df["CANCELLATION_REASON"].astype("string").str.strip().replace({"": pd.NA})
    df["diverted"] = pd.to_numeric(df["DIVERTED"], errors="coerce").fillna(0).astype("int8")

    df["flight_status"] = pd.Series("Completed", index=df.index, dtype="string")
    df.loc[df["diverted"] == 1, "flight_status"] = "Diverted"
    df.loc[df["cancelled"] == 1, "flight_status"] = "Cancelled"

    df["arrival_on_time"] = pd.Series(pd.NA, index=df.index, dtype="boolean")
    completed = (df["flight_status"] == "Completed") & df["arrival_delayed_15"].notna()
    df.loc[completed, "arrival_on_time"] = (df.loc[completed, "arrival_delayed_15"] == 0)

    # Delay Causes
    df["carrier_delay_minutes"] = pd.to_numeric(df["AIRLINE_DELAY"], errors="coerce").fillna(0).astype("Float32")
    df["weather_delay_minutes"] = pd.to_numeric(df["WEATHER_DELAY"], errors="coerce").fillna(0).astype("Float32")
    df["national_air_system_delay_minutes"] = pd.to_numeric(df["AIR_SYSTEM_DELAY"], errors="coerce").fillna(0).astype("Float32")
    df["security_delay_minutes"] = pd.to_numeric(df["SECURITY_DELAY"], errors="coerce").fillna(0).astype("Float32")
    df["late_aircraft_delay_minutes"] = pd.to_numeric(df["LATE_AIRCRAFT_DELAY"], errors="coerce").fillna(0).astype("Float32")

    df["total_reported_delay_minutes"] = (
        df["carrier_delay_minutes"]
        + df["weather_delay_minutes"]
        + df["national_air_system_delay_minutes"]
        + df["security_delay_minutes"]
        + df["late_aircraft_delay_minutes"]
    ).astype("Float32")
    df["delay_cause_reported"] = (df["total_reported_delay_minutes"] > 0).astype("boolean")

    # Elapsed Times, Distances
    df["taxi_out_minutes"] = pd.to_numeric(df["TAXI_OUT"], errors="coerce").astype("Float32")
    df["taxi_in_minutes"] = pd.to_numeric(df["TAXI_IN"], errors="coerce").astype("Float32")
    df["scheduled_elapsed_minutes"] = pd.to_numeric(df["SCHEDULED_TIME"], errors="coerce").astype("Float32")
    df["actual_elapsed_minutes"] = pd.to_numeric(df["ELAPSED_TIME"], errors="coerce").astype("Float32")
    df["air_time_minutes"] = pd.to_numeric(df["AIR_TIME"], errors="coerce").astype("Float32")
    df["distance_miles"] = pd.to_numeric(df["DISTANCE"], errors="coerce").fillna(0).astype("Float32")
    df["distance_group"] = (df["distance_miles"] // 250 + 1).clip(upper=11).astype("int8")

    # Calendar Dimensions
    df["flight_count"] = pd.Series(1, index=df.index, dtype="int8")
    df["route_code"] = (df["origin_airport_code"] + "-" + df["destination_airport_code"]).astype("string")
    df["year"] = df["flight_date"].dt.year.astype("int16")
    df["quarter"] = df["flight_date"].dt.quarter.astype("int8")
    df["month"] = df["flight_date"].dt.month.astype("int8")
    df["day_of_month"] = df["flight_date"].dt.day.astype("int8")
    df["day_of_week"] = df["flight_date"].dt.dayofweek.add(1).astype("int8")
    df["is_weekend"] = df["day_of_week"].isin([6, 7]).astype("boolean")

    exact_dup_count = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    df = df.sort_values(
        by=["flight_date", "reporting_airline_id", "flight_number", "origin_airport_id", "scheduled_departure_time"],
        kind="stable",
    ).reset_index(drop=True)

    df.attrs["exact_duplicate_count"] = exact_dup_count
    return df


INTEGER_COLUMNS = {
    "year": "Int16",
    "quarter": "Int8",
    "month": "Int8",
    "day_of_month": "Int8",
    "day_of_week": "Int8",
    "reporting_airline_id": "Int32",
    "flight_number": "Int32",
    "origin_airport_id": "Int32",
    "destination_airport_id": "Int32",
    "scheduled_departure_time": "Int16",
    "actual_departure_time": "Int16",
    "departure_delayed_15": "Int8",
    "scheduled_arrival_time": "Int16",
    "actual_arrival_time": "Int16",
    "arrival_delayed_15": "Int8",
    "cancelled": "Int8",
    "diverted": "Int8",
    "flight_count": "Int8",
    "distance_group": "Int8",
}

FLOAT_COLUMNS = [
    "departure_delay_minutes_signed",
    "departure_delay_minutes",
    "taxi_out_minutes",
    "taxi_in_minutes",
    "arrival_delay_minutes_signed",
    "arrival_delay_minutes",
    "scheduled_elapsed_minutes",
    "actual_elapsed_minutes",
    "air_time_minutes",
    "distance_miles",
    *DELAY_CAUSE_COLUMNS,
]

STRING_COLUMNS = [
    "reporting_airline_code",
    "tail_number",
    "origin_airport_code",
    "origin_city_name",
    "origin_state_code",
    "origin_state_name",
    "destination_airport_code",
    "destination_city_name",
    "destination_state_code",
    "destination_state_name",
    "departure_time_block",
    "arrival_time_block",
    "cancellation_code",
]


def _raise_if_invalid_scheduled_times(flight_data: pd.DataFrame) -> None:
    for column_name in ["scheduled_departure_time", "scheduled_arrival_time"]:
        parsed_hour = extract_hour_from_hhmm(flight_data[column_name])
        invalid_count = int(parsed_hour.isna().sum())
        if invalid_count:
            raise ValueError(
                f"Cleaning stopped because {invalid_count:,} rows have invalid {column_name} values."
            )


def _raise_if_missing_warehouse_values(flight_data: pd.DataFrame) -> None:
    missing_counts = {
        column: int(flight_data[column].isna().sum())
        for column in WAREHOUSE_REQUIRED_CLEAN_COLUMNS
        if column in flight_data.columns and flight_data[column].isna().any()
    }
    if missing_counts:
        details = ", ".join(
            f"{column}={count:,}" for column, count in sorted(missing_counts.items())
        )
        raise ValueError(
            f"Cleaning stopped because required warehouse values are missing: {details}"
        )


def _clean_bts_dataframe(flight_data: pd.DataFrame) -> pd.DataFrame:
    missing_source_columns = sorted(set(SOURCE_COLUMNS) - set(flight_data.columns))
    if missing_source_columns:
        raise ValueError(
            f"Cleaning stopped because source columns are missing: {missing_source_columns}"
        )

    flight_data = flight_data[SOURCE_COLUMNS].copy()
    flight_data = flight_data.rename(columns=SOURCE_COLUMN_RENAME_MAP)

    flight_data["flight_date"] = pd.to_datetime(
        flight_data["flight_date"],
        format="%m/%d/%Y %I:%M:%S %p",
        errors="coerce",
    )
    invalid_date_count = int(flight_data["flight_date"].isna().sum())
    if invalid_date_count:
        raise ValueError(
            f"Cleaning stopped because {invalid_date_count:,} invalid flight dates were found."
        )

    for column_name in STRING_COLUMNS:
        flight_data[column_name] = (
            flight_data[column_name].astype("string").str.strip().replace("", pd.NA)
        )

    for column_name, data_type in INTEGER_COLUMNS.items():
        flight_data[column_name] = pd.to_numeric(
            flight_data[column_name], errors="coerce"
        ).astype(data_type)

    for column_name in FLOAT_COLUMNS:
        flight_data[column_name] = pd.to_numeric(
            flight_data[column_name], errors="coerce"
        ).astype("Float32")

    missing_flight_key_count = int(
        flight_data[CLEAN_FLIGHT_KEY_COLUMNS].isna().any(axis=1).sum()
    )
    if missing_flight_key_count:
        raise ValueError(
            f"Cleaning stopped because {missing_flight_key_count:,} rows have missing flight key values."
        )

    invalid_status_flags = (
        ~flight_data["cancelled"].isin([0, 1])
        | ~flight_data["diverted"].isin([0, 1])
    )
    invalid_status_flag_count = int(invalid_status_flags.fillna(True).sum())
    if invalid_status_flag_count:
        raise ValueError(
            f"Cleaning stopped because {invalid_status_flag_count:,} rows have invalid cancelled/diverted indicators."
        )

    both_status_count = int(
        ((flight_data["cancelled"] == 1) & (flight_data["diverted"] == 1)).sum()
    )
    if both_status_count:
        raise ValueError(
            f"Cleaning stopped because {both_status_count:,} rows are marked as both cancelled and diverted."
        )

    _raise_if_invalid_scheduled_times(flight_data)

    exact_duplicate_count = int(flight_data.duplicated().sum())
    flight_data = flight_data.drop_duplicates().copy()

    flight_data["delay_cause_reported"] = (
        flight_data[DELAY_CAUSE_COLUMNS].notna().any(axis=1).astype("boolean")
    )
    flight_data[DELAY_CAUSE_COLUMNS] = (
        flight_data[DELAY_CAUSE_COLUMNS].fillna(0).astype("Float32")
    )
    flight_data["total_reported_delay_minutes"] = (
        flight_data[DELAY_CAUSE_COLUMNS].sum(axis=1).astype("Float32")
    )

    flight_data["year"] = flight_data["flight_date"].dt.year.astype("Int16")
    flight_data["quarter"] = flight_data["flight_date"].dt.quarter.astype("Int8")
    flight_data["month"] = flight_data["flight_date"].dt.month.astype("Int8")
    flight_data["day_of_month"] = flight_data["flight_date"].dt.day.astype("Int8")
    flight_data["day_of_week"] = (
        flight_data["flight_date"].dt.dayofweek.add(1).astype("Int8")
    )

    flight_data["route_code"] = (
        flight_data["origin_airport_code"]
        + "-"
        + flight_data["destination_airport_code"]
    ).astype("string")
    flight_data["is_weekend"] = flight_data["day_of_week"].isin([6, 7]).astype("boolean")
    flight_data["scheduled_departure_hour"] = extract_hour_from_hhmm(
        flight_data["scheduled_departure_time"]
    )
    flight_data["scheduled_arrival_hour"] = extract_hour_from_hhmm(
        flight_data["scheduled_arrival_time"]
    )

    flight_data["flight_status"] = pd.Series(
        "Completed", index=flight_data.index, dtype="string"
    )
    flight_data.loc[flight_data["diverted"] == 1, "flight_status"] = "Diverted"
    flight_data.loc[flight_data["cancelled"] == 1, "flight_status"] = "Cancelled"

    flight_data["arrival_on_time"] = pd.Series(
        pd.NA, index=flight_data.index, dtype="boolean"
    )
    completed_arrival_rows = (
        (flight_data["flight_status"] == "Completed")
        & flight_data["arrival_delayed_15"].notna()
    )
    flight_data.loc[completed_arrival_rows, "arrival_on_time"] = (
        flight_data.loc[completed_arrival_rows, "arrival_delayed_15"] == 0
    )

    _raise_if_missing_warehouse_values(flight_data)

    flight_data = flight_data.sort_values(
        by=[
            "flight_date",
            "reporting_airline_id",
            "flight_number",
            "origin_airport_id",
            "scheduled_departure_time",
        ],
        kind="stable",
    ).reset_index(drop=True)
    flight_data.attrs["exact_duplicate_count"] = exact_duplicate_count
    return flight_data


def clean_flight_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Standard entry point to clean flight data (handles both Kaggle and BTS formats)."""
    if "AIRLINE" in raw_df.columns:
        return _clean_kaggle_data(raw_df, month_filter=None)
    return _clean_bts_dataframe(raw_df)



def find_raw_input() -> Path:

    """Automatically find the raw flight CSV in data/raw/."""
    candidates = [
        RAW_DATA_DIR / "flights.csv",
        RAW_DATA_DIR / "flights_2024_01.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    # Any csv in data/raw/
    csvs = list(RAW_DATA_DIR.glob("*.csv"))
    if csvs:
        return csvs[0]
    raise FileNotFoundError(
        f"No raw flight CSV found in {RAW_DATA_DIR}. Please place flights.csv in {RAW_DATA_DIR}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean raw flight operational data.")
    parser.add_argument("--input", type=Path, default=None, help="Path to raw CSV file.")
    parser.add_argument("--month", type=int, default=1, help="Month to extract (1-12, default: 1). Pass 0 for all.")
    args = parser.parse_args()

    input_file = args.input or find_raw_input()
    month_filter = None if args.month == 0 else args.month

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("AEROPULSE REAL-WORLD DATA CLEANING PIPELINE")
    print("=" * 70)
    print(f"Reading raw flight data: {input_file.name}")

    raw_df = pd.read_csv(input_file, low_memory=False)
    original_rows = len(raw_df)
    original_cols = len(raw_df.columns)

    if "AIRLINE" in raw_df.columns and "ORIGIN_AIRPORT" in raw_df.columns:
        print(f"Detected Kaggle US DOT format ({original_rows:,} rows)")
        cleaned_df = _clean_kaggle_data(raw_df, month_filter=month_filter)
    else:
        print(f"Detected BTS TranStats format ({original_rows:,} rows)")
        # Legacy BTS TranStats handler
        from src.contracts import SOURCE_COLUMNS
        cleaned_df = raw_df[SOURCE_COLUMNS].rename(columns=SOURCE_COLUMN_RENAME_MAP)
        # Dates and types
        cleaned_df["flight_date"] = pd.to_datetime(cleaned_df["flight_date"])
        cleaned_df = cleaned_df.drop_duplicates()

    dup_count = cleaned_df.attrs.get("exact_duplicate_count", 0)

    # Save output to both standard and backward-compatible paths
    cleaned_df.to_parquet(DEFAULT_OUTPUT_PARQUET, index=False, engine="pyarrow", compression="snappy")
    cleaned_df.to_parquet(COMPAT_OUTPUT_PARQUET, index=False, engine="pyarrow", compression="snappy")

    output_size_mb = DEFAULT_OUTPUT_PARQUET.stat().st_size / (1024 * 1024)
    completed = int((cleaned_df["flight_status"] == "Completed").sum())
    cancelled = int((cleaned_df["flight_status"] == "Cancelled").sum())
    diverted = int((cleaned_df["flight_status"] == "Diverted").sum())
    on_time = int((cleaned_df["arrival_on_time"] == True).sum())
    delayed = int((cleaned_df["arrival_on_time"] == False).sum())
    otp_pct = (on_time / completed * 100) if completed else 0.0

    summary_text = f"""======================================================================
AEROPULSE FLIGHT DATA CLEANING SUMMARY
======================================================================
Source file: {input_file.name}
Output file: {DEFAULT_OUTPUT_PARQUET.name}

ROW AND COLUMN METRICS
----------------------------------------------------------------------
Total input rows:            {original_rows:,}
Input columns:               {original_cols}
Filtered / Cleaned rows:     {len(cleaned_df):,}
Cleaned columns:             {len(cleaned_df.columns)}
Duplicates dropped:          {dup_count:,}

OPERATIONAL FLIGHT STATUS
----------------------------------------------------------------------
Completed flights:           {completed:,} ({completed/len(cleaned_df)*100:.1f}%)
Cancelled flights:           {cancelled:,} ({cancelled/len(cleaned_df)*100:.1f}%)
Diverted flights:            {diverted:,} ({diverted/len(cleaned_df)*100:.1f}%)

ON-TIME PERFORMANCE (OTP-15)
----------------------------------------------------------------------
On-time arrivals:            {on_time:,} ({otp_pct:.2f}%)
Delayed arrivals (>=15 min): {delayed:,} ({100-otp_pct:.2f}%)

STORAGE EFFICIENCY
----------------------------------------------------------------------
Output Parquet size:         {output_size_mb:.2f} MB
Output path:                 {DEFAULT_OUTPUT_PARQUET}
======================================================================
"""
    CLEANING_SUMMARY_FILE.write_text(summary_text, encoding="utf-8")
    print(summary_text)


if __name__ == "__main__":
    main()
