import pandas as pd
import pytest

from src.contracts import SOURCE_COLUMN_RENAME_MAP
from src.transform.clean_flight_data import clean_flight_dataframe, extract_hour_from_hhmm

RAW_COLUMNS = list(SOURCE_COLUMN_RENAME_MAP)


def make_raw_flight_row(**overrides) -> dict:
    row = dict(
        YEAR=2024,
        QUARTER=1,
        MONTH=1,
        DAY_OF_MONTH=1,
        DAY_OF_WEEK=1,
        FL_DATE="01/01/2024 12:00:00 AM",
        OP_UNIQUE_CARRIER="AA",
        OP_CARRIER_AIRLINE_ID=19805,
        TAIL_NUM="N123AA",
        OP_CARRIER_FL_NUM=100,
        ORIGIN_AIRPORT_ID=12478,
        ORIGIN="JFK",
        ORIGIN_CITY_NAME="New York, NY",
        ORIGIN_STATE_ABR="NY",
        ORIGIN_STATE_NM="New York",
        DEST_AIRPORT_ID=12892,
        DEST="LAX",
        DEST_CITY_NAME="Los Angeles, CA",
        DEST_STATE_ABR="CA",
        DEST_STATE_NM="California",
        CRS_DEP_TIME=800,
        DEP_TIME=805,
        DEP_DELAY=5,
        DEP_DELAY_NEW=5,
        DEP_DEL15=0,
        DEP_TIME_BLK="0800-0859",
        TAXI_OUT=15,
        TAXI_IN=8,
        CRS_ARR_TIME=1100,
        ARR_TIME=1058,
        ARR_DELAY=-2,
        ARR_DELAY_NEW=0,
        ARR_DEL15=0,
        ARR_TIME_BLK="1100-1159",
        CANCELLED=0,
        CANCELLATION_CODE=None,
        DIVERTED=0,
        CRS_ELAPSED_TIME=300,
        ACTUAL_ELAPSED_TIME=293,
        AIR_TIME=270,
        FLIGHTS=1,
        DISTANCE=2475,
        DISTANCE_GROUP=10,
        CARRIER_DELAY=None,
        WEATHER_DELAY=None,
        NAS_DELAY=None,
        SECURITY_DELAY=None,
        LATE_AIRCRAFT_DELAY=None,
    )
    row.update(overrides)
    return row


def make_raw_flight_data(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)[RAW_COLUMNS]


def test_extract_hour_from_hhmm_accepts_valid_values_and_2400() -> None:
    values = pd.Series([5, 59, 100, 930, 2359, 2400])
    result = extract_hour_from_hhmm(values)
    assert result.tolist() == [0, 0, 1, 9, 23, 0]


def test_extract_hour_from_hhmm_rejects_invalid_values() -> None:
    values = pd.Series([2360, 2401, 2460, 2500, -1, None, "bad"])
    result = extract_hour_from_hhmm(values)
    assert result.isna().all()


def test_route_code_and_hour_extraction() -> None:
    cleaned = clean_flight_dataframe(make_raw_flight_data([make_raw_flight_row()]))
    row = cleaned.iloc[0]
    assert row["route_code"] == "JFK-LAX"
    assert row["scheduled_departure_hour"] == 8
    assert row["scheduled_arrival_hour"] == 11


def test_calendar_fields_are_derived_from_flight_date() -> None:
    raw = make_raw_flight_data(
        [make_raw_flight_row(DAY_OF_WEEK=7, QUARTER=4, MONTH=12, DAY_OF_MONTH=31)]
    )
    cleaned = clean_flight_dataframe(raw)
    row = cleaned.iloc[0]
    assert row["quarter"] == 1
    assert row["month"] == 1
    assert row["day_of_month"] == 1
    assert row["day_of_week"] == 1
    assert row["is_weekend"] == False


def test_flight_status_transitions() -> None:
    raw = make_raw_flight_data(
        [
            make_raw_flight_row(OP_CARRIER_FL_NUM=1),
            make_raw_flight_row(
                OP_CARRIER_FL_NUM=2,
                CANCELLED=1,
                ARR_DEL15=None,
                DEP_DEL15=None,
            ),
            make_raw_flight_row(OP_CARRIER_FL_NUM=3, DIVERTED=1, ARR_DEL15=None),
        ]
    )
    by_flight = clean_flight_dataframe(raw).set_index("flight_number")
    assert by_flight.loc[1, "flight_status"] == "Completed"
    assert by_flight.loc[2, "flight_status"] == "Cancelled"
    assert by_flight.loc[3, "flight_status"] == "Diverted"


def test_cancelled_and_diverted_conflict_is_rejected() -> None:
    raw = make_raw_flight_data(
        [make_raw_flight_row(CANCELLED=1, DIVERTED=1, ARR_DEL15=None, DEP_DEL15=None)]
    )
    with pytest.raises(ValueError, match="both cancelled and diverted"):
        clean_flight_dataframe(raw)


def test_arrival_on_time_only_applies_to_completed_flights() -> None:
    raw = make_raw_flight_data(
        [
            make_raw_flight_row(OP_CARRIER_FL_NUM=1, ARR_DEL15=0),
            make_raw_flight_row(OP_CARRIER_FL_NUM=2, ARR_DEL15=1),
            make_raw_flight_row(
                OP_CARRIER_FL_NUM=3,
                CANCELLED=1,
                ARR_DEL15=None,
                DEP_DEL15=None,
            ),
        ]
    )
    by_flight = clean_flight_dataframe(raw).set_index("flight_number")
    assert by_flight.loc[1, "arrival_on_time"] == True
    assert by_flight.loc[2, "arrival_on_time"] == False
    assert pd.isna(by_flight.loc[3, "arrival_on_time"])


def test_delay_cause_reporting_and_total() -> None:
    raw = make_raw_flight_data(
        [
            make_raw_flight_row(OP_CARRIER_FL_NUM=1),
            make_raw_flight_row(
                OP_CARRIER_FL_NUM=2,
                CARRIER_DELAY=10,
                WEATHER_DELAY=0,
                NAS_DELAY=5,
            ),
        ]
    )
    by_flight = clean_flight_dataframe(raw).set_index("flight_number")
    assert by_flight.loc[1, "delay_cause_reported"] == False
    assert by_flight.loc[1, "total_reported_delay_minutes"] == 0.0
    assert by_flight.loc[2, "delay_cause_reported"] == True
    assert by_flight.loc[2, "total_reported_delay_minutes"] == 15.0


def test_exact_duplicate_rows_are_removed_and_counted() -> None:
    first = make_raw_flight_row(OP_CARRIER_FL_NUM=1)
    second = make_raw_flight_row(OP_CARRIER_FL_NUM=2, TAIL_NUM="N456AA")
    cleaned = clean_flight_dataframe(
        make_raw_flight_data([first, dict(first), second])
    )
    assert cleaned.attrs["exact_duplicate_count"] == 1
    assert len(cleaned) == 2


def test_missing_source_columns_raise_value_error() -> None:
    raw = make_raw_flight_data([make_raw_flight_row()]).drop(columns=["TAIL_NUM"])
    with pytest.raises(ValueError, match="source columns are missing"):
        clean_flight_dataframe(raw)


def test_invalid_flight_date_raises_value_error() -> None:
    raw = make_raw_flight_data([make_raw_flight_row(FL_DATE="not-a-date")])
    with pytest.raises(ValueError, match="invalid flight dates"):
        clean_flight_dataframe(raw)


def test_missing_flight_key_value_raises_value_error() -> None:
    raw = make_raw_flight_data([make_raw_flight_row(ORIGIN_AIRPORT_ID=None)])
    with pytest.raises(ValueError, match="missing flight key values"):
        clean_flight_dataframe(raw)


def test_invalid_scheduled_time_raises_value_error() -> None:
    raw = make_raw_flight_data([make_raw_flight_row(CRS_DEP_TIME=2460)])
    with pytest.raises(ValueError, match="invalid scheduled_departure_time"):
        clean_flight_dataframe(raw)


def test_missing_required_warehouse_value_raises_value_error() -> None:
    raw = make_raw_flight_data([make_raw_flight_row(ORIGIN=None)])
    with pytest.raises(ValueError, match="required warehouse values are missing"):
        clean_flight_dataframe(raw)


def test_blank_strings_become_missing() -> None:
    raw = make_raw_flight_data([make_raw_flight_row(CANCELLATION_CODE="   ")])
    cleaned = clean_flight_dataframe(raw)
    assert pd.isna(cleaned.iloc[0]["cancellation_code"])
