import pandas as pd
import pytest

from src.load.load_warehouse import (
    create_airline_dimension,
    create_airport_dimension,
    create_date_dimension,
    create_fact_load_data,
    validate_load_input,
)


def clean_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "flight_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "reporting_airline_id": [1, 1],
            "reporting_airline_code": ["AA", "AA"],
            "flight_number": [10, 11],
            "tail_number": ["N1AA", "N2AA"],
            "origin_airport_id": [100, 200],
            "origin_airport_code": ["AAA", "BBB"],
            "origin_city_name": ["A", "B"],
            "origin_state_code": ["AA", "BB"],
            "origin_state_name": ["State A", "State B"],
            "destination_airport_id": [200, 100],
            "destination_airport_code": ["BBB", "AAA"],
            "destination_city_name": ["B", "A"],
            "destination_state_code": ["BB", "AA"],
            "destination_state_name": ["State B", "State A"],
            "scheduled_departure_time": [800, 900],
            "actual_departure_time": [805, 905],
            "scheduled_departure_hour": [8, 9],
            "departure_time_block": ["0800-0859", "0900-0959"],
            "scheduled_arrival_time": [1000, 1100],
            "actual_arrival_time": [1000, 1110],
            "scheduled_arrival_hour": [10, 11],
            "arrival_time_block": ["1000-1059", "1100-1159"],
            "departure_delay_minutes_signed": [5.0, 5.0],
            "departure_delay_minutes": [5.0, 5.0],
            "departure_delayed_15": [0, 0],
            "arrival_delay_minutes_signed": [0.0, 10.0],
            "arrival_delay_minutes": [0.0, 10.0],
            "arrival_delayed_15": [0, 0],
            "arrival_on_time": [True, True],
            "taxi_out_minutes": [10.0, 10.0],
            "taxi_in_minutes": [5.0, 5.0],
            "scheduled_elapsed_minutes": [120.0, 120.0],
            "actual_elapsed_minutes": [115.0, 125.0],
            "air_time_minutes": [100.0, 105.0],
            "flight_count": [1, 1],
            "distance_miles": [500.0, 500.0],
            "distance_group": [2, 2],
            "cancelled": [0, 0],
            "cancellation_code": [None, None],
            "diverted": [0, 0],
            "flight_status": ["Completed", "Completed"],
            "route_code": ["AAA-BBB", "BBB-AAA"],
            "delay_cause_reported": [False, False],
            "carrier_delay_minutes": [0.0, 0.0],
            "weather_delay_minutes": [0.0, 0.0],
            "national_air_system_delay_minutes": [0.0, 0.0],
            "security_delay_minutes": [0.0, 0.0],
            "late_aircraft_delay_minutes": [0.0, 0.0],
            "total_reported_delay_minutes": [0.0, 0.0],
        }
    )


def test_date_dimension_is_derived_from_flight_date() -> None:
    dimension = create_date_dimension(clean_rows())
    assert dimension["date_key"].tolist() == [20240101, 20240102]
    assert dimension["day_name"].tolist() == ["Monday", "Tuesday"]


def test_airline_dimension_rejects_conflicting_codes() -> None:
    data = clean_rows()
    data.loc[1, "reporting_airline_code"] = "ZZ"
    with pytest.raises(ValueError, match="Conflicting airline"):
        create_airline_dimension(data)


def test_airport_dimension_combines_origin_and_destination() -> None:
    dimension = create_airport_dimension(clean_rows())
    assert sorted(dimension["airport_id"].tolist()) == [100, 200]


def test_airport_dimension_rejects_missing_code() -> None:
    data = clean_rows()
    data.loc[0, "origin_airport_code"] = pd.NA
    with pytest.raises(ValueError, match="missing IDs or airport codes"):
        create_airport_dimension(data)


def test_validate_load_input_rejects_missing_required_values() -> None:
    data = clean_rows()
    data.loc[0, "distance_miles"] = pd.NA
    with pytest.raises(ValueError, match="required values are missing"):
        validate_load_input(data)


def test_fact_load_data_maps_surrogate_keys() -> None:
    data = clean_rows()
    fact = create_fact_load_data(data, {1: 7}, {100: 11, 200: 12})
    assert fact["airline_key"].tolist() == [7, 7]
    assert fact["origin_airport_key"].tolist() == [11, 12]
    assert fact["destination_airport_key"].tolist() == [12, 11]
