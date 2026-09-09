import os

import pandas as pd
import pytest

psycopg = pytest.importorskip("psycopg")

from src.database import get_database_settings
from src.load.load_warehouse import load_warehouse

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_POSTGRES_TESTS") != "1",
    reason="PostgreSQL integration tests are enabled in CI.",
)


def integration_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "flight_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "reporting_airline_id": [1, 1],
            "reporting_airline_code": ["AA", "AA"],
            "flight_number": [10, 11],
            "tail_number": ["N1AA", "N2AA"],
            "origin_airport_id": [100, 200],
            "origin_airport_code": ["AAA", "BBB"],
            "origin_city_name": ["Alpha, AA", "Beta, BB"],
            "origin_state_code": ["AA", "BB"],
            "origin_state_name": ["Alpha", "Beta"],
            "destination_airport_id": [200, 100],
            "destination_airport_code": ["BBB", "AAA"],
            "destination_city_name": ["Beta, BB", "Alpha, AA"],
            "destination_state_code": ["BB", "AA"],
            "destination_state_name": ["Beta", "Alpha"],
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
            "departure_delayed_15": [False, False],
            "arrival_delay_minutes_signed": [0.0, 10.0],
            "arrival_delay_minutes": [0.0, 10.0],
            "arrival_delayed_15": [False, False],
            "arrival_on_time": [True, True],
            "taxi_out_minutes": [10.0, 10.0],
            "taxi_in_minutes": [5.0, 5.0],
            "scheduled_elapsed_minutes": [120.0, 120.0],
            "actual_elapsed_minutes": [115.0, 125.0],
            "air_time_minutes": [100.0, 105.0],
            "flight_count": [1, 1],
            "distance_miles": [500.0, 500.0],
            "distance_group": [2, 2],
            "cancelled": [False, False],
            "cancellation_code": [None, None],
            "diverted": [False, False],
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


def test_postgres_load_and_analytics_views() -> None:
    counts = load_warehouse(integration_rows(), replace=True)
    assert counts == {
        "dim_date": 2,
        "dim_airline": 1,
        "dim_airport": 2,
        "fact_flight": 2,
    }

    settings = get_database_settings(require_env_file=False)
    with psycopg.connect(**settings) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM analytics.vw_flight_detail;")
            assert cursor.fetchone()[0] == 2
            cursor.execute("SELECT total_flights FROM analytics.vw_overview_metrics;")
            assert cursor.fetchone()[0] == 2
