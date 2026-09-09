-- Create a separate schema for reporting and dashboard views
CREATE SCHEMA IF NOT EXISTS analytics;


-- ============================================================
-- 1. Detailed analytical flight view
-- ============================================================

CREATE OR REPLACE VIEW analytics.vw_flight_detail AS
SELECT
    flight_fact.flight_key,

    flight_fact.date_key,
    date_dimension.full_date AS flight_date,
    date_dimension.year_number,
    date_dimension.quarter_number,
    date_dimension.month_number,
    date_dimension.month_name,
    date_dimension.day_of_month,
    date_dimension.day_of_week_number,
    date_dimension.day_name,
    date_dimension.is_weekend,

    flight_fact.airline_key,
    airline_dimension.reporting_airline_id,
    airline_dimension.reporting_airline_code,

    flight_fact.origin_airport_key,
    origin_airport.airport_id AS origin_airport_id,
    origin_airport.airport_code AS origin_airport_code,
    origin_airport.city_name AS origin_city_name,
    origin_airport.state_code AS origin_state_code,
    origin_airport.state_name AS origin_state_name,

    flight_fact.destination_airport_key,
    destination_airport.airport_id AS destination_airport_id,
    destination_airport.airport_code AS destination_airport_code,
    destination_airport.city_name AS destination_city_name,
    destination_airport.state_code AS destination_state_code,
    destination_airport.state_name AS destination_state_name,

    flight_fact.flight_number,
    flight_fact.tail_number,
    flight_fact.route_code,

    flight_fact.scheduled_departure_time,
    flight_fact.actual_departure_time,
    flight_fact.scheduled_departure_hour,
    flight_fact.departure_time_block,

    flight_fact.scheduled_arrival_time,
    flight_fact.actual_arrival_time,
    flight_fact.scheduled_arrival_hour,
    flight_fact.arrival_time_block,

    flight_fact.departure_delay_minutes_signed,
    flight_fact.departure_delay_minutes,
    flight_fact.departure_delayed_15,

    flight_fact.arrival_delay_minutes_signed,
    flight_fact.arrival_delay_minutes,
    flight_fact.arrival_delayed_15,
    flight_fact.arrival_on_time,

    flight_fact.taxi_out_minutes,
    flight_fact.taxi_in_minutes,
    flight_fact.scheduled_elapsed_minutes,
    flight_fact.actual_elapsed_minutes,
    flight_fact.air_time_minutes,

    flight_fact.flight_count,
    flight_fact.distance_miles,
    flight_fact.distance_group,

    flight_fact.cancelled,
    flight_fact.cancellation_code,
    flight_fact.diverted,
    flight_fact.flight_status,

    flight_fact.delay_cause_reported,
    flight_fact.carrier_delay_minutes,
    flight_fact.weather_delay_minutes,
    flight_fact.national_air_system_delay_minutes,
    flight_fact.security_delay_minutes,
    flight_fact.late_aircraft_delay_minutes,
    flight_fact.total_reported_delay_minutes

FROM warehouse.fact_flight AS flight_fact

JOIN warehouse.dim_date AS date_dimension
    ON flight_fact.date_key =
       date_dimension.date_key

JOIN warehouse.dim_airline AS airline_dimension
    ON flight_fact.airline_key =
       airline_dimension.airline_key

JOIN warehouse.dim_airport AS origin_airport
    ON flight_fact.origin_airport_key =
       origin_airport.airport_key

JOIN warehouse.dim_airport AS destination_airport
    ON flight_fact.destination_airport_key =
       destination_airport.airport_key;


-- ============================================================
-- 2. Overall dashboard metrics
-- ============================================================

CREATE OR REPLACE VIEW analytics.vw_overview_metrics AS
SELECT
    COUNT(*) AS total_flights,

    COUNT(*) FILTER (
        WHERE flight_status = 'Completed'
    ) AS completed_flights,

    COUNT(*) FILTER (
        WHERE flight_status = 'Cancelled'
    ) AS cancelled_flights,

    COUNT(*) FILTER (
        WHERE flight_status = 'Diverted'
    ) AS diverted_flights,

    COUNT(*) FILTER (
        WHERE arrival_on_time IS NOT NULL
    ) AS eligible_arrivals,

    COUNT(*) FILTER (
        WHERE arrival_on_time = TRUE
    ) AS on_time_arrivals,

    COUNT(*) FILTER (
        WHERE arrival_on_time = FALSE
    ) AS delayed_arrivals,

    ROUND(
        (
            100.0
            * COUNT(*) FILTER (
                WHERE arrival_on_time = TRUE
            )
            / NULLIF(
                COUNT(*) FILTER (
                    WHERE arrival_on_time IS NOT NULL
                ),
                0
            )
        )::NUMERIC,
        2
    ) AS on_time_arrival_rate_percentage,

    ROUND(
        (
            100.0
            * COUNT(*) FILTER (
                WHERE flight_status = 'Cancelled'
            )
            / NULLIF(COUNT(*), 0)
        )::NUMERIC,
        2
    ) AS cancellation_rate_percentage,

    ROUND(
        AVG(arrival_delay_minutes_signed)::NUMERIC,
        2
    ) AS average_signed_arrival_delay_minutes,

    ROUND(
        AVG(arrival_delay_minutes)::NUMERIC,
        2
    ) AS average_arrival_delay_minutes,

    ROUND(
        (
            PERCENTILE_CONT(0.5)
            WITHIN GROUP (
                ORDER BY arrival_delay_minutes_signed
            )
            FILTER (
                WHERE arrival_delay_minutes_signed IS NOT NULL
            )
        )::NUMERIC,
        2
    ) AS median_signed_arrival_delay_minutes,

    ROUND(
        SUM(total_reported_delay_minutes)::NUMERIC,
        2
    ) AS total_reported_delay_minutes

FROM analytics.vw_flight_detail;


-- ============================================================
-- 3. Daily performance
-- ============================================================

CREATE OR REPLACE VIEW analytics.vw_daily_performance AS
SELECT
    flight_date,
    day_name,
    is_weekend,

    COUNT(*) AS total_flights,

    COUNT(*) FILTER (
        WHERE flight_status = 'Completed'
    ) AS completed_flights,

    COUNT(*) FILTER (
        WHERE flight_status = 'Cancelled'
    ) AS cancelled_flights,

    COUNT(*) FILTER (
        WHERE flight_status = 'Diverted'
    ) AS diverted_flights,

    COUNT(*) FILTER (
        WHERE arrival_on_time = TRUE
    ) AS on_time_arrivals,

    COUNT(*) FILTER (
        WHERE arrival_on_time = FALSE
    ) AS delayed_arrivals,

    ROUND(
        (
            100.0
            * COUNT(*) FILTER (
                WHERE arrival_on_time = TRUE
            )
            / NULLIF(
                COUNT(*) FILTER (
                    WHERE arrival_on_time IS NOT NULL
                ),
                0
            )
        )::NUMERIC,
        2
    ) AS on_time_arrival_rate_percentage,

    ROUND(
        (
            100.0
            * COUNT(*) FILTER (
                WHERE flight_status = 'Cancelled'
            )
            / NULLIF(COUNT(*), 0)
        )::NUMERIC,
        2
    ) AS cancellation_rate_percentage,

    ROUND(
        AVG(arrival_delay_minutes_signed)::NUMERIC,
        2
    ) AS average_signed_arrival_delay_minutes,

    ROUND(
        SUM(total_reported_delay_minutes)::NUMERIC,
        2
    ) AS total_reported_delay_minutes

FROM analytics.vw_flight_detail

GROUP BY
    flight_date,
    day_name,
    is_weekend;


-- ============================================================
-- 4. Airline performance
-- ============================================================

CREATE OR REPLACE VIEW analytics.vw_airline_performance AS
SELECT
    reporting_airline_id,
    reporting_airline_code,

    COUNT(*) AS total_flights,

    COUNT(*) FILTER (
        WHERE flight_status = 'Completed'
    ) AS completed_flights,

    COUNT(*) FILTER (
        WHERE flight_status = 'Cancelled'
    ) AS cancelled_flights,

    COUNT(*) FILTER (
        WHERE flight_status = 'Diverted'
    ) AS diverted_flights,

    COUNT(*) FILTER (
        WHERE arrival_on_time IS NOT NULL
    ) AS eligible_arrivals,

    COUNT(*) FILTER (
        WHERE arrival_on_time = TRUE
    ) AS on_time_arrivals,

    COUNT(*) FILTER (
        WHERE arrival_on_time = FALSE
    ) AS delayed_arrivals,

    ROUND(
        (
            100.0
            * COUNT(*) FILTER (
                WHERE arrival_on_time = TRUE
            )
            / NULLIF(
                COUNT(*) FILTER (
                    WHERE arrival_on_time IS NOT NULL
                ),
                0
            )
        )::NUMERIC,
        2
    ) AS on_time_arrival_rate_percentage,

    ROUND(
        (
            100.0
            * COUNT(*) FILTER (
                WHERE flight_status = 'Cancelled'
            )
            / NULLIF(COUNT(*), 0)
        )::NUMERIC,
        2
    ) AS cancellation_rate_percentage,

    ROUND(
        AVG(arrival_delay_minutes_signed)::NUMERIC,
        2
    ) AS average_signed_arrival_delay_minutes,

    ROUND(
        AVG(departure_delay_minutes_signed)::NUMERIC,
        2
    ) AS average_signed_departure_delay_minutes,

    ROUND(
        (
            PERCENTILE_CONT(0.5)
            WITHIN GROUP (
                ORDER BY arrival_delay_minutes_signed
            )
            FILTER (
                WHERE arrival_delay_minutes_signed IS NOT NULL
            )
        )::NUMERIC,
        2
    ) AS median_signed_arrival_delay_minutes,

    ROUND(
        AVG(taxi_out_minutes)::NUMERIC,
        2
    ) AS average_taxi_out_minutes,

    ROUND(
        SUM(total_reported_delay_minutes)::NUMERIC,
        2
    ) AS total_reported_delay_minutes

FROM analytics.vw_flight_detail

GROUP BY
    reporting_airline_id,
    reporting_airline_code;


-- ============================================================
-- 5. Origin airport performance
-- ============================================================

CREATE OR REPLACE VIEW analytics.vw_origin_airport_performance AS
SELECT
    origin_airport_id,
    origin_airport_code,
    origin_city_name,
    origin_state_code,
    origin_state_name,

    COUNT(*) AS departing_flights,

    COUNT(*) FILTER (
        WHERE flight_status = 'Completed'
    ) AS completed_flights,

    COUNT(*) FILTER (
        WHERE flight_status = 'Cancelled'
    ) AS cancelled_flights,

    COUNT(*) FILTER (
        WHERE flight_status = 'Diverted'
    ) AS diverted_flights,

    COUNT(*) FILTER (
        WHERE arrival_on_time = TRUE
    ) AS on_time_arrivals,

    COUNT(*) FILTER (
        WHERE arrival_on_time = FALSE
    ) AS delayed_arrivals,

    ROUND(
        (
            100.0
            * COUNT(*) FILTER (
                WHERE arrival_on_time = TRUE
            )
            / NULLIF(
                COUNT(*) FILTER (
                    WHERE arrival_on_time IS NOT NULL
                ),
                0
            )
        )::NUMERIC,
        2
    ) AS on_time_arrival_rate_percentage,

    ROUND(
        (
            100.0
            * COUNT(*) FILTER (
                WHERE flight_status = 'Cancelled'
            )
            / NULLIF(COUNT(*), 0)
        )::NUMERIC,
        2
    ) AS cancellation_rate_percentage,

    ROUND(
        AVG(departure_delay_minutes_signed)::NUMERIC,
        2
    ) AS average_signed_departure_delay_minutes,

    ROUND(
        AVG(arrival_delay_minutes_signed)::NUMERIC,
        2
    ) AS average_signed_arrival_delay_minutes,

    ROUND(
        AVG(taxi_out_minutes)::NUMERIC,
        2
    ) AS average_taxi_out_minutes

FROM analytics.vw_flight_detail

GROUP BY
    origin_airport_id,
    origin_airport_code,
    origin_city_name,
    origin_state_code,
    origin_state_name;


-- ============================================================
-- 6. Route performance
-- ============================================================

CREATE OR REPLACE VIEW analytics.vw_route_performance AS
SELECT
    route_code,

    origin_airport_id,
    origin_airport_code,
    origin_city_name,
    origin_state_code,

    destination_airport_id,
    destination_airport_code,
    destination_city_name,
    destination_state_code,

    COUNT(*) AS total_flights,

    COUNT(*) FILTER (
        WHERE flight_status = 'Completed'
    ) AS completed_flights,

    COUNT(*) FILTER (
        WHERE flight_status = 'Cancelled'
    ) AS cancelled_flights,

    COUNT(*) FILTER (
        WHERE flight_status = 'Diverted'
    ) AS diverted_flights,

    COUNT(*) FILTER (
        WHERE arrival_on_time = TRUE
    ) AS on_time_arrivals,

    COUNT(*) FILTER (
        WHERE arrival_on_time = FALSE
    ) AS delayed_arrivals,

    ROUND(
        (
            100.0
            * COUNT(*) FILTER (
                WHERE arrival_on_time = TRUE
            )
            / NULLIF(
                COUNT(*) FILTER (
                    WHERE arrival_on_time IS NOT NULL
                ),
                0
            )
        )::NUMERIC,
        2
    ) AS on_time_arrival_rate_percentage,

    ROUND(
        (
            100.0
            * COUNT(*) FILTER (
                WHERE flight_status = 'Cancelled'
            )
            / NULLIF(COUNT(*), 0)
        )::NUMERIC,
        2
    ) AS cancellation_rate_percentage,

    ROUND(
        AVG(arrival_delay_minutes_signed)::NUMERIC,
        2
    ) AS average_signed_arrival_delay_minutes,

    ROUND(
        AVG(distance_miles)::NUMERIC,
        2
    ) AS average_distance_miles

FROM analytics.vw_flight_detail

GROUP BY
    route_code,
    origin_airport_id,
    origin_airport_code,
    origin_city_name,
    origin_state_code,
    destination_airport_id,
    destination_airport_code,
    destination_city_name,
    destination_state_code;


-- ============================================================
-- 7. Scheduled departure-hour performance
-- ============================================================

CREATE OR REPLACE VIEW analytics.vw_departure_hour_performance AS
SELECT
    scheduled_departure_hour,

    COUNT(*) AS total_flights,

    COUNT(*) FILTER (
        WHERE flight_status = 'Cancelled'
    ) AS cancelled_flights,

    COUNT(*) FILTER (
        WHERE arrival_on_time = TRUE
    ) AS on_time_arrivals,

    COUNT(*) FILTER (
        WHERE arrival_on_time = FALSE
    ) AS delayed_arrivals,

    ROUND(
        (
            100.0
            * COUNT(*) FILTER (
                WHERE arrival_on_time = TRUE
            )
            / NULLIF(
                COUNT(*) FILTER (
                    WHERE arrival_on_time IS NOT NULL
                ),
                0
            )
        )::NUMERIC,
        2
    ) AS on_time_arrival_rate_percentage,

    ROUND(
        (
            100.0
            * COUNT(*) FILTER (
                WHERE flight_status = 'Cancelled'
            )
            / NULLIF(COUNT(*), 0)
        )::NUMERIC,
        2
    ) AS cancellation_rate_percentage,

    ROUND(
        AVG(arrival_delay_minutes_signed)::NUMERIC,
        2
    ) AS average_signed_arrival_delay_minutes

FROM analytics.vw_flight_detail

GROUP BY
    scheduled_departure_hour;


-- ============================================================
-- 8. Delay causes by airline, in long format for Power BI
-- ============================================================

CREATE OR REPLACE VIEW analytics.vw_delay_cause_by_airline AS
WITH airline_delay_totals AS (
    SELECT
        reporting_airline_id,
        reporting_airline_code,

        COUNT(*) FILTER (
            WHERE delay_cause_reported = TRUE
        ) AS flights_with_reported_delay_causes,

        SUM(carrier_delay_minutes)::NUMERIC
            AS carrier_delay_minutes,

        SUM(weather_delay_minutes)::NUMERIC
            AS weather_delay_minutes,

        SUM(national_air_system_delay_minutes)::NUMERIC
            AS national_air_system_delay_minutes,

        SUM(security_delay_minutes)::NUMERIC
            AS security_delay_minutes,

        SUM(late_aircraft_delay_minutes)::NUMERIC
            AS late_aircraft_delay_minutes

    FROM analytics.vw_flight_detail

    GROUP BY
        reporting_airline_id,
        reporting_airline_code
),

delay_cause_long_format AS (
    SELECT
        airline_delay_totals.reporting_airline_id,
        airline_delay_totals.reporting_airline_code,
        airline_delay_totals.flights_with_reported_delay_causes,
        delay_values.delay_cause,
        delay_values.total_delay_minutes

    FROM airline_delay_totals

    CROSS JOIN LATERAL (
        VALUES
            (
                'Carrier',
                airline_delay_totals.carrier_delay_minutes
            ),
            (
                'Weather',
                airline_delay_totals.weather_delay_minutes
            ),
            (
                'National Air System',
                airline_delay_totals.national_air_system_delay_minutes
            ),
            (
                'Security',
                airline_delay_totals.security_delay_minutes
            ),
            (
                'Late Aircraft',
                airline_delay_totals.late_aircraft_delay_minutes
            )
    ) AS delay_values (
        delay_cause,
        total_delay_minutes
    )
)

SELECT
    reporting_airline_id,
    reporting_airline_code,
    flights_with_reported_delay_causes,
    delay_cause,

    ROUND(
        total_delay_minutes,
        2
    ) AS total_delay_minutes,

    ROUND(
        (
            100.0
            * total_delay_minutes
            / NULLIF(
                SUM(total_delay_minutes)
                OVER (
                    PARTITION BY reporting_airline_id
                ),
                0
            )
        )::NUMERIC,
        2
    ) AS airline_delay_share_percentage

FROM delay_cause_long_format;