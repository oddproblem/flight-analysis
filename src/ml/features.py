"""
Feature engineering for flight delay prediction.

This module derives operational and temporal features from cleaned flight data
for use in the ML delay forecasting pipeline. Features are designed to capture
domain-meaningful signals — not just raw correlations.

Feature design rationale:
- Carrier historical delay rate: accounts for airline-specific operational quality
- Route congestion index: captures airport-pair bottlenecks independent of carrier
- Departure hour: nonlinear delay cascade effect across the day
- Distance group: proxy for flight duration and weather exposure window
- Day of week: captures schedule-driven demand and crew rotation effects
- Is weekend: leisure vs. business demand shift
"""

from __future__ import annotations

import pandas as pd
import numpy as np


# Delay cost per minute (FAA standard estimate, U.S. air carrier operations)
FAA_DELAY_COST_PER_MINUTE_USD = 101.90

# Operational delay threshold used by BTS and FAA performance benchmarks
DELAY_THRESHOLD_MINUTES = 15

# Features used for model training (ordered)
FEATURE_COLUMNS = [
    "scheduled_departure_hour",
    "day_of_week",
    "is_weekend_flag",
    "distance_group",
    "carrier_historical_delay_rate",
    "carrier_avg_delay_minutes",
    "route_congestion_index",
    "route_avg_delay_minutes",
    "departure_hour_sin",
    "departure_hour_cos",
    "is_peak_hour",
    "is_red_eye",
]

TARGET_BINARY = "arrival_delayed_15"       # 1 = delayed ≥15 min, 0 = on time
TARGET_REGRESSION = "arrival_delay_minutes"  # continuous delay duration in minutes



def engineer_features(df: pd.DataFrame, fit_stats: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Derive all ML features from cleaned flight data.

    Args:
        df: Cleaned flight DataFrame (output of clean_flight_dataframe).
        fit_stats: Pre-computed carrier/route statistics from training set.
                   Pass None during training (will compute and return).
                   Pass the returned dict during inference to avoid leakage.

    Returns:
        (feature_df, fit_stats): Feature DataFrame and statistics dictionary.
    """
    df = df.copy()

    # ── 1. Convert flag columns to numeric ───────────────────────────────────
    df["is_weekend_flag"] = df["is_weekend"].astype(int) if "is_weekend" in df.columns else 0
    df["scheduled_departure_hour"] = pd.to_numeric(
        df["scheduled_departure_hour"], errors="coerce"
    ).fillna(12).astype(int)
    df["day_of_week"] = pd.to_numeric(df["day_of_week"], errors="coerce").fillna(3).astype(int)
    df["distance_group"] = pd.to_numeric(df["distance_group"], errors="coerce").fillna(5).astype(int)

    # ── 2. Cyclic encoding of departure hour ─────────────────────────────────
    hour = df["scheduled_departure_hour"]
    df["departure_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["departure_hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # ── 3. Peak and red-eye hour flags ───────────────────────────────────────
    df["is_peak_hour"] = hour.between(7, 9) | hour.between(17, 20)
    df["is_peak_hour"] = df["is_peak_hour"].astype(int)
    df["is_red_eye"] = (hour <= 5).astype(int)

    # ── 4. Carrier-level historical statistics ────────────────────────────────
    if fit_stats is None:
        fit_stats = {}
        completed = df[df["flight_status"] == "Completed"].copy() if "flight_status" in df.columns else df

        carrier_stats = (
            completed.groupby("reporting_airline_code")["arrival_delayed_15"]
            .agg(delay_rate="mean", count="count")
            .reset_index()
        )
        carrier_delay_rate = dict(zip(carrier_stats["reporting_airline_code"], carrier_stats["delay_rate"]))

        carrier_avg_delay = (
            completed.groupby("reporting_airline_code")["arrival_delay_minutes"]
            .mean()
            .to_dict()
        )

        fit_stats["carrier_delay_rate"] = carrier_delay_rate
        fit_stats["carrier_avg_delay"] = carrier_avg_delay
        fit_stats["global_delay_rate"] = completed["arrival_delayed_15"].mean()
        fit_stats["global_avg_delay"] = completed["arrival_delay_minutes"].mean()

    global_delay_rate = fit_stats.get("global_delay_rate", 0.24)
    global_avg_delay = fit_stats.get("global_avg_delay", 10.0)

    df["carrier_historical_delay_rate"] = df["reporting_airline_code"].map(
        fit_stats["carrier_delay_rate"]
    ).fillna(global_delay_rate)

    df["carrier_avg_delay_minutes"] = df["reporting_airline_code"].map(
        fit_stats["carrier_avg_delay"]
    ).fillna(global_avg_delay)

    # ── 5. Route congestion index ─────────────────────────────────────────────
    if "route_congestion_index" not in fit_stats:
        completed = df[df["flight_status"] == "Completed"].copy() if "flight_status" in df.columns else df
        route_stats = (
            completed.groupby("route_code")["arrival_delayed_15"]
            .agg(delay_rate="mean", count="count")
        )
        # Smooth low-count routes toward global average (Bayesian shrinkage)
        k = 30  # shrinkage factor
        route_smooth = (
            route_stats["delay_rate"] * route_stats["count"]
            + global_delay_rate * k
        ) / (route_stats["count"] + k)

        route_avg_delay = (
            completed.groupby("route_code")["arrival_delay_minutes"].mean().to_dict()
        )

        fit_stats["route_congestion_index"] = route_smooth.to_dict()
        fit_stats["route_avg_delay"] = route_avg_delay

    df["route_congestion_index"] = df["route_code"].map(
        fit_stats["route_congestion_index"]
    ).fillna(global_delay_rate)

    df["route_avg_delay_minutes"] = df["route_code"].map(
        fit_stats["route_avg_delay"]
    ).fillna(global_avg_delay)

    return df[FEATURE_COLUMNS], fit_stats


def compute_cost_impact(
    delay_minutes: float | pd.Series,
    cost_per_minute: float = FAA_DELAY_COST_PER_MINUTE_USD,
) -> float | pd.Series:
    """Convert delay minutes to USD cost using FAA benchmark rate."""
    return delay_minutes * cost_per_minute
