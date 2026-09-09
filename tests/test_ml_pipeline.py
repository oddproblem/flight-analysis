"""Unit tests for the AeroPulse machine learning pipeline."""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.ml.features import (
    FEATURE_COLUMNS,
    TARGET_BINARY,
    TARGET_REGRESSION,
    engineer_features,
)
from src.ml.predict import FlightDelayPredictor


class TestMLPipeline(unittest.TestCase):
    def setUp(self):
        # Create a small realistic operational dataset
        self.sample_df = pd.DataFrame({
            "flight_date": pd.date_range("2015-01-01", periods=10, freq="D"),
            "reporting_airline_code": ["AA", "DL", "UA", "WN", "AA", "DL", "B6", "AS", "UA", "WN"],
            "origin_airport_code": ["JFK", "ATL", "ORD", "DFW", "LAX", "SFO", "BOS", "SEA", "DEN", "MCO"],
            "destination_airport_code": ["LAX", "ORD", "ATL", "DEN", "JFK", "LAX", "FLL", "SFO", "IAH", "ATL"],
            "route_code": ["JFK-LAX", "ATL-ORD", "ORD-ATL", "DFW-DEN", "LAX-JFK", "SFO-LAX", "BOS-FLL", "SEA-SFO", "DEN-IAH", "MCO-ATL"],
            "scheduled_departure_hour": [6, 8, 12, 14, 16, 17, 19, 21, 10, 15],
            "day_of_week": [1, 2, 3, 4, 5, 6, 7, 1, 2, 3],
            "is_weekend": [False, False, False, False, False, True, True, False, False, False],
            "distance_group": [10, 3, 3, 4, 10, 2, 5, 3, 4, 2],
            "arrival_delay_minutes": [0.0, 18.0, 5.0, 42.0, 0.0, 12.0, 65.0, 0.0, 8.0, 25.0],
            "arrival_delayed_15": [0, 1, 0, 1, 0, 0, 1, 0, 0, 1],
            "flight_status": ["Completed"] * 10,
        })

    def test_feature_engineering_columns(self):
        X, fit_stats = engineer_features(self.sample_df)
        for col in FEATURE_COLUMNS:
            self.assertIn(col, X.columns)
        self.assertEqual(len(X), len(self.sample_df))
        self.assertIn("carrier_delay_rate", fit_stats)
        self.assertIn("route_congestion_index", fit_stats)

    def test_predictor_inference(self):
        predictor = FlightDelayPredictor()
        result = predictor.predict(
            carrier="AA",
            origin="JFK",
            destination="LAX",
            scheduled_hour=14,
            day_of_week=2,
            distance_group=10,
        )
        self.assertIn("delay_probability", result)
        self.assertIn("estimated_delay_minutes", result)
        self.assertIn("risk_level", result)
        self.assertIn(result["risk_level"], ["low", "moderate", "high"])
        self.assertGreaterEqual(result["delay_probability"], 0.0)
        self.assertLessEqual(result["delay_probability"], 1.0)
        self.assertGreaterEqual(result["estimated_delay_minutes"], 0.0)


if __name__ == "__main__":
    unittest.main()
