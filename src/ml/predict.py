"""
FlightDelayPredictor — Production inference class for AeroPulse.

Loads pre-trained models from models/ and provides a clean predict() interface
for use in the Streamlit dashboard and any downstream API.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.ml.features import (
    FAA_DELAY_COST_PER_MINUTE_USD,
    FEATURE_COLUMNS,
    engineer_features,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"


class FlightDelayPredictor:
    """
    Production delay predictor that wraps trained GBM models.

    Usage:
        predictor = FlightDelayPredictor()
        result = predictor.predict(
            carrier="AA",
            origin="JFK",
            destination="LAX",
            scheduled_hour=14,
            day_of_week=2,       # 1=Mon .. 7=Sun
            distance_group=10,
        )
        print(result)
    """

    RISK_THRESHOLDS = {
        "low":      (0.00, 0.20),
        "moderate": (0.20, 0.40),
        "high":     (0.40, 1.01),
    }

    def __init__(self) -> None:
        self._reg_model = None
        self._cls_model = None
        self._fit_stats: dict = {}
        self._metadata: dict = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        reg_path = MODELS_DIR / "delay_regressor.joblib"
        cls_path = MODELS_DIR / "delay_classifier.joblib"
        stats_path = MODELS_DIR / "fit_stats.joblib"
        meta_path = MODELS_DIR / "training_metadata.json"

        if not reg_path.exists():
            raise FileNotFoundError(
                "Trained models not found. Run: python -m src.ml.train"
            )

        self._reg_model = joblib.load(reg_path)
        self._cls_model = joblib.load(cls_path)
        self._fit_stats = joblib.load(stats_path)

        if meta_path.exists():
            with open(meta_path) as f:
                self._metadata = json.load(f)

        self._loaded = True

    def predict(
        self,
        carrier: str,
        origin: str,
        destination: str,
        scheduled_hour: int,
        day_of_week: int,
        distance_group: int = 5,
        is_weekend: bool = False,
    ) -> dict:
        """
        Predict delay risk and estimated delay duration for a single flight.

        Args:
            carrier: IATA airline code (e.g. "AA", "DL", "UA")
            origin: IATA origin airport code (e.g. "JFK")
            destination: IATA destination airport code (e.g. "LAX")
            scheduled_hour: Departure hour 0–23
            day_of_week: Day of week 1 (Mon) – 7 (Sun)
            distance_group: BTS distance group 1–11 (proxy for flight duration)
            is_weekend: True if Saturday or Sunday

        Returns:
            dict with delay probability, estimated minutes, risk level, cost impact
        """
        self._load()

        route_code = f"{origin}-{destination}"

        # Build a single-row DataFrame matching the training schema
        row = pd.DataFrame([{
            "reporting_airline_code": carrier,
            "route_code": route_code,
            "scheduled_departure_hour": scheduled_hour,
            "day_of_week": day_of_week,
            "distance_group": distance_group,
            "is_weekend": is_weekend,
            "flight_status": "Completed",
            "arrival_delayed_15": 0,       # dummy target for feature logic
            "arrival_delay_minutes": 0.0,  # dummy target
        }])

        X, _ = engineer_features(row, fit_stats=self._fit_stats)

        delay_prob = float(self._cls_model.predict_proba(X)[0, 1])
        est_delay_minutes = float(max(0, self._reg_model.predict(X)[0]))
        cost_impact_usd = est_delay_minutes * FAA_DELAY_COST_PER_MINUTE_USD

        # Determine risk tier
        risk_level = "low"
        for level, (lo, hi) in self.RISK_THRESHOLDS.items():
            if lo <= delay_prob < hi:
                risk_level = level
                break

        # Pull top drivers from trained model metadata
        saved_importances = self._metadata.get("feature_importance", {})
        if saved_importances:
            top_drivers = list(saved_importances.items())[:4]
        else:
            top_drivers = [(f, 0.1) for f in FEATURE_COLUMNS[:4]]

        return {
            "carrier": carrier,
            "route": route_code,
            "scheduled_hour": scheduled_hour,
            "day_of_week": day_of_week,
            "delay_probability": round(delay_prob, 4),
            "delay_probability_pct": round(delay_prob * 100, 1),
            "estimated_delay_minutes": round(est_delay_minutes, 1),
            "risk_level": risk_level,
            "cost_impact_usd": round(cost_impact_usd, 2),
            "top_delay_drivers": [
                {"feature": f, "importance": round(float(v), 4)}
                for f, v in top_drivers
            ],
            "interpretation": self._interpret(risk_level, delay_prob, est_delay_minutes),
        }

    @staticmethod
    def _interpret(risk_level: str, prob: float, est_min: float) -> str:
        if risk_level == "low":
            return (
                f"This flight has a {prob*100:.0f}% on-time risk -- well within normal "
                f"operating parameters. Expected delay: ~{est_min:.0f} min."
            )
        elif risk_level == "moderate":
            return (
                f"Moderate delay risk ({prob*100:.0f}%). Operations should monitor gate "
                f"turnaround times. Expected delay: ~{est_min:.0f} min."
            )
        else:
            return (
                f"High delay risk ({prob*100:.0f}%). Recommend proactive passenger "
                f"rebooking and gate buffer allocation. Expected delay: ~{est_min:.0f} min."
            )


    def get_model_metrics(self) -> dict:
        """Return training metadata and evaluation metrics."""
        self._load()
        return self._metadata
