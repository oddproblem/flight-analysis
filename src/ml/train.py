"""
AeroPulse ML Training Pipeline — Flight Delay Forecasting.

Methodology:
    1. Load cleaned real-world flight operational data.
    2. Filter to completed, non-diverted flights only (operational focus).
    3. Strict out-of-time train/test split — Days 1-23 train, Days 24-31 test.
       This mirrors production deployment where the model predicts future flights.
       Random splits would introduce data leakage via carrier/route statistics.
    4. Build three tiers of predictions:
       a. Naive baseline: global median delay (zero-information benchmark).
       b. Operational heuristic: route × carrier historical median (operational benchmark).
       c. ML model: GradientBoostingRegressor and LogisticRegression (classification).
    5. Evaluate all tiers with MAE, RMSE, MAPE, Median AE, R² and binary metrics.
    6. Persist model artefacts to models/.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.ml.features import (
    FEATURE_COLUMNS,
    TARGET_BINARY,
    TARGET_REGRESSION,
    engineer_features,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
PRIMARY_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "flights_clean.parquet"
COMPAT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "flights_2024_01_clean.parquet"

TRAIN_CUTOFF_DAY = 23   # Days 1–23 are training history
TEST_START_DAY = 24     # Days 24–31 are the out-of-time validation window


def _load_data() -> pd.DataFrame:
    """Load cleaned real flight data."""
    if PRIMARY_DATA_PATH.exists():
        print(f"  Loading real flight data from: {PRIMARY_DATA_PATH.name}")
        df = pd.read_parquet(PRIMARY_DATA_PATH)
    elif COMPAT_DATA_PATH.exists():
        print(f"  Loading real flight data from: {COMPAT_DATA_PATH.name}")
        df = pd.read_parquet(COMPAT_DATA_PATH)
    else:
        raise FileNotFoundError(
            "Cleaned data file not found. Run the cleaning pipeline first:\n"
            "  python -m src.transform.clean_flight_data"
        )
    return df



def _filter_for_ml(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only completed flights with known delay outcomes."""
    df = df[df["flight_status"] == "Completed"].copy()
    df = df[df[TARGET_REGRESSION].notna()].copy()
    df = df[df[TARGET_BINARY].notna()].copy()
    df[TARGET_BINARY] = df[TARGET_BINARY].astype(int)
    df[TARGET_REGRESSION] = df[TARGET_REGRESSION].astype(float)
    return df.reset_index(drop=True)


def _time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strict chronological train/test split on day of month."""
    train = df[df["day_of_month"] <= TRAIN_CUTOFF_DAY].copy()
    test = df[df["day_of_month"] >= TEST_START_DAY].copy()
    return train, test


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    median_ae = float(np.median(np.abs(y_true - y_pred)))
    # MAPE: skip rows where true=0 to avoid division by zero
    nonzero = y_true != 0
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100) if nonzero.any() else float("nan")
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "model": label,
        "mae_minutes": round(mae, 3),
        "rmse_minutes": round(rmse, 3),
        "median_ae_minutes": round(median_ae, 3),
        "mape_pct": round(mape, 2),
        "r2": round(r2, 4),
    }


def _classification_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, label: str) -> dict:
    y_pred_class = (y_pred_prob >= 0.5).astype(int)
    return {
        "model": label,
        "roc_auc": round(roc_auc_score(y_true, y_pred_prob), 4),
        "accuracy": round(accuracy_score(y_true, y_pred_class), 4),
        "precision": round(precision_score(y_true, y_pred_class, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred_class, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred_class, zero_division=0), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# BASELINES
# ─────────────────────────────────────────────────────────────────────────────

def _fit_baselines(train: pd.DataFrame) -> dict:
    """
    Fit two baseline predictors from training data only.

    Baseline 1 — Naive Global Median:
        Predicts the same median arrival delay for every flight.
        This is the zero-information floor that any model must beat.

    Baseline 2 — Route × Carrier Operational Heuristic:
        For each (route_code, reporting_airline_code) pair, predict
        the historical median delay seen during the training window.
        This is what an airline operations team would use manually.
        Falls back to carrier median → global median for unseen pairs.
    """
    global_median = float(train[TARGET_REGRESSION].median())
    global_binary_rate = float(train[TARGET_BINARY].mean())

    # Route × carrier median delay
    route_carrier = (
        train.groupby(["route_code", "reporting_airline_code"])[TARGET_REGRESSION]
        .median()
        .to_dict()
    )
    # Carrier fallback
    carrier_median = (
        train.groupby("reporting_airline_code")[TARGET_REGRESSION]
        .median()
        .to_dict()
    )

    return {
        "global_median": global_median,
        "global_binary_rate": global_binary_rate,
        "route_carrier_median": route_carrier,
        "carrier_median": carrier_median,
    }


def _apply_heuristic(test: pd.DataFrame, baselines: dict) -> np.ndarray:
    """Apply the route × carrier heuristic baseline to test set."""
    preds = []
    for _, row in test.iterrows():
        key = (row["route_code"], row["reporting_airline_code"])
        pred = baselines["route_carrier_median"].get(key)
        if pred is None:
            pred = baselines["carrier_median"].get(row["reporting_airline_code"])
        if pred is None:
            pred = baselines["global_median"]
        preds.append(float(pred))
    return np.array(preds)


# ─────────────────────────────────────────────────────────────────────────────
# ML MODELS
# ─────────────────────────────────────────────────────────────────────────────

def _train_regression_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Histogram-based Gradient Boosting regressor to predict delay duration in minutes.

    Utilizes multi-threaded binning for sub-second training across 300,000+ real records.
    max_iter=100 with learning_rate=0.08 provides strong convergence without overfitting.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("gbr", HistGradientBoostingRegressor(
            max_iter=100,
            max_depth=6,
            learning_rate=0.08,
            min_samples_leaf=30,
            random_state=42,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def _train_classification_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Histogram-based Gradient Boosting classifier to predict binary delay risk (>=15 min).
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("gbc", HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=6,
            learning_rate=0.08,
            min_samples_leaf=30,
            random_state=42,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


# -----------------------------------------------------------------------------
# MAIN TRAINING ENTRY POINT
# -----------------------------------------------------------------------------

def train_and_evaluate() -> dict:
    """
    Full training pipeline: data -> features -> baselines -> ML -> evaluation.
    Returns a results dictionary with all metric tables and saved artefact paths.
    """
    print("=" * 70)
    print("AEROPULSE ML TRAINING PIPELINE")
    print("=" * 70)

    # -- Load and filter -------------------------------------------------------
    print("\n[1/6] Loading data...")
    df = _load_data()
    df = _filter_for_ml(df)
    print(f"  Modelling dataset: {len(df):,} completed flights")

    # -- Time-based split -----------------------------------------------------
    print(f"\n[2/6] Out-of-time split (train <= day {TRAIN_CUTOFF_DAY}, test >= day {TEST_START_DAY})...")
    train_raw, test_raw = _time_split(df)
    print(f"  Train: {len(train_raw):,} flights | Test: {len(test_raw):,} flights")

    # -- Fit baselines (on train only) -----------------------------------------
    print("\n[3/6] Fitting baselines from training data...")
    baselines = _fit_baselines(train_raw)
    print(f"  Global median delay: {baselines['global_median']:.1f} min")
    print(f"  Route×carrier pairs learned: {len(baselines['route_carrier_median']):,}")

    # -- Feature engineering (fit on train, apply to test - no leakage) --------
    print("\n[4/6] Engineering features (fit on train, apply to test - no leakage)...")
    X_train, fit_stats = engineer_features(train_raw, fit_stats=None)
    X_test, _ = engineer_features(test_raw, fit_stats=fit_stats)
    y_train_reg = train_raw[TARGET_REGRESSION].values
    y_test_reg = test_raw[TARGET_REGRESSION].values
    y_train_cls = train_raw[TARGET_BINARY].values
    y_test_cls = test_raw[TARGET_BINARY].values
    print(f"  Features: {list(X_train.columns)}")

    # -- Train ML models -------------------------------------------------------
    print("\n[5/6] Training ML models...")
    reg_model = _train_regression_model(X_train, y_train_reg)
    cls_model = _train_classification_model(X_train, y_train_cls)
    print("  HistGradientBoosting regressor trained [OK]")
    print("  HistGradientBoosting classifier trained [OK]")

    # -- Evaluate all tiers ----------------------------------------------------
    print("\n[6/6] Evaluating on out-of-time test set...")

    # Regression predictions
    pred_naive = np.full(len(y_test_reg), baselines["global_median"])
    pred_heuristic = _apply_heuristic(test_raw, baselines)
    pred_ml_reg = reg_model.predict(X_test)
    pred_ml_reg = np.clip(pred_ml_reg, 0, None)  # delay cannot be negative

    # Classification predictions (probability of delay >= 15 min)
    pred_naive_prob = np.full(len(y_test_cls), baselines["global_binary_rate"])
    pred_ml_prob = cls_model.predict_proba(X_test)[:, 1]

    regression_results = [
        _regression_metrics(y_test_reg, pred_naive,     "Naive Global Median"),
        _regression_metrics(y_test_reg, pred_heuristic, "Route x Carrier Heuristic"),
        _regression_metrics(y_test_reg, pred_ml_reg,    "HistGBM Delay Forecaster"),
    ]

    classification_results = [
        _classification_metrics(y_test_cls, pred_naive_prob, "Naive Rate Baseline"),
        _classification_metrics(y_test_cls, pred_ml_prob,    "HistGBM Delay Classifier"),
    ]

    # -- Feature importance (Permutation Importance on Out-of-Time Test Set) ---
    sample_size = min(5000, len(X_test))
    sample_indices = np.random.RandomState(42).choice(len(X_test), size=sample_size, replace=False)
    perm_eval = permutation_importance(
        reg_model, X_test.iloc[sample_indices], y_test_reg[sample_indices],
        n_repeats=3, random_state=42
    )
    raw_imp = np.clip(perm_eval.importances_mean, 0, None)
    total_imp = np.sum(raw_imp)
    norm_imp = (raw_imp / total_imp) if total_imp > 0 else raw_imp
    feature_importance = dict(zip(FEATURE_COLUMNS, [round(float(v), 5) for v in norm_imp]))
    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

    # -- Residual analysis ----------------------------------------------------
    residuals = y_test_reg - pred_ml_reg
    residual_analysis = {
        "mean_residual": round(float(np.mean(residuals)), 3),
        "std_residual": round(float(np.std(residuals)), 3),
        "pct_under_predicted": round(float((residuals < 0).mean() * 100), 2),
        "pct_over_predicted": round(float((residuals > 0).mean() * 100), 2),
        "p95_absolute_error": round(float(np.percentile(np.abs(residuals), 95)), 1),
        "p99_absolute_error": round(float(np.percentile(np.abs(residuals), 99)), 1),
    }

    # -- Save artefacts --------------------------------------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(reg_model, MODELS_DIR / "delay_regressor.joblib")
    joblib.dump(cls_model, MODELS_DIR / "delay_classifier.joblib")
    joblib.dump(fit_stats, MODELS_DIR / "fit_stats.joblib")

    metadata = {
        "train_cutoff_day": TRAIN_CUTOFF_DAY,
        "test_start_day": TEST_START_DAY,
        "train_rows": len(train_raw),
        "test_rows": len(test_raw),
        "features": FEATURE_COLUMNS,
        "regression_results": regression_results,
        "classification_results": classification_results,
        "feature_importance": feature_importance,
        "residual_analysis": residual_analysis,
        "baselines": {
            "global_median_minutes": baselines["global_median"],
            "global_delay_rate": baselines["global_binary_rate"],
        },
    }
    with open(MODELS_DIR / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # -- Print comparison table ------------------------------------------------
    print("\n" + "=" * 70)
    print("REGRESSION RESULTS (Out-of-Time Test Set)")
    print("=" * 70)
    print(f"{'Model':<35} {'MAE':>8} {'RMSE':>8} {'MdAE':>8} {'MAPE%':>8} {'R2':>8}")
    print("-" * 70)
    for r in regression_results:
        print(
            f"{r['model']:<35} "
            f"{r['mae_minutes']:>8.2f} "
            f"{r['rmse_minutes']:>8.2f} "
            f"{r['median_ae_minutes']:>8.2f} "
            f"{r['mape_pct']:>8.1f} "
            f"{r['r2']:>8.4f}"
        )

    print("\n" + "=" * 70)
    print("CLASSIFICATION RESULTS (Delay >= 15 min, Out-of-Time Test Set)")
    print("=" * 70)
    print(f"{'Model':<35} {'AUC':>8} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    print("-" * 70)
    for c in classification_results:
        print(
            f"{c['model']:<35} "
            f"{c['roc_auc']:>8.4f} "
            f"{c['accuracy']:>8.4f} "
            f"{c['precision']:>8.4f} "
            f"{c['recall']:>8.4f} "
            f"{c['f1']:>8.4f}"
        )

    print("\n" + "=" * 70)
    print("TOP DELAY DRIVERS (Permutation Feature Importance)")
    print("=" * 70)
    for feat, imp in feature_importance.items():
        bar = "#" * int(imp * 100)
        print(f"  {feat:<40} {imp:.4f}  {bar}")

    print("\n" + "=" * 70)
    print("RESIDUAL ANALYSIS")
    print("=" * 70)
    for k, v in residual_analysis.items():
        print(f"  {k:<40} {v}")

    print(f"\nArtefacts saved to: {MODELS_DIR}")
    print("=" * 70)


    return metadata


if __name__ == "__main__":
    train_and_evaluate()
