"""Train and chronologically evaluate a LightGBM 30-minute traffic forecaster."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
FEATURES = [
    "total_demand_gbps",
    "lag_1",
    "lag_2",
    "lag_4",
    "lag_8",
    "lag_96",
    "rolling_mean_4",
    "rolling_mean_8",
    "target_hour",
    "target_minute",
    "target_dayofweek",
    "target_is_weekend",
    "target_is_festival",
]
PREDICTION_HORIZON_STEPS = 2


def train_forecaster(dataset: pd.DataFrame):
    required = set(FEATURES) | {
        "timestamp", "target_timestamp", "target"
    }
    missing = required - set(dataset.columns)
    if missing:
        raise ValueError(f"Forecast dataset is missing columns: {sorted(missing)}")

    data = dataset.copy().reset_index(drop=True)
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise")
    data["target_timestamp"] = pd.to_datetime(data["target_timestamp"], errors="raise")

    expected_target_times = data["timestamp"] + pd.Timedelta(minutes=30)
    if not (data["target_timestamp"] == expected_target_times).all():
        raise ValueError("Forecast target timestamps are not exactly 30 minutes after input timestamps.")

    X = data[FEATURES]
    y = pd.to_numeric(data["target"], errors="raise")
    if X.isna().any().any() or y.isna().any():
        raise ValueError("Forecast features or targets contain missing values.")

    split = int(len(data) * 0.8)
    # A 2-step-ahead target overlaps the boundary unless the final
    # two training rows are purged. The test set remains untouched.
    gap = PREDICTION_HORIZON_STEPS
    train_end = split - gap
    if train_end < 1 or split >= len(data):
        raise ValueError("Not enough data for the chronological train/test split.")

    X_train, X_test = X.iloc[:train_end], X.iloc[split:]
    y_train, y_test = y.iloc[:train_end], y.iloc[split:]
    test_rows = data.iloc[split:].copy()

    model = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=15,
        random_state=42,
        verbosity=-1,
    )
    model.fit(X_train, y_train)

    predicted = np.maximum(model.predict(X_test), 0.0)
    baseline_predicted = X_test["total_demand_gbps"].to_numpy()

    mae = float(mean_absolute_error(y_test, predicted))
    rmse = float(np.sqrt(mean_squared_error(y_test, predicted)))
    r2 = float(r2_score(y_test, predicted))
    baseline_mae = float(mean_absolute_error(y_test, baseline_predicted))
    improvement = (
        (baseline_mae - mae) / baseline_mae * 100.0
        if baseline_mae > 0 else 0.0
    )

    predictions = pd.DataFrame(
        {
            "timestamp": test_rows["timestamp"].to_numpy(),
            "target_timestamp": test_rows["target_timestamp"].to_numpy(),
            "actual": y_test.to_numpy(),
            "predicted": predicted,
            "baseline": baseline_predicted,
        }
    )

    metrics = {
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "purged_boundary_samples": int(gap),
        "prediction_horizon_minutes": 30,
        "mae_gbps": mae,
        "rmse_gbps": rmse,
        "r2": r2,
        "persistence_mae_gbps": baseline_mae,
        "mae_improvement_percent": float(improvement),
        "features": FEATURES,
    }

    return model, predictions, metrics


if __name__ == "__main__":
    dataset_path = ROOT / "data" / "processed" / "forecast_dataset.csv"
    dataset = pd.read_csv(
        dataset_path,
        parse_dates=["timestamp", "target_timestamp"],
    )

    model, predictions, metrics = train_forecaster(dataset)

    models_dir = ROOT / "models"
    processed_dir = ROOT / "data" / "processed"
    models_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, models_dir / "traffic_forecaster.pkl")
    predictions.to_csv(processed_dir / "forecast_predictions.csv", index=False)
    with open(processed_dir / "forecast_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\nLIGHTGBM MODEL PERFORMANCE")
    print("--------------------------")
    print("Training samples:", metrics["train_samples"])
    print("Testing samples:", metrics["test_samples"])
    print("Purged boundary samples:", metrics["purged_boundary_samples"])
    print(f"MAE:  {metrics['mae_gbps']:.4f} Gbps")
    print(f"RMSE: {metrics['rmse_gbps']:.4f} Gbps")
    print(f"R2:   {metrics['r2']:.4f}")
    print(f"Persistence MAE: {metrics['persistence_mae_gbps']:.4f} Gbps")
    print(f"MAE improvement: {metrics['mae_improvement_percent']:.2f}%")
    print("\nModel saved: models/traffic_forecaster.pkl")
    print("Predictions saved: data/processed/forecast_predictions.csv")
    print("Metrics saved: data/processed/forecast_metrics.json")
