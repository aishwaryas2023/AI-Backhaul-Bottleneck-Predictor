"""Build the 30-minute-ahead aggregate traffic forecasting dataset."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PREDICTION_HORIZON_STEPS = 2  # 2 x 15 minutes = 30 minutes


def parse_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )


def prepare_dataset(traffic_df: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "tower", "demand_gbps", "event", "fiber_cut"}
    missing = required - set(traffic_df.columns)
    if missing:
        raise ValueError(f"Traffic data is missing columns: {sorted(missing)}")

    traffic = traffic_df.copy()
    traffic["timestamp"] = pd.to_datetime(traffic["timestamp"], errors="raise")
    traffic["demand_gbps"] = pd.to_numeric(traffic["demand_gbps"], errors="raise")
    traffic["fiber_cut"] = parse_bool_series(traffic["fiber_cut"])

    if traffic.duplicated(["timestamp", "tower"]).any():
        raise ValueError("Duplicate timestamp/tower records found.")
    if traffic["demand_gbps"].isna().any() or (traffic["demand_gbps"] < 0).any():
        raise ValueError("Traffic demand must be present and non-negative.")

    tower_counts = traffic.groupby("timestamp")["tower"].nunique()
    expected_towers = traffic["tower"].nunique()
    if not (tower_counts == expected_towers).all():
        raise ValueError("Every timestamp must contain one record for each tower.")

    network = (
        traffic.groupby("timestamp", sort=True)
        .agg(
            total_demand_gbps=("demand_gbps", "sum"),
            event=("event", "first"),
            fiber_cut=("fiber_cut", "max"),
        )
        .reset_index()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # Verify a continuous 15-minute sampling interval.
    deltas = network["timestamp"].diff().dropna()
    if not (deltas == pd.Timedelta(minutes=15)).all():
        raise ValueError("Aggregate traffic timestamps must be continuous 15-minute intervals.")

    for lag in [1, 2, 4, 8, 96]:
        network[f"lag_{lag}"] = network["total_demand_gbps"].shift(lag)

    # Rolling means use only observations strictly before the input timestamp.
    shifted_demand = network["total_demand_gbps"].shift(1)
    network["rolling_mean_4"] = shifted_demand.rolling(4).mean()
    network["rolling_mean_8"] = shifted_demand.rolling(8).mean()

    network["target_timestamp"] = (
        network["timestamp"]
        + pd.Timedelta(minutes=15 * PREDICTION_HORIZON_STEPS)
    )
    network["target"] = network["total_demand_gbps"].shift(-PREDICTION_HORIZON_STEPS)

    # Calendar features are evaluated at the forecast target time.
    # The festival schedule is pre-announced in this synthetic scenario.
    target_event = network["event"].shift(-PREDICTION_HORIZON_STEPS)
    network["target_hour"] = network["target_timestamp"].dt.hour
    network["target_minute"] = network["target_timestamp"].dt.minute
    network["target_dayofweek"] = network["target_timestamp"].dt.dayofweek
    network["target_is_weekend"] = (network["target_dayofweek"] >= 5).astype(int)
    network["target_is_festival"] = target_event.eq("festival").astype(int)

    # The final horizon rows have no observed target and are removed.
    network = network.dropna().reset_index(drop=True)

    if network.empty:
        raise ValueError("No usable rows remain after feature construction.")

    return network


if __name__ == "__main__":
    traffic_path = ROOT / "data" / "raw" / "traffic.csv"
    traffic = pd.read_csv(traffic_path, parse_dates=["timestamp"])
    dataset = prepare_dataset(traffic)

    output_path = ROOT / "data" / "processed" / "forecast_dataset.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)

    print("\nFORECAST DATASET CREATED")
    print("Shape:", dataset.shape)
    print("Input range:", dataset["timestamp"].min(), "to", dataset["timestamp"].max())
    print("Target range:", dataset["target_timestamp"].min(), "to", dataset["target_timestamp"].max())
    print("Saved: data/processed/forecast_dataset.csv")
    print(dataset.head(3).to_string(index=False))
