"""Cross-file integrity."""

from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

from src.network import create_network

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def read_csv(name, dates=()):
    path = PROCESSED / name
    if not path.exists():
        raise FileNotFoundError(f"Required output is missing: {path}")
    return pd.read_csv(path, parse_dates=list(dates))


def as_bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def main():
    traffic = pd.read_csv(RAW / "traffic.csv", parse_dates=["timestamp"])
    forecast = read_csv("forecast_dataset.csv", ["timestamp", "target_timestamp"])
    predictions = read_csv("forecast_predictions.csv", ["timestamp", "target_timestamp"])
    adjusted = read_csv("forecast_adjusted_traffic.csv", ["timestamp", "forecast_origin_timestamp"])
    full_baseline = read_csv("baseline_links.csv", ["timestamp"])
    test_baseline = read_csv("test_baseline_links.csv", ["timestamp"])
    optimized = read_csv("optimized_links.csv", ["timestamp"])

    # Raw synthetic traffic checks.
    assert len(traffic) == 21 * 96 * 8, f"Unexpected raw traffic row count: {len(traffic)}"
    assert traffic["timestamp"].nunique() == 21 * 96
    assert traffic["tower"].nunique() == 8
    assert not traffic.duplicated(["timestamp", "tower"]).any()
    raw_delta = traffic[["timestamp"]].drop_duplicates().sort_values("timestamp")["timestamp"].diff().dropna()
    assert (raw_delta == pd.Timedelta(minutes=15)).all()
    assert traffic.loc[traffic["event"] == "festival", "timestamp"].nunique() == 40
    traffic["fiber_cut"] = as_bool(traffic["fiber_cut"])
    assert traffic.loc[traffic["fiber_cut"], "timestamp"].nunique() == 16

    # Forecast dataset and chronological split checks.
    assert (forecast["target_timestamp"] == forecast["timestamp"] + pd.Timedelta(minutes=30)).all()
    split = int(len(forecast) * 0.8)
    assert len(predictions) == len(forecast) - split
    test_rows = forecast.iloc[split:].reset_index(drop=True)
    assert predictions["timestamp"].equals(test_rows["timestamp"])
    assert predictions["target_timestamp"].equals(test_rows["target_timestamp"])
    assert np.allclose(predictions["actual"], test_rows["target"], rtol=0, atol=1e-9)
    assert np.allclose(predictions["baseline"], test_rows["total_demand_gbps"], rtol=0, atol=1e-9)
    assert np.isfinite(predictions[["actual", "predicted", "baseline"]].to_numpy()).all()
    assert (predictions["predicted"] >= 0).all()

    train_end = split - 2
    last_train_target_time = forecast.iloc[train_end - 1]["target_timestamp"]
    first_test_input_time = forecast.iloc[split]["timestamp"]
    assert last_train_target_time < first_test_input_time, "Train/test target boundary overlaps."

    # Forecast-adjusted traffic must preserve origin-time tower shares and sum to the prediction.
    predictions_by_target = predictions.set_index("target_timestamp")["predicted"]
    for target_time, group in adjusted.groupby("timestamp"):
        assert group["tower"].nunique() == 8
        assert group["tower"].is_unique
        expected_total = float(predictions_by_target.loc[target_time])
        assert np.isclose(group["demand_gbps"].sum(), expected_total, atol=1e-8)
        origin = group["forecast_origin_timestamp"].iloc[0]
        origin_rows = traffic[traffic["timestamp"] == origin]
        origin_totals = origin_rows.groupby("tower")["demand_gbps"].sum()
        expected_shares = origin_totals / origin_totals.sum()
        actual_shares = group.set_index("tower")["tower_share_at_origin"]
        for tower in expected_shares.index:
            assert np.isclose(actual_shares.loc[tower], expected_shares.loc[tower], atol=1e-10)

    # Each simulation must report 22 links for every timestamp with unique undirected edge IDs.
    for label, frame in [
        ("full baseline", full_baseline),
        ("test baseline", test_baseline),
        ("optimized", optimized),
    ]:
        counts = frame.groupby("timestamp").size()
        assert (counts == 22).all(), f"{label} does not contain 22 links per timestamp."
        edge_ids = frame.apply(
            lambda row: tuple(sorted((row["source"], row["destination"]))), axis=1
        )
        if frame.assign(_edge=edge_ids).duplicated(["timestamp", "_edge"]).any():
            raise AssertionError(f"Duplicate undirected link records in {label}.")
        assert np.isfinite(frame["utilization"]).all()
        assert (frame["capacity_gbps"] > 0).all()
        active = as_bool(frame["active"])
        expected_congested = active & (frame["utilization"] > 80.0)
        actual_congested = as_bool(frame["congested"])
        assert np.array_equal(expected_congested.to_numpy(), actual_congested.to_numpy())

    # Comparison files must cover the same test timestamps.
    assert set(test_baseline["timestamp"]) == set(optimized["timestamp"])
    for ts, rows in adjusted.groupby("timestamp"):
        assert np.isclose(rows["demand_gbps"].sum(), predictions_by_target.loc[ts], atol=1e-8)

    # Full baseline outage: every fiber link is inactive, with zero load.
    outage_start = pd.Timestamp("2026-10-20 18:00:00")
    outage_end = pd.Timestamp("2026-10-20 22:00:00")
    outage = full_baseline[
        (full_baseline["timestamp"] >= outage_start)
        & (full_baseline["timestamp"] < outage_end)
    ].copy()
    assert outage["timestamp"].nunique() == 16
    fiber = outage[outage["link_type"] == "fiber"]
    assert (fiber.groupby("timestamp").size() == 10).all()
    assert not as_bool(fiber["active"]).any()
    assert (fiber["traffic_gbps"] == 0).all()

    # Both outage topologies must remain connected to CORE.
    nominal_graph = create_network()
    outage_graph = nx.Graph()
    for u, v, attrs in nominal_graph.edges(data=True):
        if attrs["link_type"] != "fiber":
            outage_graph.add_edge(u, v)
    assert nx.is_connected(outage_graph), "Baseline topology is disconnected after fiber links are removed."

    backup_graph = nx.Graph()
    backup_pairs = {frozenset(("H1", "CORE")), frozenset(("H2", "CORE"))}
    for u, v, attrs in nominal_graph.edges(data=True):
        if attrs["link_type"] != "fiber" or frozenset((u, v)) in backup_pairs:
            backup_graph.add_edge(u, v)
    assert nx.is_connected(backup_graph), "Optimized topology is disconnected during backup failover."

    # Every routed traffic interval should be connected; no demand is silently dropped.
    for label, frame in [("full baseline", full_baseline), ("test baseline", test_baseline), ("optimized", optimized)]:
        if "unrouted_demand_gbps" in frame.columns:
            unrouted_per_interval = frame.groupby("timestamp")["unrouted_demand_gbps"].first()
            assert np.allclose(unrouted_per_interval.to_numpy(), 0.0, atol=1e-10), (
                f"{label} contains unrouted demand."
            )

    # Forecast-period optimized outage: both hypothetical backup hub-core links exist.
    optimized_outage = optimized[
        (optimized["timestamp"] >= outage_start)
        & (optimized["timestamp"] < outage_end)
    ].copy()
    assert optimized_outage["timestamp"].nunique() == 16
    backups = optimized_outage[optimized_outage["link_type"] == "backup_microwave"]
    assert backups["timestamp"].nunique() == 16
    assert (backups.groupby("timestamp").size() == 2).all()
    assert as_bool(backups["active"]).all()
    assert np.allclose(backups["capacity_gbps"], 3.0)

    # The topology remains connected after the configured fiber outage.
    assert not as_bool(optimized_outage.loc[optimized_outage["link_type"] == "fiber", "active"]).any()

    print("\nALL PROJECT VALIDATION CHECKS PASSED")
    print("Raw traffic rows:", len(traffic))
    print("Forecast rows:", len(forecast))
    print("Forecast test predictions:", len(predictions))
    print("Evaluation timestamps:", adjusted["timestamp"].nunique())
    print("Network links per timestamp: 22")
    print("Fiber outage intervals: 16")
    print("Disabled fiber links per outage interval: 10")
    print("Active backup links per optimized outage interval: 2")


if __name__ == "__main__":
    main()
