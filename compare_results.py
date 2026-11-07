"""Evaluate baseline and dynamic routing on identical forecast-adjusted traffic."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.simulator import simulate_network

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"


def summarize(name: str, frame: pd.DataFrame) -> dict:
    active = frame[frame["active"]].copy()
    if active.empty:
        raise ValueError(f"{name} has no active link records.")

    congestion_count = int(active["congested"].sum())
    congestion_rate = float(active["congested"].mean() * 100.0)
    per_timestamp_unrouted = (
        frame.groupby("timestamp")["unrouted_demand_gbps"].first()
        if "unrouted_demand_gbps" in frame.columns
        else pd.Series(dtype=float)
    )

    return {
        "total_link_records": int(len(frame)),
        "active_link_records": int(len(active)),
        "congested_active_link_records": congestion_count,
        "congestion_rate_active_records_percent": congestion_rate,
        "peak_active_utilization_percent": float(active["utilization"].max()),
        "mean_active_utilization_percent": float(active["utilization"].mean()),
        "mean_unrouted_demand_gbps_per_interval": (
            float(per_timestamp_unrouted.mean()) if not per_timestamp_unrouted.empty else 0.0
        ),
        "max_unrouted_demand_gbps_per_interval": (
            float(per_timestamp_unrouted.max()) if not per_timestamp_unrouted.empty else 0.0
        ),
    }


def main():
    adjusted_path = PROCESSED / "forecast_adjusted_traffic.csv"
    optimized_path = PROCESSED / "optimized_links.csv"
    if not adjusted_path.exists() or not optimized_path.exists():
        raise FileNotFoundError(
            "Run the traffic generator, forecaster and optimizer before comparison."
        )

    evaluation_traffic = pd.read_csv(
        adjusted_path,
        parse_dates=["timestamp", "forecast_origin_timestamp"],
    )
    optimized = pd.read_csv(optimized_path, parse_dates=["timestamp"])

    # Recompute baseline routing over the exact same forecast-adjusted traffic.
    baseline = simulate_network(evaluation_traffic)

    if set(baseline["timestamp"].unique()) != set(optimized["timestamp"].unique()):
        raise AssertionError("Baseline and optimized outputs have different timestamp coverage.")

    expected_edges = 22
    baseline_counts = baseline.groupby("timestamp").size()
    optimized_counts = optimized.groupby("timestamp").size()
    if not (baseline_counts == expected_edges).all():
        raise AssertionError("Baseline must contain 22 link records per timestamp.")
    if not (optimized_counts == expected_edges).all():
        raise AssertionError("Optimized output must contain 22 link records per timestamp.")

    baseline.to_csv(PROCESSED / "test_baseline_links.csv", index=False)

    baseline_metrics = summarize("Baseline routing", baseline)
    optimized_metrics = summarize("Dynamic optimization", optimized)

    baseline_rate = baseline_metrics["congestion_rate_active_records_percent"]
    optimized_rate = optimized_metrics["congestion_rate_active_records_percent"]
    rate_reduction = (
        (baseline_rate - optimized_rate) / baseline_rate * 100.0
        if baseline_rate > 0 else 0.0
    )

    baseline_peak = baseline_metrics["peak_active_utilization_percent"]
    optimized_peak = optimized_metrics["peak_active_utilization_percent"]
    peak_reduction = (
        (baseline_peak - optimized_peak) / baseline_peak * 100.0
        if baseline_peak > 0 else 0.0
    )

    report = {
        "evaluation_note": (
            "Both routing strategies were evaluated on the same aggregate forecast, "
            "distributed using tower traffic shares observed at the forecast origin."
        ),
        "baseline": baseline_metrics,
        "dynamic_optimization": optimized_metrics,
        "congestion_rate_reduction_percent": float(rate_reduction),
        "peak_utilization_reduction_percent": float(peak_reduction),
    }

    with open(PROCESSED / "comparison_metrics.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    for name, metrics in [
        ("BASELINE ROUTING", baseline_metrics),
        ("DYNAMIC OPTIMIZATION", optimized_metrics),
    ]:
        print(f"\n{name}\n" + "-" * 48)
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

    print("\nRELATIVE IMPROVEMENT")
    print("--------------------")
    print(f"Congestion-rate reduction: {rate_reduction:.2f}%")
    print(f"Peak-utilization reduction: {peak_reduction:.2f}%")
    print("\nSaved: data/processed/test_baseline_links.csv")
    print("Saved: data/processed/comparison_metrics.json")


if __name__ == "__main__":
    main()
