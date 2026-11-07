"""Validate the simulated all-fiber outage in baseline results."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "data" / "processed" / "baseline_links.csv"
START = pd.Timestamp("2026-10-20 18:00:00")
END = pd.Timestamp("2026-10-20 22:00:00")


def main():
    df = pd.read_csv(RESULTS, parse_dates=["timestamp"])
    df["active"] = df["active"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})

    outage = df[(df["timestamp"] >= START) & (df["timestamp"] < END)].copy()
    if outage.empty:
        raise AssertionError("No records found in the configured outage window.")

    timestamps = sorted(outage["timestamp"].unique())
    assert len(timestamps) == 16, f"Expected 16 15-minute intervals, found {len(timestamps)}."

    fiber = outage[outage["link_type"] == "fiber"].copy()
    if fiber.empty:
        raise AssertionError("No fiber-link records found in the outage window.")

    fiber_counts = fiber.groupby("timestamp").size()
    assert (fiber_counts == 10).all(), (
        "Expected 10 fiber links at every outage interval. "
        f"Observed counts: {fiber_counts.to_dict()}"
    )
    assert not fiber["active"].any(), "One or more fiber links remain active during the outage."
    assert (fiber["traffic_gbps"] == 0).all(), "Inactive fiber links carry non-zero traffic."
    assert outage["fiber_cut"].all(), "The outage flag is not set for every interval."

    print("FIBER OUTAGE VALIDATION PASSED")
    print("Outage window:", START, "to", END)
    print("Intervals:", len(timestamps))
    print("Fiber links disabled per interval:", 10)
    print("All 10 fiber links are inactive with zero load in every interval.")


if __name__ == "__main__":
    main()
