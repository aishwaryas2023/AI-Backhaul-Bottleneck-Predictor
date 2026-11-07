"""Validate the CORE-SAT satellite path during the simulated fiber outage."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "data" / "processed" / "baseline_links.csv"
START = pd.Timestamp("2026-10-20 18:00:00")
END = pd.Timestamp("2026-10-20 22:00:00")


def main():
    df = pd.read_csv(RESULTS, parse_dates=["timestamp"])
    df["active"] = df["active"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})

    is_satellite_edge = df.apply(
        lambda row: frozenset((row["source"], row["destination"])) == frozenset(("CORE", "SAT")),
        axis=1,
    )
    satellite = df[
        is_satellite_edge
        & (df["timestamp"] >= START)
        & (df["timestamp"] < END)
    ].sort_values("timestamp").copy()

    if satellite.empty:
        raise AssertionError("No CORE-SAT link records found during the outage.")
    if len(satellite) != 16:
        raise AssertionError(f"Expected 16 satellite records, found {len(satellite)}.")
    if not satellite["active"].all():
        raise AssertionError("The satellite link must remain active during the fiber outage.")

    print("CORE-SAT LINK DURING FIBER OUTAGE")
    print(satellite[["timestamp", "source", "destination", "utilization", "active", "traffic_gbps"]].to_string(index=False))
    peak_utilization = float(satellite["utilization"].max())
    print("Peak satellite utilization:", round(peak_utilization, 2), "%")
    if peak_utilization <= 80.0:
        raise AssertionError("Expected the satellite link to exceed the 80% congestion threshold in this scenario.")

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(satellite["timestamp"], satellite["utilization"], marker="o", label="CORE-SAT utilization")
    ax.axhline(80, linestyle="--", label="Congestion threshold (80%)")
    ax.set_title("CORE-SAT Link Utilization During the Simulated Fiber Outage")
    ax.set_xlabel("Time")
    ax.set_ylabel("Utilization (%)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "satellite_outage_utilization.png", dpi=300)
    plt.close(fig)
    print("Saved: results/satellite_outage_utilization.png")


if __name__ == "__main__":
    main()
