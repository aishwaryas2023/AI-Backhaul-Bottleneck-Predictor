"""Generate fair, active-link-only baseline versus optimized charts."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"


def read_results(filename):
    frame = pd.read_csv(PROCESSED / filename, parse_dates=["timestamp"])
    for column in ["active", "congested", "fiber_cut"]:
        if column in frame.columns and not pd.api.types.is_bool_dtype(frame[column]):
            frame[column] = (
                frame[column].astype(str).str.strip().str.lower()
                .isin({"true", "1", "yes"})
            )
    return frame


def main():
    baseline = read_results("test_baseline_links.csv")
    optimized = read_results("optimized_links.csv")
    RESULTS.mkdir(parents=True, exist_ok=True)

    baseline_active = baseline[baseline["active"]].copy()
    optimized_active = optimized[optimized["active"]].copy()
    if baseline_active.empty or optimized_active.empty:
        raise ValueError("Cannot compare results without active link records.")

    baseline_rate = baseline_active["congested"].mean() * 100.0
    optimized_rate = optimized_active["congested"].mean() * 100.0
    baseline_count = int(baseline_active["congested"].sum())
    optimized_count = int(optimized_active["congested"].sum())

    # 1. Congestion-rate comparison (the denominators are active link records).
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(
        ["Baseline Routing", "Dynamic Optimization"],
        [baseline_rate, optimized_rate],
    )
    ax.set_ylabel("Congested Active Link Records (%)")
    ax.set_title("Congestion Rate: Baseline vs Dynamic Optimization")
    for bar, rate, count in zip(bars, [baseline_rate, optimized_rate], [baseline_count, optimized_count]):
        ax.annotate(
            f"{rate:.3f}%\n({count:,} records)",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            xytext=(0, 5),
            textcoords="offset points",
        )
    fig.tight_layout()
    fig.savefig(RESULTS / "congestion_comparison.png", dpi=300)
    plt.close(fig)

    # 2. Maximum active-link utilization by timestamp.
    baseline_max = baseline_active.groupby("timestamp")["utilization"].max().rename("Baseline")
    optimized_max = optimized_active.groupby("timestamp")["utilization"].max().rename("Dynamic Optimization")
    utilization = pd.concat([baseline_max, optimized_max], axis=1).dropna()
    if utilization.empty:
        raise ValueError("No common timestamps available for utilization comparison.")

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(utilization.index, utilization["Baseline"], label="Baseline Routing")
    ax.plot(utilization.index, utilization["Dynamic Optimization"], label="Dynamic Optimization")
    ax.axhline(80, linestyle="--", label="Congestion threshold (80%)")
    ax.set_title("Maximum Active-Link Utilization Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Maximum Utilization (%)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS / "utilization_comparison.png", dpi=300)
    plt.close(fig)

    # 3. Outage-only comparison. Use the outage flag and exclude inactive links.
    if "fiber_cut" not in baseline.columns or "fiber_cut" not in optimized.columns:
        raise ValueError("fiber_cut is missing from comparison output.")

    base_outage = baseline[baseline["fiber_cut"] & baseline["active"]]
    opt_outage = optimized[optimized["fiber_cut"] & optimized["active"]]
    base_outage_max = base_outage.groupby("timestamp")["utilization"].max().rename("Baseline")
    opt_outage_max = opt_outage.groupby("timestamp")["utilization"].max().rename("Dynamic Optimization")
    outage = pd.concat([base_outage_max, opt_outage_max], axis=1).dropna()

    if outage.empty:
        print("No fiber-outage intervals occur in the forecast evaluation window.")
    else:
        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.plot(outage.index, outage["Baseline"], marker="o", label="Baseline Routing")
        ax.plot(outage.index, outage["Dynamic Optimization"], marker="o", label="Dynamic Optimization")
        ax.axhline(80, linestyle="--", label="Congestion threshold (80%)")
        ax.set_title("Maximum Active-Link Utilization During Fiber Outage")
        ax.set_xlabel("Time")
        ax.set_ylabel("Maximum Utilization (%)")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(RESULTS / "outage_comparison.png", dpi=300)
        plt.close(fig)

    print("Comparison visualizations generated:")
    print("results/congestion_comparison.png")
    print("results/utilization_comparison.png")
    if not outage.empty:
        print("results/outage_comparison.png")


if __name__ == "__main__":
    main()
