"""Create baseline visualizations using active-link utilization only."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def make_link_name(row):
    return " - ".join(sorted((str(row["source"]), str(row["destination"]))))


def main():
    input_path = ROOT / "data" / "processed" / "baseline_links.csv"
    df = pd.read_csv(input_path, parse_dates=["timestamp"])
    required = {"timestamp", "source", "destination", "utilization", "active"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Baseline results missing columns: {sorted(missing)}")

    df["active"] = (
        df["active"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    )
    active = df[df["active"]].copy()
    if active.empty:
        raise ValueError("No active link records found in baseline results.")

    active["link"] = active.apply(make_link_name, axis=1)
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Average utilization is conditional on the link being active.
    avg_util = (
        active.groupby("link")["utilization"]
        .mean()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    avg_util.plot(kind="bar", ax=ax)
    ax.axhline(80, linestyle="--", label="Congestion threshold (80%)")
    ax.set_title("Average Utilization While Each Baseline Link Is Active")
    ax.set_ylabel("Utilization (%)")
    ax.set_xlabel("Network Link")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "link_utilization.png", dpi=300)
    plt.close(fig)

    busiest_link = avg_util.index[0]
    all_link_rows = df[df.apply(make_link_name, axis=1) == busiest_link].copy()
    all_link_rows = all_link_rows.sort_values("timestamp")
    all_link_rows.loc[~all_link_rows["active"], "utilization"] = float("nan")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(all_link_rows["timestamp"], all_link_rows["utilization"])
    ax.axhline(80, linestyle="--", label="Congestion threshold (80%)")
    ax.set_title(f"Utilization Over Time: {busiest_link}")
    ax.set_ylabel("Utilization (%)")
    ax.set_xlabel("Time")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "peak_link_utilization.png", dpi=300)
    plt.close(fig)

    print("Baseline visualizations saved successfully.")
    print("Busiest link while active:", busiest_link)


if __name__ == "__main__":
    main()
