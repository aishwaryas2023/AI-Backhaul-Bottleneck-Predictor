"""Generate reproducible synthetic traffic for the demo."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

TOWER_BASES = {
    "T1": 0.28,
    "T2": 0.32,
    "T3": 0.30,
    "T4": 0.26,
    "T5": 0.34,
    "T6": 0.29,
    "T7": 0.31,
    "T8": 0.27,
}

FESTIVAL_DAYS = {12, 18}
FIBER_OUTAGE_START = pd.Timestamp("2026-10-20 18:00:00")
FIBER_OUTAGE_END = pd.Timestamp("2026-10-20 22:00:00")


def generate_traffic(days: int = 21, seed: int = 42) -> pd.DataFrame:
    if not isinstance(days, int) or days <= 0:
        raise ValueError("days must be a positive integer")

    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(
        start="2026-10-01 00:00:00",
        periods=days * 96,
        freq="15min",
    )

    rows = []
    for timestamp in timestamps:
        hour = timestamp.hour + timestamp.minute / 60.0
        weekend_factor = 0.90 if timestamp.dayofweek >= 5 else 1.0

        morning_peak = np.exp(-0.5 * ((hour - 9) / 1.7) ** 2)
        evening_peak = np.exp(-0.5 * ((hour - 20) / 2.0) ** 2)
        daily_factor = 0.72 + 0.85 * morning_peak + 1.35 * evening_peak

        festival = timestamp.day in FESTIVAL_DAYS and 17 <= hour < 22
        fiber_cut = FIBER_OUTAGE_START <= timestamp < FIBER_OUTAGE_END

        for tower, base in TOWER_BASES.items():
            demand = base * daily_factor * weekend_factor
            demand *= rng.normal(1.0, 0.055)

            # Synthetic demand surge at three towers during scheduled events.
            if festival and tower in {"T3", "T4", "T5"}:
                demand += 4.0

            demand = max(0.05, demand)
            rows.append(
                {
                    "timestamp": timestamp,
                    "tower": tower,
                    "demand_gbps": round(float(demand), 4),
                    "event": "festival" if festival else "normal",
                    "fiber_cut": bool(fiber_cut),
                }
            )

    traffic = pd.DataFrame(rows)

    expected_rows = days * 96 * len(TOWER_BASES)
    if len(traffic) != expected_rows:
        raise RuntimeError("Generated traffic row count is incorrect.")
    if traffic.duplicated(["timestamp", "tower"]).any():
        raise RuntimeError("Duplicate timestamp/tower records were generated.")
    if traffic["demand_gbps"].isna().any() or (traffic["demand_gbps"] < 0).any():
        raise RuntimeError("Invalid traffic demand was generated.")

    output = ROOT / "data" / "raw" / "traffic.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    traffic.to_csv(output, index=False)
    return traffic


if __name__ == "__main__":
    data = generate_traffic()
    print("\nTRAFFIC GENERATION SUCCESSFUL")
    print("Dataset shape:", data.shape)
    print("Unique timestamps:", data["timestamp"].nunique())
    print("Number of towers:", data["tower"].nunique())
    print("Festival intervals:", data.loc[data["event"] == "festival", "timestamp"].nunique())
    print("Fiber outage intervals:", data.loc[data["fiber_cut"], "timestamp"].nunique())
    print("Saved: data/raw/traffic.csv")
