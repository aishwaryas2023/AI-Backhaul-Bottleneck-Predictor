"""Run the complete Airwave Architects project pipeline."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PIPELINE = [
    ("Generate reproducible synthetic traffic", "module", "src.traffic_generator"),
    ("Run full baseline network simulation", "module", "src.simulator"),
    ("Prepare 30-minute forecasting dataset", "module", "src.prepare_dataset"),
    ("Train and evaluate LightGBM forecaster", "module", "src.train_forecaster"),
    ("Run forecast-adjusted dynamic routing", "module", "src.optimizer"),
    ("Compare routing on identical evaluation traffic", "script", "compare_results.py"),
    ("Predict link-level bottlenecks (30 min ahead)", "script", "predict_bottlenecks.py"),
    ("Generate full-baseline visualizations", "module", "src.visualize"),
    ("Generate comparison visualizations", "script", "compare_visualize.py"),
    ("Validate fiber outage scenario", "script", "check_outage.py"),
    ("Validate satellite bottleneck scenario", "script", "check_satellite_outage.py"),
    ("Run cross-file project integrity checks", "script", "validate_project.py"),
]


def run_pipeline() -> None:
    print("\n" + "=" * 64)
    print(" AIRWAVE ARCHITECTS | PREDICTIVE BACKHAUL OPTIMIZATION")
    print("=" * 64)

    for index, (description, kind, target) in enumerate(PIPELINE, start=1):
        print(f"\n[{index}/{len(PIPELINE)}] {description}", flush=True)
        print("-" * 64, flush=True)

        if kind == "module":
            command = [sys.executable, "-m", target]
        else:
            command = [sys.executable, str(ROOT / target)]

        try:
            subprocess.run(command, cwd=ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"\nPIPELINE FAILED: {description}")
            print("Exit code:", exc.returncode)
            raise SystemExit(exc.returncode) from exc

    print("\n" + "=" * 64)
    print(" PIPELINE COMPLETED AND VALIDATION PASSED")
    print("=" * 64)
    print("Generated outputs:")
    print("- data/raw/traffic.csv")
    print("- data/processed/forecast_dataset.csv")
    print("- data/processed/forecast_predictions.csv")
    print("- data/processed/forecast_adjusted_traffic.csv")
    print("- data/processed/baseline_links.csv")
    print("- data/processed/test_baseline_links.csv")
    print("- data/processed/optimized_links.csv")
    print("- data/processed/bottleneck_prediction_metrics.json")
    print("- data/processed/bottleneck_episodes.csv")
    print("- models/traffic_forecaster.pkl")
    print("- results/*.png (includes bottleneck_prediction.png)")


if __name__ == "__main__":
    run_pipeline()