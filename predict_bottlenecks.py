"""Link-level bottleneck prediction: forecast-driven (30 min ahead) vs what happened.
A link is a bottleneck when it is active and utilization > 80%.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
THR = 80.0


def load(name):
    df = pd.read_csv(PROCESSED / name, parse_dates=["timestamp"])
    df["active"] = df["active"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    df["link"] = df.apply(lambda r: "-".join(sorted((str(r["source"]), str(r["destination"])))), axis=1)
    return df


def main():
    pred = load("test_baseline_links.csv")
    actual = load("baseline_links.csv")
    actual = actual[actual["timestamp"].isin(pred["timestamp"].unique())]

    m = pred[["timestamp", "link", "utilization", "active"]].merge(
        actual[["timestamp", "link", "utilization", "active"]],
        on=["timestamp", "link"], suffixes=("_pred", "_act"))
    m["pred_flag"] = m["active_pred"] & (m["utilization_pred"] > THR)
    m["act_flag"] = m["active_act"] & (m["utilization_act"] > THR)
    m = m.sort_values(["link", "timestamp"])

    tp = int((m.pred_flag & m.act_flag).sum())
    fp = int((m.pred_flag & ~m.act_flag).sum())
    fn = int((~m.pred_flag & m.act_flag).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if tp else float("nan")

    # congestion episodes = consecutive actual-congested intervals on one link
    eps = []
    for link, g in m.groupby("link"):
        a, p, ts = g["act_flag"].to_numpy(), g["pred_flag"].to_numpy(), g["timestamp"].to_numpy()
        i = 0
        while i < len(a):
            if a[i]:
                j = i
                while j + 1 < len(a) and a[j + 1]:
                    j += 1
                eps.append({"link": link, "start": str(ts[i]), "end": str(ts[j]),
                            "flagged_at_start": bool(p[i]), "flagged_any": bool(p[i:j + 1].any())})
                i = j + 1
            else:
                i += 1
    ep = pd.DataFrame(eps)
    n_ep = len(ep)
    ep_any = float(ep["flagged_any"].mean() * 100) if n_ep else float("nan")
    ep_start = float(ep["flagged_at_start"].mean() * 100) if n_ep else float("nan")

    report = {
        "threshold_percent": THR, "forecast_lead_time_minutes": 30,
        "link_interval_records": int(len(m)),
        "actual_bottleneck_records": int(m.act_flag.sum()),
        "predicted_bottleneck_records": int(m.pred_flag.sum()),
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "actual_congestion_episodes": n_ep,
        "episodes_flagged_at_least_once_percent": ep_any,
        "episodes_flagged_at_onset_percent": ep_start,
    }
    (PROCESSED / "bottleneck_prediction_metrics.json").write_text(json.dumps(report, indent=2))
    if n_ep:
        ep.to_csv(PROCESSED / "bottleneck_episodes.csv", index=False)

    print("BOTTLENECK PREDICTION (30 min ahead, threshold 80%)")
    for k, v in report.items():
        print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")

    # chart for the link with the highest actual utilisation
    if m["act_flag"].any():
        worst = m.loc[m["act_flag"]].groupby("link")["utilization_act"].max().idxmax()
        g = m[m["link"] == worst]
        RESULTS.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 5.5))
        ax.plot(g["timestamp"], g["utilization_act"].clip(upper=200), label="Actual utilization")
        ax.plot(g["timestamp"], g["utilization_pred"].clip(upper=200), label="Predicted (30 min ahead)")
        ax.axhline(THR, linestyle="--", label="Congestion threshold (80%)")
        ax.set_title(f"Bottleneck prediction on link {worst}")
        ax.set_ylabel("Utilization (%, clipped at 200)")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS / "bottleneck_prediction.png", dpi=300)
        plt.close(fig)
        print("Saved: results/bottleneck_prediction.png")


if __name__ == "__main__":
    main()