"""Dashboard for the synthetic Airtel Sync network study."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import networkx as nx
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.network import create_network


st.set_page_config(
    page_title="Airwave Architects | Network Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main { background-color: #f7f9fc; }
        h1, h2, h3 { color: #102a43; }
        div[data-testid="stMetric"] {
            background-color: white;
            padding: 16px;
            border-radius: 10px;
            border: 1px solid #e2e8f0;
        }
        div[data-testid="stMetricLabel"] { color: #52667a; }
        .block-container { padding-top: 1.6rem; }
        footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


def parse_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


@st.cache_data(show_spinner="Loading and validating project data...")
def load_data():
    paths = {
        "traffic": ROOT / "data/raw/traffic.csv",
        "baseline": ROOT / "data/processed/test_baseline_links.csv",
        "optimized": ROOT / "data/processed/optimized_links.csv",
        "predictions": ROOT / "data/processed/forecast_predictions.csv",
        "forecast": ROOT / "data/processed/forecast_dataset.csv",
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Required {name} file not found: {path}")

    traffic = pd.read_csv(paths["traffic"])
    baseline = pd.read_csv(paths["baseline"])
    optimized = pd.read_csv(paths["optimized"])
    predictions = pd.read_csv(paths["predictions"])
    forecast = pd.read_csv(paths["forecast"])

    required = {
        "traffic": {"timestamp", "tower", "demand_gbps", "event", "fiber_cut"},
        "baseline": {
            "timestamp", "source", "destination", "link_type", "capacity_gbps",
            "traffic_gbps", "utilization", "congested", "active", "fiber_cut",
        },
        "optimized": {
            "timestamp", "source", "destination", "link_type", "capacity_gbps",
            "traffic_gbps", "utilization", "congested", "active", "fiber_cut",
        },
        "predictions": {"timestamp", "target_timestamp", "actual", "predicted", "baseline"},
        "forecast": {"timestamp", "target_timestamp", "total_demand_gbps", "target"},
    }
    frames = {
        "traffic": traffic,
        "baseline": baseline,
        "optimized": optimized,
        "predictions": predictions,
        "forecast": forecast,
    }
    for name, columns in required.items():
        missing = columns - set(frames[name].columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
        if frames[name].empty:
            raise ValueError(f"{name} dataset is empty")

    for name in ["traffic", "baseline", "optimized", "predictions", "forecast"]:
        for column in (["timestamp", "target_timestamp"] if name in {"predictions", "forecast"}
                       else ["timestamp"]):
            frames[name][column] = pd.to_datetime(frames[name][column], errors="coerce")
            if frames[name][column].isna().any():
                raise ValueError(f"{name}.{column} contains invalid timestamps")

    traffic["fiber_cut"] = parse_bool_series(traffic["fiber_cut"])
    for frame in [baseline, optimized]:
        frame["active"] = parse_bool_series(frame["active"])
        frame["congested"] = parse_bool_series(frame["congested"])
        frame["fiber_cut"] = parse_bool_series(frame["fiber_cut"])

    for column in ["demand_gbps"]:
        traffic[column] = pd.to_numeric(traffic[column], errors="raise")
    for frame in [baseline, optimized]:
        for column in ["capacity_gbps", "traffic_gbps", "utilization"]:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
    for column in ["actual", "predicted", "baseline"]:
        predictions[column] = pd.to_numeric(predictions[column], errors="raise")
    for column in ["total_demand_gbps", "target"]:
        forecast[column] = pd.to_numeric(forecast[column], errors="raise")

    if not (predictions["target_timestamp"] == predictions["timestamp"] + pd.Timedelta(minutes=30)).all():
        raise ValueError("Prediction target timestamps are not aligned to a 30-minute horizon")

    # Confirm predictions correspond to the saved chronological test rows.
    split = int(len(forecast) * 0.8)
    test_rows = forecast.iloc[split:].reset_index(drop=True)
    predictions = predictions.reset_index(drop=True)
    if len(test_rows) != len(predictions):
        raise ValueError("Prediction row count does not match the chronological test set")
    if not predictions["timestamp"].equals(test_rows["timestamp"]):
        raise ValueError("Prediction input timestamps do not match the forecast test set")
    if not predictions["target_timestamp"].equals(test_rows["target_timestamp"]):
        raise ValueError("Prediction target timestamps do not match the forecast test set")
    if not np.allclose(predictions["actual"], test_rows["target"], atol=1e-9, rtol=0):
        raise ValueError("Saved prediction actual values do not match forecast test targets")
    if not np.allclose(predictions["baseline"], test_rows["total_demand_gbps"], atol=1e-9, rtol=0):
        raise ValueError("Saved persistence baseline does not match input-time aggregate demand")

    if set(baseline["timestamp"].unique()) != set(optimized["timestamp"].unique()):
        raise ValueError("Baseline and optimized results have different timestamp coverage")

    return traffic, baseline, optimized, predictions, forecast


def build_forecast_metrics(predictions: pd.DataFrame):
    actual = predictions["actual"]
    predicted = predictions["predicted"]
    persistence = predictions["baseline"]
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    persistence_mae = mean_absolute_error(actual, persistence)
    improvement = (persistence_mae - mae) / persistence_mae * 100 if persistence_mae > 0 else 0.0
    comparison = predictions[["target_timestamp", "actual", "predicted", "baseline"]].copy()
    comparison["error"] = comparison["predicted"] - comparison["actual"]
    return comparison, mae, rmse, r2, persistence_mae, improvement


def active_link_summary(frame: pd.DataFrame):
    active = frame[frame["active"]].copy()
    if active.empty:
        raise ValueError("No active link records found in evaluation results")
    return {
        "active": active,
        "congested_count": int(active["congested"].sum()),
        "congestion_rate": float(active["congested"].mean() * 100),
        "peak_utilization": float(active["utilization"].max()),
        "average_utilization": float(active["utilization"].mean()),
    }


if st.sidebar.button("🔄 Refresh project data"):
    load_data.clear()
    st.rerun()

try:
    traffic, baseline, optimized, predictions, forecast = load_data()
    forecast_comparison, model_mae, model_rmse, model_r2, persistence_mae, mae_improvement = (
        build_forecast_metrics(predictions)
    )
    baseline_summary = active_link_summary(baseline)
    optimized_summary = active_link_summary(optimized)
except Exception as error:
    st.error("Project data could not be loaded or validated.")
    st.exception(error)
    st.stop()


st.title("📡 AI-Based Backhaul Bottleneck Predictor")
st.markdown("**AIRWAVE ARCHITECTS** | Airtel Sync 2026 | Network Track")
st.caption("Predictive Traffic Forecasting • Dynamic Routing • Network Resilience")
st.markdown("---")

page = st.sidebar.radio(
    "DASHBOARD NAVIGATION",
    [
        "Network Overview",
        "Traffic Analysis",
        "AI Forecasting",
        "Bottleneck Prediction",
        "Network Optimization",
        "Outage Resilience",
    ],
)


if page == "Network Overview":
    st.header("Network Overview")
    graph = create_network()
    towers = [node for node, attrs in graph.nodes(data=True) if attrs.get("node_type") == "tower"]
    hubs = [node for node, attrs in graph.nodes(data=True) if attrs.get("node_type") == "hub"]
    duration = (traffic["timestamp"].max().date() - traffic["timestamp"].min().date()).days + 1

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Network Nodes", graph.number_of_nodes())
    col2.metric("Network Links", graph.number_of_edges())
    col3.metric("Access Towers", len(towers))
    col4.metric("Simulation Period", f"{duration} days")

    st.subheader("Simulated Backhaul Network Topology")
    positions = {
        "T1": (0.0, 5.0), "T2": (0.0, 4.0), "T3": (0.0, 3.0), "T4": (0.0, 2.0),
        "T5": (4.0, 5.0), "T6": (4.0, 4.0), "T7": (4.0, 3.0), "T8": (4.0, 2.0),
        "H1": (1.5, 4.0), "H2": (2.5, 3.0), "CORE": (6.0, 3.5), "SAT": (4.3, 0.5),
    }
    edge_colors = {"fiber": "#2563eb", "microwave": "#f59e0b", "satellite": "#8b5cf6"}
    fig = go.Figure()
    for link_type, color in edge_colors.items():
        edge_x, edge_y, hover = [], [], []
        for u, v, data in graph.edges(data=True):
            if data["link_type"] != link_type:
                continue
            x0, y0 = positions[u]
            x1, y1 = positions[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            hover.append(
                f"{u} ↔ {v}<br>Type: {link_type}<br>Capacity: {data['capacity']} Gbps<br>"
                f"Latency: {data['latency']} ms<br>Cost: {data['cost']}"
            )
        fig.add_trace(
            go.Scatter(
                x=edge_x, y=edge_y, mode="lines", name=link_type.title(),
                line=dict(width=2, color=color), hoverinfo="skip",
            )
        )

    node_groups = {
        "Access tower": towers,
        "Aggregation hub": hubs,
        "Core": ["CORE"],
        "Satellite": ["SAT"],
    }
    node_colors = {
        "Access tower": "#2563eb",
        "Aggregation hub": "#f59e0b",
        "Core": "#059669",
        "Satellite": "#8b5cf6",
    }
    for label, nodes in node_groups.items():
        fig.add_trace(
            go.Scatter(
                x=[positions[node][0] for node in nodes],
                y=[positions[node][1] for node in nodes],
                text=nodes,
                mode="markers+text",
                name=label,
                textposition="top center",
                marker=dict(size=23, color=node_colors[label], line=dict(width=2, color="white")),
                hovertemplate="<b>%{text}</b><extra></extra>",
            )
        )

    fig.update_layout(
        height=560,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(visible=False, range=[-0.8, 7.0]),
        yaxis=dict(visible=False, range=[-0.2, 5.8]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=15, r=15, t=35, b=15),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        "Illustrative 12-node, 22-link topology. Link capacities, latency and cost are "
        "synthetic simulation assumptions, not measured Airtel infrastructure parameters."
    )

    st.subheader("Simulation Configuration")
    configuration = pd.DataFrame(
        {
            "Parameter": ["Duration", "Sampling interval", "Access towers", "Aggregation hubs", "Network nodes", "Network links", "Traffic records"],
            "Value": [f"{duration} days", "15 minutes", len(towers), len(hubs), graph.number_of_nodes(), graph.number_of_edges(), f"{len(traffic):,}"],
        }
    )
    st.dataframe(configuration, hide_index=True, use_container_width=True)


elif page == "Traffic Analysis":
    st.header("Network Traffic Analysis")
    network_traffic = (
        traffic.groupby("timestamp")
        .agg(
            demand_gbps=("demand_gbps", "sum"),
            event=("event", "first"),
            fiber_cut=("fiber_cut", "max"),
        )
        .reset_index()
    )
    peak = network_traffic.loc[network_traffic["demand_gbps"].idxmax()]
    festival_intervals = int((network_traffic["event"] == "festival").sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Traffic Records", f"{len(traffic):,}")
    col2.metric("Average Aggregate Demand", f"{network_traffic['demand_gbps'].mean():.2f} Gbps")
    col3.metric("Peak Aggregate Demand", f"{peak['demand_gbps']:.2f} Gbps")
    col4.metric("Festival Intervals", festival_intervals)

    st.subheader("Aggregate Network Traffic")
    fig = px.line(
        network_traffic,
        x="timestamp",
        y="demand_gbps",
        labels={"timestamp": "Time", "demand_gbps": "Aggregate demand (Gbps)"},
    )
    fig.update_layout(height=420, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Average Tower-wise Demand")
    tower_demand = traffic.groupby("tower", as_index=False)["demand_gbps"].mean()
    fig = px.bar(
        tower_demand,
        x="tower",
        y="demand_gbps",
        color="tower",
        labels={"tower": "Access tower", "demand_gbps": "Average demand (Gbps)"},
    )
    fig.update_layout(height=360, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Traffic Event Distribution")
    event_counts = network_traffic["event"].value_counts().rename_axis("Event").reset_index(name="15-minute intervals")
    fig = px.pie(event_counts, names="Event", values="15-minute intervals", hole=0.42)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Event counts use unique network timestamps, not eight separate tower rows per interval.")


elif page == "AI Forecasting":
    st.header("AI-Based Traffic Forecasting")
    st.markdown("LightGBM regression model forecasting aggregate network demand 30 minutes ahead.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", f"{model_mae:.4f} Gbps")
    col2.metric("RMSE", f"{model_rmse:.4f} Gbps")
    col3.metric("R²", f"{model_r2:.4f}")
    col4.metric("Persistence MAE", f"{persistence_mae:.4f} Gbps")
    st.success(f"MAE improvement over persistence baseline: {mae_improvement:.2f}% on the chronological test set.")

    st.subheader("Actual vs Predicted Aggregate Demand")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_comparison["target_timestamp"], y=forecast_comparison["actual"], name="Actual", mode="lines"))
    fig.add_trace(go.Scatter(x=forecast_comparison["target_timestamp"], y=forecast_comparison["predicted"], name="LightGBM prediction", mode="lines", line=dict(dash="dot")))
    fig.update_layout(height=440, hovermode="x unified", xaxis_title="Target timestamp", yaxis_title="Aggregate demand (Gbps)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Prediction Error Distribution")
    fig = px.histogram(
        forecast_comparison,
        x="error",
        nbins=30,
        labels={"error": "Prediction minus actual (Gbps)"},
    )
    fig.update_layout(height=330)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Predictions are matched using saved input and target timestamps. "
        "The final two training rows are purged to prevent the 30-minute target "
        "from crossing the chronological train/test boundary."
    )


elif page == "Bottleneck Prediction":
    st.header("Link-Level Bottleneck Prediction")
    st.markdown(
        "Forecast traffic is routed over the network 30 minutes ahead. Links predicted above "
        "80% utilization are flagged, then checked against what actually happened."
    )

    metrics_path = ROOT / "data/processed/bottleneck_prediction_metrics.json"
    episodes_path = ROOT / "data/processed/bottleneck_episodes.csv"
    actual_path = ROOT / "data/processed/baseline_links.csv"
    if not metrics_path.exists() or not actual_path.exists():
        st.warning("Bottleneck results not found. Run `python main.py` first.")
        st.stop()

    bm = json.loads(metrics_path.read_text(encoding="utf-8"))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precision", f"{bm['precision'] * 100:.1f}%")
    col2.metric("Recall", f"{bm['recall'] * 100:.1f}%")
    col3.metric("Congestion Episodes", bm["actual_congestion_episodes"])
    col4.metric("Flagged at Onset", f"{bm['episodes_flagged_at_onset_percent']:.0f}%")
    st.caption(
        f"{bm['true_positives']} correct flags, {bm['false_positives']} false alarms, "
        f"{bm['false_negatives']} missed, out of {bm['actual_bottleneck_records']:,} actual "
        f"bottleneck records. {bm['episodes_flagged_at_least_once_percent']:.0f}% of episodes "
        "were flagged at least once."
    )

    actual_all = pd.read_csv(actual_path, parse_dates=["timestamp"])
    actual_all["active"] = parse_bool_series(actual_all["active"])
    actual_all = actual_all[actual_all["timestamp"].isin(baseline["timestamp"].unique())]

    def link_name(frame):
        return frame["source"].astype(str).combine(
            frame["destination"].astype(str), lambda a, b: " - ".join(sorted((a, b)))
        )

    pred_view = baseline[["timestamp", "source", "destination", "utilization", "active"]].copy()
    pred_view["link"] = link_name(pred_view)
    act_view = actual_all[["timestamp", "source", "destination", "utilization", "active"]].copy()
    act_view["link"] = link_name(act_view)

    hot_links = (
        act_view[act_view["active"] & (act_view["utilization"] > 80)]
        .groupby("link").size().sort_values(ascending=False).index.tolist()
    )
    choices = hot_links + [l for l in sorted(act_view["link"].unique()) if l not in hot_links]
    chosen = st.selectbox("Link", choices)

    p = pred_view[pred_view["link"] == chosen].sort_values("timestamp")
    a = act_view[act_view["link"] == chosen].sort_values("timestamp")
    p = p.assign(utilization=p["utilization"].where(p["active"]))
    a = a.assign(utilization=a["utilization"].where(a["active"]))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a["timestamp"], y=a["utilization"].clip(upper=200), name="Actual", mode="lines"))
    fig.add_trace(go.Scatter(x=p["timestamp"], y=p["utilization"].clip(upper=200), name="Predicted (30 min ahead)", mode="lines", line=dict(dash="dot")))
    fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="80% threshold")
    fig.update_layout(
        height=430, hovermode="x unified", xaxis_title="Time",
        yaxis_title="Utilization (%, clipped at 200)",
    )
    st.plotly_chart(fig, use_container_width=True)

    if episodes_path.exists():
        st.subheader("Congestion Episodes")
        episodes = pd.read_csv(episodes_path)
        episodes = episodes.rename(columns={
            "link": "Link", "start": "Start", "end": "End",
            "flagged_at_start": "Flagged at onset", "flagged_any": "Flagged at least once",
        })
        st.dataframe(episodes, hide_index=True, use_container_width=True)

    st.info(
        "The 30-minute lead time comes from how the test is built: forecast traffic for each "
        "time is derived from information available 30 minutes earlier. The fiber outage and "
        "the festival schedule are treated as known. The model forecasts aggregate demand; "
        "link bottlenecks come from routing that forecast. The test window contains only a few "
        "congestion episodes, so these scores are indicative."
    )


elif page == "Network Optimization":
    st.header("Dynamic Network Optimization")
    base = baseline_summary
    opt = optimized_summary
    congestion_reduction = (
        (base["congestion_rate"] - opt["congestion_rate"]) / base["congestion_rate"] * 100
        if base["congestion_rate"] > 0 else 0.0
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Baseline Congested Records", f"{base['congested_count']:,}")
    col2.metric("Optimized Congested Records", f"{opt['congested_count']:,}")
    col3.metric("Congestion-rate Reduction", f"{congestion_reduction:.1f}%")
    col4.metric("Optimized Peak Utilization", f"{opt['peak_utilization']:.2f}%")

    comparison = pd.DataFrame(
        {
            "Metric": ["Active link records", "Congested active records", "Congestion rate (%)", "Peak active utilization (%)", "Mean active utilization (%)"],
            "Baseline routing": [len(base["active"]), base["congested_count"], base["congestion_rate"], base["peak_utilization"], base["average_utilization"]],
            "Dynamic optimization": [len(opt["active"]), opt["congested_count"], opt["congestion_rate"], opt["peak_utilization"], opt["average_utilization"]],
        }
    )
    st.subheader("Baseline vs Optimized Performance")
    st.dataframe(comparison.round(3), hide_index=True, use_container_width=True)

    congestion_data = pd.DataFrame(
        {
            "Strategy": ["Baseline routing", "Dynamic optimization"],
            "Congestion rate (%)": [base["congestion_rate"], opt["congestion_rate"]],
            "Congested records": [base["congested_count"], opt["congested_count"]],
        }
    )
    fig = px.bar(congestion_data, x="Strategy", y="Congestion rate (%)", color="Strategy", text="Congested records")
    fig.update_layout(height=360, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    base_max = base["active"].groupby("timestamp")["utilization"].max().rename("Baseline")
    opt_max = opt["active"].groupby("timestamp")["utilization"].max().rename("Dynamic optimization")
    utilization = pd.concat([base_max, opt_max], axis=1).dropna().reset_index()
    fig = px.line(
        utilization,
        x="timestamp",
        y=["Baseline", "Dynamic optimization"],
        labels={"timestamp": "Time", "value": "Maximum active-link utilization (%)", "variable": "Strategy"},
    )
    fig.update_layout(height=420, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Both routing strategies are evaluated on identical forecast-adjusted traffic. "
        "The aggregate forecast is allocated using tower traffic shares observed at the "
        "forecast origin; individual tower demands are not separately predicted. "
        "The routing method is a greedy congestion-aware path-selection heuristic, "
        "not a global capacity-constrained optimizer."
    )


elif page == "Outage Resilience":
    st.header("Network Resilience During Simulated Fiber Outage")
    outage_rows = traffic[traffic["fiber_cut"]]
    if outage_rows.empty:
        st.warning("No fiber outage is configured in the traffic data.")
        st.stop()

    outage_start = outage_rows["timestamp"].min()
    outage_end = outage_rows["timestamp"].max() + pd.Timedelta(minutes=15)

    base_outage = baseline[
        (baseline["timestamp"] >= outage_start)
        & (baseline["timestamp"] < outage_end)
        & baseline["active"]
    ].copy()
    opt_outage = optimized[
        (optimized["timestamp"] >= outage_start)
        & (optimized["timestamp"] < outage_end)
        & optimized["active"]
    ].copy()

    if base_outage.empty or opt_outage.empty:
        st.warning("The outage is not present in the forecast evaluation window.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("Baseline Peak Utilization", f"{base_outage['utilization'].max():.2f}%")
    col2.metric("Optimized Peak Utilization", f"{opt_outage['utilization'].max():.2f}%")
    col3.metric("Optimized Congested Active Records", f"{int(opt_outage['congested'].sum()):,}")

    base_max = base_outage.groupby("timestamp")["utilization"].max().rename("Baseline")
    opt_max = opt_outage.groupby("timestamp")["utilization"].max().rename("Dynamic optimization")
    outage_comparison = pd.concat([base_max, opt_max], axis=1).dropna().reset_index()

    st.subheader("Maximum Active-Link Utilization During Outage")
    fig = px.line(
        outage_comparison,
        x="timestamp",
        y=["Baseline", "Dynamic optimization"],
        markers=True,
        labels={"timestamp": "Time", "value": "Maximum active-link utilization (%)", "variable": "Strategy"},
    )
    fig.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="80% congestion threshold")
    fig.update_layout(height=430, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Emergency Microwave Backup Links")
    backup = opt_outage[opt_outage["link_type"] == "backup_microwave"].copy()
    if backup.empty:
        st.warning("No active emergency microwave backup links were found in the outage evaluation.")
    else:
        backup["link"] = backup["source"].astype(str) + " ↔ " + backup["destination"].astype(str)
        summary = (
            backup.groupby("link")
            .agg(
                Peak_utilization_percent=("utilization", "max"),
                Mean_utilization_percent=("utilization", "mean"),
                Congested_records=("congested", "sum"),
                Capacity_Gbps=("capacity_gbps", "first"),
            )
            .reset_index()
        )
        fig = px.bar(
            summary,
            x="link",
            y="Peak_utilization_percent",
            color="link",
            text_auto=".2f",
            labels={"Peak_utilization_percent": "Peak utilization (%)"},
        )
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(summary.round(3), hide_index=True, use_container_width=True)

    st.warning(
        "Scenario assumption: all fiber links are disabled during the modeled outage. "
        "Two hypothetical 3 Gbps hub-to-core microwave backup links are activated in "
        "the optimized scenario. These are synthetic resilience assumptions, not "
        "measured Airtel infrastructure or proof of a deployed backup network."
    )


st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:#64748b;">
        <b>AIRWAVE ARCHITECTS</b><br>
        Airtel Sync 2026 | Network Track<br>
        AI-Based Predictive Backhaul Optimization
    </div>
    """,
    unsafe_allow_html=True,
)