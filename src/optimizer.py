"""Forecast-adjusted, congestion-aware greedy routing simulation.

The model forecasts aggregate demand. To construct tower-level test traffic,
this simulation applies the latest observed tower shares at the forecast origin
rather than reading actual future tower demand. The fiber-outage flag is treated
as an observed/announced network state at the target interval; it is not predicted
by the LightGBM model.
"""

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from src.network import create_network

ROOT = Path(__file__).resolve().parents[1]


def get_link_key(u, v):
    return tuple(sorted((u, v)))


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def activate_backup_links(graph: nx.Graph, fiber_cut: bool) -> nx.Graph:
    """Apply the hypothetical hub-to-core microwave failover scenario."""
    backup_pairs = {
        frozenset(("H1", "CORE")),
        frozenset(("H2", "CORE")),
    }

    for u, v, data in graph.edges(data=True):
        data["active"] = True

        if fiber_cut and data["link_type"] == "fiber":
            if frozenset((u, v)) in backup_pairs:
                # Synthetic emergency links replacing the two hub-core fiber links.
                data["link_type"] = "backup_microwave"
                data["capacity"] = 3.0
                data["latency"] = 18.0
                data["cost"] = 4.0
                data["active"] = True
            else:
                data["active"] = False

    return graph


def build_forecast_adjusted_traffic(
    traffic_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
) -> pd.DataFrame:
    """Allocate predicted aggregate demand using tower shares observed at origin.

    Actual future per-tower demand is deliberately not accessed. Only the
    target interval's fiber_cut flag is read to represent the controller's
    observed/announced network state for reactive failover.
    """
    required_traffic = {"timestamp", "tower", "demand_gbps", "fiber_cut"}
    required_predictions = {"timestamp", "target_timestamp", "predicted"}

    missing_traffic = required_traffic - set(traffic_df.columns)
    missing_predictions = required_predictions - set(prediction_df.columns)
    if missing_traffic:
        raise ValueError(f"Traffic data missing columns: {sorted(missing_traffic)}")
    if missing_predictions:
        raise ValueError(f"Prediction data missing columns: {sorted(missing_predictions)}")

    traffic = traffic_df.copy()
    predictions = prediction_df.copy()

    traffic["timestamp"] = pd.to_datetime(traffic["timestamp"], errors="raise")
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"], errors="raise")
    predictions["target_timestamp"] = pd.to_datetime(
        predictions["target_timestamp"], errors="raise"
    )
    traffic["demand_gbps"] = pd.to_numeric(traffic["demand_gbps"], errors="raise")
    predictions["predicted"] = pd.to_numeric(predictions["predicted"], errors="raise")
    traffic["fiber_cut"] = traffic["fiber_cut"].map(parse_bool)

    expected_target = predictions["timestamp"] + pd.Timedelta(minutes=30)
    if not (predictions["target_timestamp"] == expected_target).all():
        raise ValueError("Every prediction target must be exactly 30 minutes after its origin.")
    if predictions["timestamp"].duplicated().any():
        raise ValueError("Duplicate forecast-origin timestamps found.")
    if not np.isfinite(predictions["predicted"].to_numpy()).all():
        raise ValueError("Predictions contain non-finite values.")

    topology = create_network()
    expected_towers = {
        node for node, attrs in topology.nodes(data=True)
        if attrs.get("node_type") == "tower"
    }

    # Index once by timestamp to avoid repeatedly filtering the entire dataset.
    traffic_by_time = {
        timestamp: group.copy()
        for timestamp, group in traffic.groupby("timestamp", sort=False)
    }

    adjusted_rows = []
    for row in predictions.itertuples(index=False):
        origin_time = row.timestamp
        target_time = row.target_timestamp
        predicted_total = max(float(row.predicted), 0.0)

        # Only traffic observed at the forecast origin is used for tower shares.
        origin_rows = traffic_by_time.get(origin_time)
        if origin_rows is None or origin_rows.empty:
            raise ValueError(f"No tower observations at forecast origin {origin_time}.")

        origin_shares = origin_rows.groupby("tower")["demand_gbps"].sum()
        if set(origin_shares.index) != expected_towers:
            raise ValueError(
                f"Incomplete tower set at {origin_time}: expected {sorted(expected_towers)}."
            )

        observed_total = float(origin_shares.sum())
        if observed_total > 0:
            shares = origin_shares / observed_total
        else:
            shares = pd.Series(1.0 / len(expected_towers), index=sorted(expected_towers))

        # Read only the outage-state flag at target time, not future traffic demand.
        target_rows = traffic_by_time.get(target_time)
        if target_rows is None or target_rows.empty:
            raise ValueError(f"No network-state record at target time {target_time}.")
        target_outage_values = target_rows["fiber_cut"].map(parse_bool).unique()
        if len(target_outage_values) != 1:
            raise ValueError(f"Inconsistent fiber_cut state at {target_time}.")
        target_fiber_cut = bool(target_outage_values[0])

        for tower in sorted(expected_towers):
            adjusted_rows.append(
                {
                    "timestamp": target_time,
                    "forecast_origin_timestamp": origin_time,
                    "tower": tower,
                    "demand_gbps": predicted_total * float(shares[tower]),
                    "tower_share_at_origin": float(shares[tower]),
                    "predicted_total_demand_gbps": predicted_total,
                    "fiber_cut": target_fiber_cut,
                }
            )

    adjusted = pd.DataFrame(adjusted_rows)
    if adjusted.empty:
        raise ValueError("No forecast-adjusted traffic could be constructed.")

    # Numerical conservation check: tower allocations must sum to the forecast.
    allocated = adjusted.groupby("timestamp")["demand_gbps"].sum()
    predicted_totals = adjusted.groupby("timestamp")["predicted_total_demand_gbps"].first()
    if not np.allclose(allocated.to_numpy(), predicted_totals.to_numpy(), rtol=1e-9, atol=1e-9):
        raise RuntimeError("Tower allocations do not sum to the predicted aggregate demand.")

    return adjusted


def optimize_network(traffic_df: pd.DataFrame) -> pd.DataFrame:
    """Route higher-demand towers first using projected-utilization penalties.

    This is a greedy congestion-aware path-selection heuristic, not a global
    multi-commodity-flow optimizer and not a hard capacity-constrained solver.
    """
    required = {"timestamp", "tower", "demand_gbps", "fiber_cut"}
    missing = required - set(traffic_df.columns)
    if missing:
        raise ValueError(f"Optimization traffic missing columns: {sorted(missing)}")

    traffic = traffic_df.copy()
    traffic["timestamp"] = pd.to_datetime(traffic["timestamp"], errors="raise")
    traffic["demand_gbps"] = pd.to_numeric(traffic["demand_gbps"], errors="raise")
    traffic["fiber_cut"] = traffic["fiber_cut"].map(parse_bool)

    if (traffic["demand_gbps"] < 0).any():
        raise ValueError("Traffic demand cannot be negative.")

    results = []
    for timestamp, current in traffic.groupby("timestamp", sort=True):
        current = current.copy()
        if current["fiber_cut"].nunique() != 1:
            raise ValueError(f"Inconsistent fiber_cut flags at {timestamp}.")
        fiber_cut = bool(current["fiber_cut"].iloc[0])

        # Recreate topology each interval so backup changes never leak forward.
        topology = activate_backup_links(create_network(), fiber_cut)
        expected_towers = {
            node for node, attrs in topology.nodes(data=True)
            if attrs.get("node_type") == "tower"
        }
        if set(current["tower"]) != expected_towers:
            raise ValueError(f"Incomplete or unexpected tower set at {timestamp}.")
        if current["tower"].duplicated().any():
            raise ValueError(f"Duplicate tower traffic rows at {timestamp}.")

        active_graph = nx.Graph()
        for u, v, data in topology.edges(data=True):
            if data["active"]:
                active_graph.add_edge(
                    u,
                    v,
                    capacity=float(data["capacity"]),
                    latency=float(data["latency"]),
                    cost=float(data["cost"]),
                    link_type=data["link_type"],
                )

        link_loads = {
            get_link_key(u, v): 0.0
            for u, v in active_graph.edges()
        }
        unrouted_demand = 0.0

        # Greedy ordering makes large flows influence path choice first.
        current = current.sort_values("demand_gbps", ascending=False)
        for row in current.itertuples(index=False):
            tower = row.tower
            demand = float(row.demand_gbps)

            for u, v, data in active_graph.edges(data=True):
                key = get_link_key(u, v)
                projected_utilization = (
                    link_loads[key] + demand
                ) / data["capacity"]
                congestion_penalty = (
                    80.0 * projected_utilization**2
                    + 200.0 * max(0.0, projected_utilization - 0.8) ** 2
                )
                data["weight"] = (
                    data["latency"] + data["cost"] + congestion_penalty
                )

            try:
                path = nx.shortest_path(
                    active_graph,
                    source=tower,
                    target="CORE",
                    weight="weight",
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                unrouted_demand += demand
                continue

            for left, right in zip(path[:-1], path[1:]):
                link_loads[get_link_key(left, right)] += demand

        for u, v, data in topology.edges(data=True):
            active = bool(data["active"])
            capacity = float(data["capacity"])
            load = link_loads.get(get_link_key(u, v), 0.0) if active else 0.0
            utilization = load / capacity * 100.0 if active else 0.0

            results.append(
                {
                    "timestamp": timestamp,
                    "source": u,
                    "destination": v,
                    "link_type": data["link_type"],
                    "capacity_gbps": capacity,
                    "traffic_gbps": load,
                    "utilization": utilization,
                    "congested": bool(active and utilization > 80.0),
                    "active": active,
                    "fiber_cut": fiber_cut,
                    "unrouted_demand_gbps": unrouted_demand,
                }
            )

    return pd.DataFrame(results)


if __name__ == "__main__":
    traffic = pd.read_csv(
        ROOT / "data" / "raw" / "traffic.csv",
        parse_dates=["timestamp"],
    )
    predictions = pd.read_csv(
        ROOT / "data" / "processed" / "forecast_predictions.csv",
        parse_dates=["timestamp", "target_timestamp"],
    )

    forecast_adjusted_traffic = build_forecast_adjusted_traffic(
        traffic,
        predictions,
    )
    processed_dir = ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    forecast_adjusted_traffic.to_csv(
        processed_dir / "forecast_adjusted_traffic.csv", index=False
    )
    optimized_results = optimize_network(forecast_adjusted_traffic)

    output_path = ROOT / "data" / "processed" / "optimized_links.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    optimized_results.to_csv(output_path, index=False)

    active = optimized_results[optimized_results["active"]]
    print("\nDYNAMIC OPTIMIZATION COMPLETED")
    print("Total link records:", len(optimized_results))
    print("Active link records:", len(active))
    print("Congested active records:", int(active["congested"].sum()))
    print("Peak active utilization:", round(active["utilization"].max(), 2), "%")
    print("Average active utilization:", round(active["utilization"].mean(), 2), "%")
    print("Saved: data/processed/optimized_links.csv")
