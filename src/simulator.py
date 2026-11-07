"""Baseline shortest-path network simulation."""

from pathlib import Path

import networkx as nx
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


def simulate_network(traffic_df: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "tower", "demand_gbps", "fiber_cut"}
    missing = required - set(traffic_df.columns)
    if missing:
        raise ValueError(f"Traffic data is missing columns: {sorted(missing)}")

    traffic = traffic_df.copy()
    traffic["timestamp"] = pd.to_datetime(traffic["timestamp"], errors="raise")
    traffic["demand_gbps"] = pd.to_numeric(traffic["demand_gbps"], errors="raise")
    traffic["fiber_cut"] = traffic["fiber_cut"].map(parse_bool)

    if traffic[["timestamp", "tower", "demand_gbps"]].isna().any().any():
        raise ValueError("Traffic input contains missing timestamps, towers or demand.")
    if (traffic["demand_gbps"] < 0).any():
        raise ValueError("Traffic demand cannot be negative.")

    topology = create_network()
    expected_towers = {node for node, attrs in topology.nodes(data=True) if attrs["node_type"] == "tower"}
    supplied_towers = set(traffic["tower"].unique())
    if supplied_towers != expected_towers:
        raise ValueError(
            f"Traffic towers do not match topology. Expected {sorted(expected_towers)}, "
            f"received {sorted(supplied_towers)}."
        )

    results = []
    for timestamp, current in traffic.groupby("timestamp", sort=True):
        if current["tower"].duplicated().any():
            raise ValueError(f"Duplicate tower rows at {timestamp}.")
        if current["fiber_cut"].nunique() != 1:
            raise ValueError(f"Inconsistent fiber_cut flags at {timestamp}.")
        fiber_cut = bool(current["fiber_cut"].iloc[0])

        # Apply the outage state afresh for each simulation interval.
        for _, _, data in topology.edges(data=True):
            data["active"] = not (fiber_cut and data["link_type"] == "fiber")

        active_graph = nx.Graph()
        for u, v, data in topology.edges(data=True):
            if data["active"]:
                active_graph.add_edge(
                    u,
                    v,
                    weight=float(data["latency"]) + float(data["cost"]),
                )

        link_loads = {get_link_key(u, v): 0.0 for u, v in topology.edges()}
        unrouted_demand = 0.0

        for row in current.itertuples(index=False):
            tower = row.tower
            demand = float(row.demand_gbps)
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
            load = link_loads[get_link_key(u, v)] if active else 0.0
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
                    # Repeated for each link at this timestamp; deduplicate by timestamp when summarizing.
                    "unrouted_demand_gbps": unrouted_demand,
                }
            )

    return pd.DataFrame(results)


if __name__ == "__main__":
    input_path = ROOT / "data" / "raw" / "traffic.csv"
    traffic = pd.read_csv(input_path, parse_dates=["timestamp"])
    results = simulate_network(traffic)

    output_path = ROOT / "data" / "processed" / "baseline_links.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    active = results[results["active"]]
    print("\nBASELINE SIMULATION COMPLETED")
    print("Total link records:", len(results))
    print("Active link records:", len(active))
    print("Congested active records:", int(active["congested"].sum()))
    print("Peak active utilization:", round(active["utilization"].max(), 2), "%")
    print("Saved: data/processed/baseline_links.csv")
