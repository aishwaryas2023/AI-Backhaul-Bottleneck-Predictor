"""Synthetic 12-node backhaul topology used for the demo."""

import networkx as nx


def create_network() -> nx.Graph:
    """Create the fixed undirected backhaul graph.

    All capacities, latencies and costs are illustrative simulation
    assumptions, not measured Airtel network specifications.
    """
    graph = nx.Graph()

    towers = [f"T{i}" for i in range(1, 9)]
    hubs = ["H1", "H2"]

    for node in towers:
        graph.add_node(node, node_type="tower")
    for node in hubs:
        graph.add_node(node, node_type="hub")
    graph.add_node("CORE", node_type="core")
    graph.add_node("SAT", node_type="satellite")

    link_types = {
        "fiber": {"capacity": 10.0, "latency": 5.0, "cost": 1.0},
        "microwave": {"capacity": 2.0, "latency": 15.0, "cost": 3.0},
        "satellite": {"capacity": 0.5, "latency": 120.0, "cost": 10.0},
    }

    tower_connections = [
        ("T1", "H1", "fiber"), ("T1", "H2", "microwave"),
        ("T2", "H1", "microwave"), ("T2", "H2", "fiber"),
        ("T3", "H1", "fiber"), ("T3", "H2", "microwave"),
        ("T4", "H1", "fiber"), ("T4", "H2", "microwave"),
        ("T5", "H2", "fiber"), ("T5", "H1", "microwave"),
        ("T6", "H2", "microwave"), ("T6", "H1", "fiber"),
        ("T7", "H2", "fiber"), ("T7", "H1", "microwave"),
        ("T8", "H2", "fiber"), ("T8", "H1", "microwave"),
    ]

    infrastructure_connections = [
        ("H1", "CORE", "fiber"),
        ("H2", "CORE", "fiber"),
        ("H1", "H2", "microwave"),
        ("H1", "SAT", "satellite"),
        ("H2", "SAT", "satellite"),
        ("SAT", "CORE", "satellite"),
    ]

    for source, destination, link_type in (
        tower_connections + infrastructure_connections
    ):
        properties = link_types[link_type]
        graph.add_edge(
            source,
            destination,
            link_type=link_type,
            capacity=properties["capacity"],
            latency=properties["latency"],
            cost=properties["cost"],
            utilization=0.0,
            active=True,
        )

    # Guard against accidental topology edits.
    if graph.number_of_nodes() != 12:
        raise RuntimeError("Expected exactly 12 network nodes.")
    if graph.number_of_edges() != 22:
        raise RuntimeError("Expected exactly 22 network links.")
    if not nx.is_connected(graph):
        raise RuntimeError("The nominal network topology is disconnected.")

    for tower in towers:
        if graph.degree(tower) != 2:
            raise RuntimeError(f"{tower} must connect to two aggregation hubs.")

    return graph


if __name__ == "__main__":
    network = create_network()
    print("NETWORK CREATED SUCCESSFULLY")
    print("Total nodes:", network.number_of_nodes())
    print("Total links:", network.number_of_edges())
    print("Connected:", nx.is_connected(network))
    print("Link types:", dict(nx.get_edge_attributes(network, "link_type")))
