"""Graph-level feature calculation for frame contact graphs."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd

GRAPH_FEATURE_COLUMNS = [
    "frame",
    "n_nodes",
    "n_edges",
    "density",
    "average_clustering",
    "n_connected_components",
    "largest_component_fraction",
    "average_degree",
    "max_degree",
    "average_betweenness_centrality",
    "average_closeness_centrality",
]


def largest_component_fraction(graph: nx.Graph) -> float:
    """Return the fraction of nodes in the largest connected component."""

    n_nodes = graph.number_of_nodes()
    if n_nodes == 0:
        return 0.0

    components = list(nx.connected_components(graph))
    if not components:
        return 0.0
    largest_size = max(len(component) for component in components)
    return largest_size / n_nodes


def compute_graph_features(
    graph: nx.Graph,
    frame: int | None = None,
    compute_centrality: bool = True,
) -> dict[str, float | int]:
    """Compute lightweight graph features for one frame graph."""

    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    degrees = [degree for _, degree in graph.degree()]
    average_degree = float(sum(degrees) / n_nodes) if n_nodes else 0.0
    max_degree = int(max(degrees)) if degrees else 0

    features: dict[str, float | int] = {
        "frame": -1 if frame is None else int(frame),
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": float(nx.density(graph)),
        "average_clustering": float(nx.average_clustering(graph)),
        "n_connected_components": int(nx.number_connected_components(graph)),
        "largest_component_fraction": float(largest_component_fraction(graph)),
        "average_degree": average_degree,
        "max_degree": max_degree,
    }

    if compute_centrality:
        betweenness = nx.betweenness_centrality(graph)
        closeness = nx.closeness_centrality(graph)
        features["average_betweenness_centrality"] = float(
            sum(betweenness.values()) / n_nodes
        ) if n_nodes else 0.0
        features["average_closeness_centrality"] = float(
            sum(closeness.values()) / n_nodes
        ) if n_nodes else 0.0
    else:
        features["average_betweenness_centrality"] = float("nan")
        features["average_closeness_centrality"] = float("nan")

    return features


def build_feature_table(
    graphs: dict[int, nx.Graph],
    compute_centrality: bool = True,
) -> pd.DataFrame:
    """Build a per-frame feature table from frame graphs."""

    records = [
        compute_graph_features(
            graph,
            frame=frame,
            compute_centrality=compute_centrality,
        )
        for frame, graph in sorted(graphs.items())
    ]
    return pd.DataFrame.from_records(records, columns=GRAPH_FEATURE_COLUMNS)


def save_feature_table(features_df: pd.DataFrame, out_path: str | Path) -> Path:
    """Save the frame feature table to CSV."""

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(path, index=False)
    return path
