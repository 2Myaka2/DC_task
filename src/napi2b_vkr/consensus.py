"""Consensus graph helpers for cluster interpretation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd

from napi2b_vkr.features import largest_component_fraction


def validate_consensus_threshold(threshold: float) -> float:
    """Validate and return an edge frequency threshold in [0, 1]."""

    value = float(threshold)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Consensus threshold must be in [0, 1], got {threshold!r}.")
    return value


def build_consensus_graph(
    graphs: dict[int, nx.Graph],
    frame_ids: list[int],
    threshold: float = 0.5,
) -> nx.Graph:
    """Build a consensus graph for the selected frame ids."""

    threshold = validate_consensus_threshold(threshold)
    if not frame_ids:
        raise ValueError("Cannot build a consensus graph without frame ids.")

    missing_frames = [frame_id for frame_id in frame_ids if frame_id not in graphs]
    if missing_frames:
        raise ValueError(f"Missing graphs for frames: {missing_frames[:10]}")

    template_graph = graphs[frame_ids[0]]
    consensus_graph = nx.Graph()
    consensus_graph.add_nodes_from(template_graph.nodes(data=True))

    edge_stats: dict[tuple[int, int], dict[str, object]] = defaultdict(
        lambda: {
            "count": 0,
            "contact_types": set(),
            "min_distances": [],
            "mean_distances": [],
        }
    )

    for frame_id in frame_ids:
        graph = graphs[frame_id]
        for source, target, attrs in graph.edges(data=True):
            key = tuple(sorted((int(source), int(target))))
            stats = edge_stats[key]
            stats["count"] += 1
            stats["contact_types"].update(attrs.get("contact_types", []))
            if "min_distance" in attrs:
                stats["min_distances"].append(float(attrs["min_distance"]))
            if "mean_distance" in attrs:
                stats["mean_distances"].append(float(attrs["mean_distance"]))

    n_frames = len(frame_ids)
    for (source, target), stats in sorted(edge_stats.items()):
        edge_frequency = stats["count"] / n_frames
        if edge_frequency < threshold:
            continue
        min_distances = stats["min_distances"]
        mean_distances = stats["mean_distances"]
        consensus_graph.add_edge(
            source,
            target,
            edge_frequency=edge_frequency,
            contact_types=sorted(stats["contact_types"]),
            min_distance=min(min_distances) if min_distances else float("nan"),
            mean_distance=(
                sum(mean_distances) / len(mean_distances)
                if mean_distances
                else float("nan")
            ),
        )

    return consensus_graph


def build_cluster_consensus_graphs(
    graphs: dict[int, nx.Graph],
    labels_df: pd.DataFrame,
    threshold: float = 0.5,
) -> dict[int, nx.Graph]:
    """Build one consensus graph per cluster."""

    required = {"frame", "cluster"}
    missing = required - set(labels_df.columns)
    if missing:
        raise ValueError(
            f"Labels table is missing required columns: {sorted(missing)}."
        )

    consensus_graphs: dict[int, nx.Graph] = {}
    for cluster_id, cluster_rows in labels_df.groupby("cluster", sort=True):
        frame_ids = cluster_rows["frame"].astype(int).tolist()
        consensus_graphs[int(cluster_id)] = build_consensus_graph(
            graphs,
            frame_ids,
            threshold=threshold,
        )
    return consensus_graphs


def compute_consensus_graph_metrics(graph: nx.Graph) -> dict[str, float | int]:
    """Compute lightweight metrics for a consensus graph."""

    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    degrees = [degree for _, degree in graph.degree()]

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": float(nx.density(graph)),
        "average_clustering": float(nx.average_clustering(graph)),
        "n_connected_components": int(nx.number_connected_components(graph)),
        "largest_component_fraction": float(largest_component_fraction(graph)),
        "average_degree": (sum(degrees) / n_nodes) if n_nodes else 0.0,
        "max_degree": max(degrees) if degrees else 0,
    }


def compute_residue_centrality_table(
    graph: nx.Graph,
    residue_table: pd.DataFrame,
) -> pd.DataFrame:
    """Compute residue-level centrality values on a consensus graph."""

    degree_centrality = nx.degree_centrality(graph)
    betweenness_centrality = nx.betweenness_centrality(graph)

    centrality_df = pd.DataFrame(
        {
            "resid": list(graph.nodes()),
            "degree_centrality": [
                float(degree_centrality[node_id]) for node_id in graph.nodes()
            ],
            "betweenness_centrality": [
                float(betweenness_centrality[node_id]) for node_id in graph.nodes()
            ],
            "degree": [int(graph.degree(node_id)) for node_id in graph.nodes()],
        }
    )
    return residue_table.merge(centrality_df, on="resid", how="left")


def compute_region_involvement(
    graph: nx.Graph,
    residue_table: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize consensus involvement by residue region."""

    centrality_df = compute_residue_centrality_table(graph, residue_table)
    centrality_df["in_consensus_edge"] = centrality_df["degree"].fillna(0).gt(0)

    grouped = (
        centrality_df.groupby("region", dropna=False)
        .agg(
            n_residues=("resid", "size"),
            n_residues_involved=("in_consensus_edge", "sum"),
            mean_degree_centrality=("degree_centrality", "mean"),
            mean_betweenness_centrality=("betweenness_centrality", "mean"),
            sum_degree_centrality=("degree_centrality", "sum"),
            sum_betweenness_centrality=("betweenness_centrality", "sum"),
        )
        .reset_index()
    )
    grouped["involvement_fraction"] = (
        grouped["n_residues_involved"] / grouped["n_residues"]
    )
    return grouped


def save_table(table_df: pd.DataFrame, out_path: str | Path) -> Path:
    """Save a consensus-derived table to CSV."""

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table_df.to_csv(path, index=False)
    return path
