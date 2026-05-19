"""Plotting helpers for NaPi2b clustering outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns

DEFAULT_FIGURE_DPI = 150


def _prepare_figure_path(out_path: str | Path) -> Path:
    """Ensure the figure directory exists and return the path."""

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_pca_clusters(
    pca_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    out_path: str | Path,
) -> Path:
    """Plot the PCA projection colored by cluster."""

    merged = pca_df.merge(labels_df, on="frame", how="inner")
    figure_path = _prepare_figure_path(out_path)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=DEFAULT_FIGURE_DPI)
    sns.scatterplot(
        data=merged,
        x="pca_1",
        y="pca_2",
        hue="cluster",
        palette="tab10",
        ax=ax,
    )
    ax.set_title("PCA Projection by Cluster")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def plot_cluster_size_bar(
    labels_df: pd.DataFrame,
    out_path: str | Path,
) -> Path:
    """Plot cluster sizes as a bar chart."""

    counts = (
        labels_df["cluster"].value_counts().sort_index().rename_axis("cluster").reset_index(
            name="n_frames"
        )
    )
    figure_path = _prepare_figure_path(out_path)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=DEFAULT_FIGURE_DPI)
    sns.barplot(
        data=counts,
        x="cluster",
        y="n_frames",
        hue="cluster",
        palette="tab10",
        legend=False,
        ax=ax,
    )
    ax.set_title("Frame Counts per Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Frames")
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def plot_feature_heatmap_by_cluster(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    out_path: str | Path,
) -> Path:
    """Plot cluster-wise mean features as a heatmap."""

    merged = features_df.merge(labels_df, on="frame", how="inner")
    feature_columns = [
        column
        for column in merged.columns
        if column not in {"frame", "cluster"}
        and pd.api.types.is_numeric_dtype(merged[column])
    ]
    heatmap_df = merged.groupby("cluster")[feature_columns].mean()
    figure_path = _prepare_figure_path(out_path)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=DEFAULT_FIGURE_DPI)
    sns.heatmap(heatmap_df, cmap="viridis", ax=ax)
    ax.set_title("Mean Features by Cluster")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Cluster")
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def plot_consensus_network(
    graph: nx.Graph,
    out_path: str | Path,
    title: str | None = None,
) -> Path:
    """Plot a consensus network using residue coordinates when available."""

    figure_path = _prepare_figure_path(out_path)
    positions: dict[int, tuple[float, float]] = {}
    for node_id, attrs in graph.nodes(data=True):
        x_ca = attrs.get("x_ca")
        y_ca = attrs.get("y_ca")
        if x_ca is not None and y_ca is not None and pd.notna(x_ca) and pd.notna(y_ca):
            positions[int(node_id)] = (float(x_ca), float(y_ca))

    if len(positions) != graph.number_of_nodes():
        positions = nx.spring_layout(graph, seed=42)

    edge_widths = [
        max(1.0, 4.0 * float(attrs.get("edge_frequency", 0.0)))
        for _, _, attrs in graph.edges(data=True)
    ]
    node_colors = [graph.degree(node_id) for node_id in graph.nodes()]

    fig, ax = plt.subplots(figsize=(10, 8), dpi=DEFAULT_FIGURE_DPI)
    nx.draw_networkx_edges(
        graph,
        positions,
        width=edge_widths,
        alpha=0.5,
        edge_color="steelblue",
        ax=ax,
    )
    nodes = nx.draw_networkx_nodes(
        graph,
        positions,
        node_size=40,
        node_color=node_colors,
        cmap="viridis",
        ax=ax,
    )
    plt.colorbar(nodes, ax=ax, label="Node degree")
    ax.set_title(title or "Consensus Network")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)
    return figure_path
