#!/usr/bin/env python3
"""Interpret clustering results using consensus graphs and summary plots."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from napi2b_vkr.consensus import (  # noqa: E402
    build_cluster_consensus_graphs,
    compute_consensus_graph_metrics,
    compute_region_involvement,
    compute_residue_centrality_table,
    save_table,
)
from napi2b_vkr.graph_builder import build_all_frame_graphs  # noqa: E402
from napi2b_vkr.io import load_phase3a_dataset, validate_condition  # noqa: E402
from napi2b_vkr.plots import (  # noqa: E402
    plot_cluster_size_bar,
    plot_consensus_network,
    plot_feature_heatmap_by_cluster,
    plot_pca_clusters,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        default="data/phase3A/phase3aexports",
        help="Path to the Phase3A export directory.",
    )
    parser.add_argument(
        "--condition",
        required=True,
        choices=["normal", "tumor"],
        help="Dataset condition to interpret.",
    )
    parser.add_argument(
        "--labels-path",
        required=True,
        help="Path to cluster_labels_{condition}.csv.",
    )
    parser.add_argument(
        "--features-path",
        required=True,
        help="Path to frame_features_{condition}.csv.",
    )
    parser.add_argument(
        "--pca-path",
        required=True,
        help="Path to pca_projection_{condition}.csv.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Consensus edge frequency threshold.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Base output directory for tables and figures.",
    )
    return parser


def main() -> int:
    """Run cluster interpretation and save summary artifacts."""

    args = build_parser().parse_args()
    condition = validate_condition(args.condition)
    dataset = load_phase3a_dataset(
        args.base_dir,
        condition,
        load_perframe_edges=True,
    )
    graphs = build_all_frame_graphs(
        dataset["contact_edges_perframe"],
        dataset["residue_table"],
    )

    labels_df = pd.read_csv(args.labels_path)
    features_df = pd.read_csv(args.features_path)
    pca_df = pd.read_csv(args.pca_path)

    consensus_graphs = build_cluster_consensus_graphs(
        graphs,
        labels_df,
        threshold=args.threshold,
    )

    merged = features_df.merge(labels_df, on="frame", how="inner")
    cluster_summary = (
        merged.groupby("cluster")
        .agg(
            n_frames=("frame", "size"),
            n_nodes=("n_nodes", "mean"),
            n_edges=("n_edges", "mean"),
            density=("density", "mean"),
            average_clustering=("average_clustering", "mean"),
            n_connected_components=("n_connected_components", "mean"),
            largest_component_fraction=("largest_component_fraction", "mean"),
            average_degree=("average_degree", "mean"),
            max_degree=("max_degree", "mean"),
        )
        .reset_index()
    )

    metrics_rows = []
    residue_tables = []
    region_tables = []
    for cluster_id, graph in sorted(consensus_graphs.items()):
        metrics = compute_consensus_graph_metrics(graph)
        metrics["cluster"] = cluster_id
        metrics_rows.append(metrics)

        residue_table = compute_residue_centrality_table(graph, dataset["residue_table"])
        residue_table["cluster"] = cluster_id
        residue_tables.append(residue_table)

        region_table = compute_region_involvement(graph, dataset["residue_table"])
        region_table["cluster"] = cluster_id
        region_tables.append(region_table)

    metrics_df = pd.DataFrame(metrics_rows).sort_values("cluster")
    residue_centrality_df = pd.concat(residue_tables, ignore_index=True)
    region_involvement_df = pd.concat(region_tables, ignore_index=True)

    output_dir = Path(args.output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"

    save_table(cluster_summary, tables_dir / f"cluster_summary_{condition}.csv")
    save_table(
        metrics_df,
        tables_dir / f"consensus_graph_metrics_{condition}.csv",
    )
    save_table(
        residue_centrality_df,
        tables_dir / f"residue_centrality_{condition}.csv",
    )
    save_table(
        region_involvement_df,
        tables_dir / f"region_involvement_{condition}.csv",
    )

    plot_pca_clusters(
        pca_df,
        labels_df,
        figures_dir / f"pca_clusters_{condition}.png",
    )
    plot_cluster_size_bar(
        labels_df,
        figures_dir / f"cluster_sizes_{condition}.png",
    )
    plot_feature_heatmap_by_cluster(
        features_df,
        labels_df,
        figures_dir / f"feature_heatmap_{condition}.png",
    )
    for cluster_id, graph in sorted(consensus_graphs.items()):
        plot_consensus_network(
            graph,
            figures_dir / f"consensus_network_cluster_{cluster_id}_{condition}.png",
            title=f"Consensus Network Cluster {cluster_id} ({condition})",
        )

    print(f"Condition: {condition}")
    print(f"Frames interpreted: {len(labels_df)}")
    print(f"Clusters: {sorted(consensus_graphs)}")
    print(f"Saved tables under: {tables_dir}")
    print(f"Saved figures under: {figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
