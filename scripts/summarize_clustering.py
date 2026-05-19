#!/usr/bin/env python3
"""Print a concise summary of clustering outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from napi2b_vkr.io import validate_condition  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        required=True,
        choices=["normal", "tumor"],
        help="Dataset condition label.",
    )
    parser.add_argument(
        "--scores-path",
        required=True,
        help="Path to clustering_scores_{condition}.csv.",
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
        "--residue-centrality-path",
        default=None,
        help="Optional path to residue_centrality_{condition}.csv.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top residues to print per cluster.",
    )
    return parser


def _print_table(title: str, frame: pd.DataFrame) -> None:
    """Print a titled table."""

    print(title)
    if frame.empty:
        print("- <empty>")
    else:
        print(frame.to_string(index=False))
    print()


def main() -> int:
    """Print summary tables for a clustering run."""

    args = build_parser().parse_args()
    validate_condition(args.condition)

    scores_df = pd.read_csv(args.scores_path)
    labels_df = pd.read_csv(args.labels_path)
    features_df = pd.read_csv(args.features_path)

    best = scores_df.assign(
        silhouette_rank=scores_df["silhouette_score"].rank(ascending=False, method="min"),
        davies_rank=scores_df["davies_bouldin_score"].rank(ascending=True, method="min"),
        calinski_rank=scores_df["calinski_harabasz_score"].rank(
            ascending=False,
            method="min",
        ),
    )
    best["total_rank"] = (
        best["silhouette_rank"] + best["davies_rank"] + best["calinski_rank"]
    )
    best = best.sort_values("total_rank").iloc[0]

    cluster_sizes = (
        labels_df["cluster"].value_counts().sort_index().rename_axis("cluster").reset_index(
            name="n_frames"
        )
    )
    merged = features_df.merge(labels_df, on="frame", how="inner")
    feature_columns = [
        column
        for column in features_df.columns
        if column != "frame" and pd.api.types.is_numeric_dtype(features_df[column])
    ]
    mean_features = merged.groupby("cluster")[feature_columns].mean().reset_index()

    print(
        f"Best clustering for {args.condition}: "
        f"method={best['method']}, n_clusters={int(best['n_clusters'])}"
    )
    print()
    _print_table("Cluster sizes", cluster_sizes)
    _print_table("Mean features by cluster", mean_features)

    if args.residue_centrality_path:
        centrality_df = pd.read_csv(args.residue_centrality_path)
        required = {"cluster", "resid", "degree_centrality", "betweenness_centrality"}
        if required.issubset(centrality_df.columns):
            for cluster_id, cluster_rows in centrality_df.groupby("cluster", sort=True):
                top_degree = cluster_rows.sort_values(
                    "degree_centrality",
                    ascending=False,
                ).head(args.top_n)
                top_betweenness = cluster_rows.sort_values(
                    "betweenness_centrality",
                    ascending=False,
                ).head(args.top_n)
                _print_table(
                    f"Top residues by degree centrality for cluster {cluster_id}",
                    top_degree.loc[:, ["resid", "resname", "region", "degree_centrality"]],
                )
                _print_table(
                    f"Top residues by betweenness centrality for cluster {cluster_id}",
                    top_betweenness.loc[
                        :,
                        ["resid", "resname", "region", "betweenness_centrality"],
                    ],
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
