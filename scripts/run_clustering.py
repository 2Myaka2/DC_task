#!/usr/bin/env python3
"""Run clustering on a precomputed frame feature table."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from napi2b_vkr.clustering import (  # noqa: E402
    assemble_cluster_table,
    evaluate_agglomerative,
    evaluate_kmeans,
    fit_final_clustering,
    prepare_feature_matrix,
    run_pca,
    select_best_clustering,
)
from napi2b_vkr.io import validate_condition  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-path",
        required=True,
        help="Path to the input frame feature table CSV.",
    )
    parser.add_argument(
        "--condition",
        required=True,
        choices=["normal", "tumor"],
        help="Dataset condition label for output filenames.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/tables",
        help="Directory for clustering result CSV files.",
    )
    return parser


def main() -> int:
    """Run clustering from a precomputed feature table."""

    args = build_parser().parse_args()
    condition = validate_condition(args.condition)
    features_df = pd.read_csv(args.features_path)

    X_scaled, feature_names = prepare_feature_matrix(features_df)
    pca_df = run_pca(X_scaled, n_components=2)
    scores_df = pd.concat(
        [
            evaluate_kmeans(X_scaled),
            evaluate_agglomerative(X_scaled),
        ],
        ignore_index=True,
    )
    best = select_best_clustering(scores_df)
    labels = fit_final_clustering(
        X_scaled,
        method=str(best["method"]),
        n_clusters=int(best["n_clusters"]),
    )
    cluster_table = assemble_cluster_table(features_df, labels, pca_df)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_dir / f"clustering_scores_{condition}.csv"
    labels_path = output_dir / f"cluster_labels_{condition}.csv"
    pca_path = output_dir / f"pca_projection_{condition}.csv"

    scores_df.to_csv(scores_path, index=False)
    cluster_table.loc[:, ["frame", "cluster"]].to_csv(labels_path, index=False)
    pca_output = pd.concat(
        [features_df.loc[:, ["frame"]].reset_index(drop=True), pca_df],
        axis=1,
    )
    pca_output.to_csv(pca_path, index=False)

    print(f"Condition: {condition}")
    print(f"Input features: {args.features_path}")
    print(f"Feature columns used: {feature_names}")
    print(
        f"Selected clustering: method={best['method']}, "
        f"n_clusters={int(best['n_clusters'])}"
    )
    print(f"Saved scores: {scores_path}")
    print(f"Saved labels: {labels_path}")
    print(f"Saved PCA projection: {pca_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
