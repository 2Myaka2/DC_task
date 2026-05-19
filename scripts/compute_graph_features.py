#!/usr/bin/env python3
"""Compute per-frame graph features for Phase3A graphs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from napi2b_vkr.features import build_feature_table, save_feature_table  # noqa: E402
from napi2b_vkr.graph_builder import build_all_frame_graphs  # noqa: E402
from napi2b_vkr.io import load_phase3a_dataset, validate_condition  # noqa: E402


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
        default="normal",
        choices=["normal", "tumor"],
        help="Dataset condition to process.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional number of first frames to process.",
    )
    parser.add_argument(
        "--compute-centrality",
        action="store_true",
        help="Enable betweenness and closeness centrality averages.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/tables",
        help="Directory for the feature table CSV.",
    )
    return parser


def main() -> int:
    """Run feature calculation from Phase3A inputs."""

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
        max_frames=args.max_frames,
    )
    features_df = build_feature_table(
        graphs,
        compute_centrality=args.compute_centrality,
    )

    suffix = "_smoke" if args.max_frames is not None else ""
    out_path = Path(args.output_dir) / f"frame_features_{condition}{suffix}.csv"
    save_feature_table(features_df, out_path)

    print(f"Condition: {condition}")
    print(f"Frames processed: {len(graphs)}")
    print(f"Centrality computed: {args.compute_centrality}")
    print(f"Saved feature table: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
