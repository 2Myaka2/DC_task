#!/usr/bin/env python3
"""Build a small number of per-frame residue graphs for a smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from napi2b_vkr.graph_builder import (  # noqa: E402
    build_all_frame_graphs,
    save_graph_summary,
    summarize_frame_graphs,
)
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
        help="Dataset condition to inspect.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=5,
        help="Number of first frames to build for the smoke test.",
    )
    return parser


def main() -> int:
    """Run the frame graph smoke test."""

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
    summary_df = summarize_frame_graphs(graphs)

    out_path = (
        REPO_ROOT
        / "results"
        / "tables"
        / f"frame_graph_summary_{condition}_smoke.csv"
    )
    save_graph_summary(summary_df, out_path)

    print(f"Condition: {condition}")
    print(f"Frames built: {len(graphs)}")
    print("Summary:")
    for row in summary_df.itertuples(index=False):
        print(
            f"- frame={row.frame}, n_nodes={row.n_nodes}, "
            f"n_edges={row.n_edges}, density={row.density:.6f}, "
            f"n_connected_components={row.n_connected_components}"
        )
    print(f"Saved summary: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
