#!/usr/bin/env python3
"""Lightweight Phase3A diagnostics without full parquet loading."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from napi2b_vkr.io import (  # noqa: E402
    check_required_files,
    find_phase3a_files,
    get_csv_diagnostics,
    get_parquet_metadata,
    get_parquet_preview,
    read_json,
    validate_condition,
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
        default="normal",
        choices=["normal", "tumor"],
        help="Dataset condition to inspect.",
    )
    return parser


def format_file_status(name: str, path: Path) -> str:
    """Render one file status line."""

    status = "OK" if path.exists() else "MISSING"
    return f"- {name}: {status} ({path})"


def main() -> int:
    """Run the lightweight inspection."""

    args = build_parser().parse_args()
    condition = validate_condition(args.condition)
    files = find_phase3a_files(args.base_dir, condition)

    print(f"Phase3A directory: {files.residue_table.parent}")
    print(f"Condition: {condition}")
    print()
    print("Required files:")
    for name, path in files.as_dict().items():
        print(format_file_status(name, path))

    try:
        check_required_files(files)
    except FileNotFoundError as exc:
        print()
        print(str(exc))
        return 1

    phase3a_manifest = read_json(files.phase3a_manifest)
    dataset_manifest = read_json(files.dataset_manifest)
    validation = read_json(files.validation)
    residue_info = get_csv_diagnostics(files.residue_table)
    parquet_info = get_parquet_metadata(files.contact_edges_perframe)
    parquet_preview = get_parquet_preview(files.contact_edges_perframe, n_rows=5)

    print()
    print("Manifest keys:")
    print(f"- phase3a_manifest.json: {list(phase3a_manifest.keys())}")
    print(f"- dataset_manifest.json: {list(dataset_manifest.keys())}")
    print(f"- {files.validation.name}: {list(validation.keys())}")

    print()
    print(f"Residue table: {files.residue_table.name}")
    print(f"- shape: {residue_info['shape']}")
    print(f"- columns: {residue_info['columns']}")

    print()
    print(f"Parquet: {files.contact_edges_perframe.name}")
    print(f"- row groups: {parquet_info['num_row_groups']}")
    print(f"- rows: {parquet_info['num_rows']}")
    print(f"- columns: {parquet_info['num_columns']}")
    print("- schema:")
    print(parquet_info["schema_arrow"])

    print()
    print("Parquet preview (first 5 rows):")
    if parquet_preview["preview"]:
        for row in parquet_preview["preview"]:
            print(f"- {row}")
    else:
        print("- <empty parquet>")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
