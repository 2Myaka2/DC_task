"""Input/output helpers for Phase3A NaPi2b graph exports.

The diagnostics in this module intentionally avoid loading full parquet files.
Use :func:`load_phase3a_dataset` only when a notebook or script is ready to
work with the per-frame edge table.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


VALID_CONDITIONS = {"normal", "tumor"}

PHASE3A_STATIC_FILES = {
    "phase3a_manifest": "phase3a_manifest.json",
    "dataset_manifest": "dataset_manifest.json",
}

PHASE3A_CONDITION_FILES = {
    "residue_table": "residue_table_{condition}.csv",
    "contact_edges_perframe": "contact_edges_perframe_{condition}.parquet",
    "protein_contact_edges_undirected": (
        "protein_contact_edges_undirected_{condition}.csv"
    ),
    "pos3d": "pos3d_{condition}.json",
    "validation": "{condition}_validation.json",
}


@dataclass(frozen=True)
class Phase3AFiles:
    """Resolved paths for one Phase3A condition."""

    phase3a_manifest: Path
    dataset_manifest: Path
    residue_table: Path
    contact_edges_perframe: Path
    protein_contact_edges_undirected: Path
    pos3d: Path
    validation: Path

    def as_dict(self) -> dict[str, Path]:
        return {
            "phase3a_manifest": self.phase3a_manifest,
            "dataset_manifest": self.dataset_manifest,
            "residue_table": self.residue_table,
            "contact_edges_perframe": self.contact_edges_perframe,
            "protein_contact_edges_undirected": self.protein_contact_edges_undirected,
            "pos3d": self.pos3d,
            "validation": self.validation,
        }


def validate_condition(condition: str) -> str:
    """Return a normalized condition name or raise a clear error."""

    normalized = str(condition).lower().strip()
    if normalized not in VALID_CONDITIONS:
        allowed = ", ".join(sorted(VALID_CONDITIONS))
        raise ValueError(f"Unknown condition {condition!r}. Expected one of: {allowed}.")
    return normalized


def resolve_phase3a_dir(base_dir: str | Path | None = None) -> Path:
    """Resolve the Phase3A export directory.

    `base_dir` may be either the repository root or the export directory itself.
    When omitted, the default is `data/phase3A/phase3aexports` relative to the
    current working directory.
    """

    if base_dir is None:
        base = Path("data/phase3A/phase3aexports")
    else:
        base = Path(base_dir)

    if base.name == "phase3aexports":
        return base

    candidate = base / "data" / "phase3A" / "phase3aexports"
    if candidate.exists() or not base.exists():
        return candidate
    return base


def find_phase3a_files(
    base_dir: str | Path | None = None,
    condition: str = "normal",
) -> Phase3AFiles:
    """Build expected Phase3A file paths for `condition`."""

    condition = validate_condition(condition)
    phase3a_dir = resolve_phase3a_dir(base_dir)
    paths: dict[str, Path] = {}

    for key, filename in PHASE3A_STATIC_FILES.items():
        paths[key] = phase3a_dir / filename
    for key, template in PHASE3A_CONDITION_FILES.items():
        paths[key] = phase3a_dir / template.format(condition=condition)

    return Phase3AFiles(**paths)


def check_required_files(files: Phase3AFiles) -> None:
    """Raise FileNotFoundError with diagnostics if any required file is missing."""

    missing = [f"{name}: {path}" for name, path in files.as_dict().items() if not path.exists()]
    if missing:
        details = "\n".join(f"- {line}" for line in missing)
        raise FileNotFoundError(f"Missing required Phase3A files:\n{details}")


def read_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read a CSV file with pandas."""

    return pd.read_csv(Path(path), **kwargs)


def read_parquet(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read parquet data explicitly.

    This is intentionally separate from diagnostics because it may load a large
    per-frame edge table into memory.
    """

    return pd.read_parquet(Path(path), columns=columns)


def get_parquet_metadata(path: str | Path) -> dict[str, Any]:
    """Return parquet schema and metadata without loading row groups."""

    parquet_file = pq.ParquetFile(Path(path))
    metadata = parquet_file.metadata
    return {
        "path": str(Path(path)),
        "num_rows": metadata.num_rows,
        "num_row_groups": metadata.num_row_groups,
        "num_columns": metadata.num_columns,
        "schema": parquet_file.schema_arrow.names,
        "schema_arrow": str(parquet_file.schema_arrow),
    }


def inspect_phase3a_dataset(
    base_dir: str | Path | None = None,
    condition: str = "normal",
) -> dict[str, Any]:
    """Return lightweight diagnostics for a Phase3A condition.

    The parquet file is inspected through metadata only.
    """

    condition = validate_condition(condition)
    files = find_phase3a_files(base_dir, condition)
    check_required_files(files)

    residue_table = read_csv(files.residue_table, nrows=5)
    undirected_edges = read_csv(files.protein_contact_edges_undirected, nrows=5)

    return {
        "condition": condition,
        "phase3a_dir": str(files.residue_table.parent),
        "files": {key: str(path) for key, path in files.as_dict().items()},
        "residue_table": {
            "columns": list(residue_table.columns),
            "preview": residue_table.to_dict(orient="records"),
        },
        "protein_contact_edges_undirected": {
            "columns": list(undirected_edges.columns),
            "preview": undirected_edges.to_dict(orient="records"),
        },
        "parquet": get_parquet_metadata(files.contact_edges_perframe),
        "phase3a_manifest_keys": list(read_json(files.phase3a_manifest).keys()),
        "dataset_manifest_keys": list(read_json(files.dataset_manifest).keys()),
        "validation_keys": list(read_json(files.validation).keys()),
    }


def load_phase3a_dataset(
    base_dir: str | Path | None = None,
    condition: str = "normal",
    *,
    load_perframe_edges: bool = False,
) -> dict[str, Any]:
    """Load Phase3A tables and manifests for one condition.

    By default this does not load `contact_edges_perframe_*.parquet`; it returns
    parquet metadata instead. Pass `load_perframe_edges=True` only for an
    explicit analysis run.
    """

    condition = validate_condition(condition)
    files = find_phase3a_files(base_dir, condition)
    check_required_files(files)

    dataset: dict[str, Any] = {
        "condition": condition,
        "files": files,
        "phase3a_manifest": read_json(files.phase3a_manifest),
        "dataset_manifest": read_json(files.dataset_manifest),
        "validation": read_json(files.validation),
        "pos3d": read_json(files.pos3d),
        "residue_table": read_csv(files.residue_table),
        "protein_contact_edges_undirected": read_csv(
            files.protein_contact_edges_undirected
        ),
        "contact_edges_perframe_metadata": get_parquet_metadata(
            files.contact_edges_perframe
        ),
    }

    if load_perframe_edges:
        dataset["contact_edges_perframe"] = read_parquet(files.contact_edges_perframe)

    return dataset
