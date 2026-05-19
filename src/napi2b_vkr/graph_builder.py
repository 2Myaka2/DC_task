"""Build NetworkX residue contact graphs from Phase3A edge tables.

Implementation will follow after the lightweight Phase3A diagnostics confirm
the per-frame parquet schema and residue identifier columns.
"""

from __future__ import annotations


RESIDUE_ID_CANDIDATES = (
    "resid",
    "residue_id",
    "residue_number",
    "res_seq",
    "resseq",
    "node_id",
)

FRAME_COLUMN_CANDIDATES = ("frame", "frame_id", "timestep", "time", "frame_idx")

EDGE_SOURCE_CANDIDATES = ("resid_i", "source", "src", "u", "node_i", "residue_i")
EDGE_TARGET_CANDIDATES = ("resid_j", "target", "dst", "v", "node_j", "residue_j")


def find_first_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    """Return the first matching column name using case-insensitive matching."""

    by_lower = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    return None
