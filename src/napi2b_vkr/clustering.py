"""Clustering helpers for graph-level frame features.

Full clustering implementation is intentionally left for the analysis step.
This module exists in the package namespace so notebooks can import from
`napi2b_vkr.clustering` consistently.
"""

from __future__ import annotations


FEATURE_ID_COLUMNS = {"condition", "frame", "frame_id", "cluster", "method"}


def feature_columns(columns: list[str]) -> list[str]:
    """Return candidate numeric feature columns by excluding known identifiers."""

    return [column for column in columns if column not in FEATURE_ID_COLUMNS]
