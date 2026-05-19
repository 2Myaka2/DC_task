"""Consensus graph helpers.

The expensive consensus and centrality calculations should be run from
notebooks only after lightweight Phase3A diagnostics have passed.
"""

from __future__ import annotations


def validate_consensus_threshold(threshold: float) -> float:
    """Validate and return an edge frequency threshold in [0, 1]."""

    value = float(threshold)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Consensus threshold must be in [0, 1], got {threshold!r}.")
    return value
