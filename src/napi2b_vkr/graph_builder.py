"""Build NetworkX residue contact graphs from Phase3A edge tables."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd

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

NODE_ATTRIBUTE_COLUMNS = (
    "resname",
    "aa1",
    "region",
    "isglycosylated",
    "label",
    "condition",
    "x_ca",
    "y_ca",
    "z_ca",
)


def find_first_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    """Return the first matching column name using case-insensitive matching."""

    by_lower = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    return None


def _require_column(
    frame: pd.DataFrame,
    candidates: tuple[str, ...],
    *,
    kind: str,
) -> str:
    """Resolve a required column name or raise a diagnostic error."""

    column = find_first_column(list(frame.columns), candidates)
    if column is None:
        raise ValueError(
            f"Could not determine {kind} column. "
            f"Expected one of {list(candidates)}, got {list(frame.columns)}."
        )
    return column


def _require_node_attribute_columns(residue_table: pd.DataFrame) -> None:
    """Validate the expected node attribute columns."""

    missing = [
        column for column in NODE_ATTRIBUTE_COLUMNS if column not in residue_table.columns
    ]
    if missing:
        raise ValueError(
            "Residue table is missing required node attribute columns: "
            f"{missing}. Available columns: {list(residue_table.columns)}."
        )


def get_residue_node_ids(residue_table: pd.DataFrame) -> list[int]:
    """Return the ordered residue node ids from the residue table."""

    resid_column = _require_column(
        residue_table,
        RESIDUE_ID_CANDIDATES,
        kind="residue id",
    )
    residue_ids = residue_table[resid_column]
    if residue_ids.isna().any():
        raise ValueError("Residue table contains missing residue ids.")
    if residue_ids.duplicated().any():
        duplicates = residue_ids[residue_ids.duplicated()].tolist()
        raise ValueError(f"Residue table contains duplicate residue ids: {duplicates[:10]}")
    return residue_ids.astype(int).tolist()


def _build_node_attribute_map(residue_table: pd.DataFrame) -> dict[int, dict[str, object]]:
    """Create a mapping from residue id to node attributes."""

    _require_node_attribute_columns(residue_table)
    resid_column = _require_column(
        residue_table,
        RESIDUE_ID_CANDIDATES,
        kind="residue id",
    )
    columns = [resid_column, *NODE_ATTRIBUTE_COLUMNS]
    records = residue_table.loc[:, columns].to_dict(orient="records")
    return {
        int(record[resid_column]): {
            key: record[key] for key in NODE_ATTRIBUTE_COLUMNS
        }
        for record in records
    }


def build_frame_graph(
    frame_edges: pd.DataFrame,
    residue_table: pd.DataFrame,
) -> nx.Graph:
    """Build one undirected residue contact graph for a single frame."""

    source_column = _require_column(
        frame_edges,
        EDGE_SOURCE_CANDIDATES,
        kind="edge source",
    )
    target_column = _require_column(
        frame_edges,
        EDGE_TARGET_CANDIDATES,
        kind="edge target",
    )

    missing_edge_columns = [
        column for column in ("contact_type", "distance") if column not in frame_edges.columns
    ]
    if missing_edge_columns:
        raise ValueError(
            "Frame edge table is missing required columns: "
            f"{missing_edge_columns}. Available columns: {list(frame_edges.columns)}."
        )

    node_ids = get_residue_node_ids(residue_table)
    node_attributes = _build_node_attribute_map(residue_table)
    graph = nx.Graph()
    graph.add_nodes_from(
        (node_id, node_attributes[node_id])
        for node_id in node_ids
    )

    if frame_edges.empty:
        return graph

    normalized_edges = frame_edges.loc[
        :, [source_column, target_column, "contact_type", "distance"]
    ].copy()
    normalized_edges[source_column] = normalized_edges[source_column].astype(int)
    normalized_edges[target_column] = normalized_edges[target_column].astype(int)

    known_nodes = set(node_ids)
    edge_node_ids = set(normalized_edges[source_column]) | set(normalized_edges[target_column])
    unknown_nodes = sorted(edge_node_ids - known_nodes)
    if unknown_nodes:
        raise ValueError(
            "Frame edges reference residue ids not present in residue_table: "
            f"{unknown_nodes[:10]}"
        )

    normalized_edges["_u"] = normalized_edges[[source_column, target_column]].min(axis=1)
    normalized_edges["_v"] = normalized_edges[[source_column, target_column]].max(axis=1)

    aggregated_edges = (
        normalized_edges.groupby(["_u", "_v"], sort=True, dropna=False)
        .agg(
            contact_types=("contact_type", lambda values: sorted(set(values))),
            n_contacts=("contact_type", "size"),
            min_distance=("distance", "min"),
            mean_distance=("distance", "mean"),
        )
        .reset_index()
    )

    for row in aggregated_edges.to_dict("records"):
        graph.add_edge(
            int(row["_u"]),
            int(row["_v"]),
            contact_types=list(row["contact_types"]),
            n_contacts=int(row["n_contacts"]),
            min_distance=float(row["min_distance"]),
            mean_distance=float(row["mean_distance"]),
    )

    return graph


def build_all_frame_graphs(
    edges_df: pd.DataFrame,
    residue_table: pd.DataFrame,
    max_frames: int | None = None,
) -> dict[int, nx.Graph]:
    """Build graphs for each frame in the edge table."""

    frame_column = _require_column(
        edges_df,
        FRAME_COLUMN_CANDIDATES,
        kind="frame",
    )
    frame_values = list(pd.unique(edges_df[frame_column]))
    if max_frames is not None:
        frame_values = frame_values[:max_frames]

    graphs: dict[int, nx.Graph] = {}
    for frame_value in frame_values:
        frame_edges = edges_df.loc[edges_df[frame_column] == frame_value]
        graphs[int(frame_value)] = build_frame_graph(frame_edges, residue_table)
    return graphs


def summarize_frame_graphs(graphs: dict[int, nx.Graph]) -> pd.DataFrame:
    """Summarize lightweight graph properties for each frame."""

    records: list[dict[str, float | int]] = []
    for frame in sorted(graphs):
        graph = graphs[frame]
        records.append(
            {
                "frame": int(frame),
                "n_nodes": graph.number_of_nodes(),
                "n_edges": graph.number_of_edges(),
                "density": nx.density(graph),
                "n_connected_components": nx.number_connected_components(graph),
            }
        )
    return pd.DataFrame.from_records(records)


def save_graph_summary(summary_df: pd.DataFrame, out_path: str | Path) -> Path:
    """Save the frame graph summary table to CSV."""

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(path, index=False)
    return path
