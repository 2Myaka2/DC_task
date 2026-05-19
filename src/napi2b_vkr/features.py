"""Graph-level feature calculation for frame contact graphs.

Expensive all-frame calculations are intentionally deferred until the analysis
notebooks are ready to run on local data.
"""

from __future__ import annotations


GRAPH_FEATURE_COLUMNS = [
    "n_nodes",
    "n_edges",
    "density",
    "average_clustering",
    "n_connected_components",
    "largest_component_fraction",
    "average_degree",
    "max_degree",
    "average_betweenness_centrality",
    "average_closeness_centrality",
]
