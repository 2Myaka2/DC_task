"""Clustering helpers for graph-level frame features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

FEATURE_ID_COLUMNS = {"condition", "frame", "frame_id", "cluster", "method"}


def feature_columns(columns: list[str]) -> list[str]:
    """Return candidate numeric feature columns by excluding known identifiers."""

    return [column for column in columns if column not in FEATURE_ID_COLUMNS]


def prepare_feature_matrix(
    features_df: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """Select numeric feature columns, drop rows with missing values, and scale."""

    candidate_columns = feature_columns(list(features_df.columns))
    usable_columns = [
        column
        for column in candidate_columns
        if pd.api.types.is_numeric_dtype(features_df[column])
    ]
    usable_columns = [
        column for column in usable_columns if not features_df[column].isna().any()
    ]
    if not usable_columns:
        raise ValueError("No usable numeric feature columns found for clustering.")

    matrix = features_df.loc[:, usable_columns].to_numpy(dtype=float)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    return scaled, usable_columns


def run_pca(X_scaled: np.ndarray, n_components: int = 2) -> pd.DataFrame:
    """Project the scaled features into PCA space."""

    pca = PCA(n_components=n_components)
    projection = pca.fit_transform(X_scaled)
    columns = [f"pca_{idx + 1}" for idx in range(projection.shape[1])]
    pca_df = pd.DataFrame(projection, columns=columns)
    explained = pca.explained_variance_ratio_
    for idx, value in enumerate(explained, start=1):
        pca_df[f"pca_{idx}_explained_variance_ratio"] = value
    return pca_df


def _evaluate_labels(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    *,
    method: str,
    n_clusters: int,
) -> dict[str, float | int | str]:
    """Compute clustering quality metrics for one label assignment."""

    return {
        "method": method,
        "n_clusters": int(n_clusters),
        "silhouette_score": float(silhouette_score(X_scaled, labels)),
        "davies_bouldin_score": float(davies_bouldin_score(X_scaled, labels)),
        "calinski_harabasz_score": float(calinski_harabasz_score(X_scaled, labels)),
    }


def evaluate_kmeans(
    X_scaled: np.ndarray,
    k_values: range = range(2, 9),
) -> pd.DataFrame:
    """Evaluate KMeans across a range of cluster counts."""

    records = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(X_scaled)
        records.append(_evaluate_labels(X_scaled, labels, method="kmeans", n_clusters=k))
    return pd.DataFrame.from_records(records)


def evaluate_agglomerative(
    X_scaled: np.ndarray,
    k_values: range = range(2, 9),
) -> pd.DataFrame:
    """Evaluate AgglomerativeClustering across a range of cluster counts."""

    records = []
    for k in k_values:
        model = AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(X_scaled)
        records.append(
            _evaluate_labels(X_scaled, labels, method="agglomerative", n_clusters=k)
        )
    return pd.DataFrame.from_records(records)


def select_best_clustering(scores_df: pd.DataFrame) -> pd.Series:
    """Select the best clustering configuration by rank aggregation."""

    ranked = scores_df.copy()
    ranked["silhouette_rank"] = ranked["silhouette_score"].rank(
        ascending=False,
        method="min",
    )
    ranked["davies_bouldin_rank"] = ranked["davies_bouldin_score"].rank(
        ascending=True,
        method="min",
    )
    ranked["calinski_harabasz_rank"] = ranked["calinski_harabasz_score"].rank(
        ascending=False,
        method="min",
    )
    ranked["total_rank"] = (
        ranked["silhouette_rank"]
        + ranked["davies_bouldin_rank"]
        + ranked["calinski_harabasz_rank"]
    )
    ranked = ranked.sort_values(
        by=[
            "total_rank",
            "silhouette_rank",
            "davies_bouldin_rank",
            "calinski_harabasz_rank",
        ],
        ascending=[True, True, True, True],
    )
    return ranked.iloc[0]


def fit_final_clustering(
    X_scaled: np.ndarray,
    method: str,
    n_clusters: int,
) -> np.ndarray:
    """Fit the final clustering model and return labels."""

    normalized = method.strip().lower()
    if normalized == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        return model.fit_predict(X_scaled)
    if normalized == "agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters)
        return model.fit_predict(X_scaled)
    raise ValueError(
        f"Unknown clustering method {method!r}. Expected 'kmeans' or 'agglomerative'."
    )


def assemble_cluster_table(
    features_df: pd.DataFrame,
    labels: np.ndarray,
    pca_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine frame features, cluster labels, and PCA projection."""

    cluster_df = features_df.reset_index(drop=True).copy()
    cluster_df["cluster"] = labels.astype(int)
    pca_reset = pca_df.reset_index(drop=True)
    return pd.concat([cluster_df, pca_reset], axis=1)
