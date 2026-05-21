#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DOMAIN_ROWS = [
    (1, 101, "N-domain", "CD"),
    (102, 125, "TM1", "TMD"),
    (126, 141, "EMD2", "ECD"),
    (142, 172, "loop1", "loop"),
    (173, 176, "EMD3", "ECD"),
    (177, 198, "TM2", "TMD"),
    (199, 216, "EMD4", "CD"),
    (217, 241, "TM3", "TMD"),
    (242, 361, "EMD5", "ECD"),
    (360, 380, "TM4", "TMD"),
    (381, 414, "EMD6", "CD"),
    (415, 444, "loop2", "loop"),
    (445, 453, "EMD7", "CD"),
    (454, 476, "TM5", "TMD"),
    (477, 481, "EMD8", "ECD"),
    (482, 508, "TM6", "TMD"),
    (509, 521, "EMD9", "CD"),
    (522, 540, "TM7", "TMD"),
    (541, 558, "EMD10", "ECD"),
    (559, 577, "TM8", "TMD"),
    (578, 690, "C-domain", "CD"),
]


def assign_domain(resid: int) -> tuple[str, str]:
    for beg, end, domain, topology in DOMAIN_ROWS:
        if beg <= int(resid) <= end:
            return domain, topology
    return "unknown", "unknown"


def plot_cluster_timeline(condition: str, results_dir: Path) -> None:
    labels = pd.read_csv(results_dir / "tables" / f"cluster_labels_{condition}.csv")
    labels = labels.sort_values("frame")

    out_dir = results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 3))
    plt.step(labels["frame"], labels["cluster"], where="post")
    plt.scatter(labels["frame"], labels["cluster"], s=10)
    plt.title(f"Cluster timeline ({condition})")
    plt.xlabel("Frame")
    plt.ylabel("Cluster")
    plt.yticks(sorted(labels["cluster"].unique()))
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"cluster_timeline_{condition}.png", dpi=200)
    plt.close()

    # consecutive segments of the same cluster
    seg = labels.copy()
    seg["segment"] = (seg["cluster"] != seg["cluster"].shift()).cumsum()
    segments = (
        seg.groupby("segment")
        .agg(
            cluster=("cluster", "first"),
            frame_start=("frame", "min"),
            frame_end=("frame", "max"),
            n_frames=("frame", "size"),
        )
        .reset_index(drop=True)
    )
    segments.to_csv(results_dir / "tables" / f"cluster_timeline_segments_{condition}.csv", index=False)


def annotate_domains(condition: str, results_dir: Path) -> None:
    path = results_dir / "tables" / f"residue_centrality_{condition}.csv"
    df = pd.read_csv(path)

    mapped = df["resid"].apply(assign_domain)
    df["domain"] = [x[0] for x in mapped]
    df["topology"] = [x[1] for x in mapped]

    out_path = results_dir / "tables" / f"residue_centrality_{condition}_domains.csv"
    df.to_csv(out_path, index=False)

    top_betweenness = (
        df.sort_values(["cluster", "betweenness_centrality"], ascending=[True, False])
        .groupby("cluster")
        .head(15)
    )
    top_betweenness.to_csv(
        results_dir / "tables" / f"top_residues_betweenness_{condition}_domains.csv",
        index=False,
    )

    top_degree = (
        df.sort_values(["cluster", "degree_centrality"], ascending=[True, False])
        .groupby("cluster")
        .head(15)
    )
    top_degree.to_csv(
        results_dir / "tables" / f"top_residues_degree_{condition}_domains.csv",
        index=False,
    )

    domain_summary = (
        df.groupby(["cluster", "domain", "topology"])
        .agg(
            n_residues=("resid", "size"),
            n_involved=("degree", lambda s: int((s > 0).sum())),
            mean_degree_centrality=("degree_centrality", "mean"),
            mean_betweenness_centrality=("betweenness_centrality", "mean"),
            max_betweenness_centrality=("betweenness_centrality", "max"),
        )
        .reset_index()
    )
    domain_summary["involvement_fraction"] = (
        domain_summary["n_involved"] / domain_summary["n_residues"]
    )
    domain_summary.to_csv(
        results_dir / "tables" / f"domain_involvement_{condition}.csv",
        index=False,
    )


def compare_thresholds(condition: str) -> pd.DataFrame:
    rows = []
    for threshold, base in [
        (0.4, Path("results_threshold_04")),
        (0.5, Path("results")),
        (0.6, Path("results_threshold_06")),
    ]:
        metrics_path = base / "tables" / f"consensus_graph_metrics_{condition}.csv"
        metrics = pd.read_csv(metrics_path)
        metrics["threshold"] = threshold
        metrics["condition"] = condition
        rows.append(metrics)

    out = pd.concat(rows, ignore_index=True)
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    out.to_csv(
        Path("results/tables") / f"threshold_sensitivity_{condition}.csv",
        index=False,
    )
    return out


def main() -> None:
    results_dir = Path("results")
    for condition in ["normal", "tumor"]:
        plot_cluster_timeline(condition, results_dir)
        annotate_domains(condition, results_dir)
        sensitivity = compare_thresholds(condition)

        print(f"\n=== {condition.upper()} threshold sensitivity ===")
        print(
            sensitivity[
                [
                    "condition",
                    "threshold",
                    "cluster",
                    "n_edges",
                    "density",
                    "n_connected_components",
                    "largest_component_fraction",
                    "average_degree",
                    "max_degree",
                ]
            ].to_string(index=False)
        )

        print(f"\nCreated timeline and domain annotation for {condition}")


if __name__ == "__main__":
    main()