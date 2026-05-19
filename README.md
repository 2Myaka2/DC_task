# NaPi2b VKR Graph Clustering

This repository contains the Python pipeline for MSc thesis work on clustering
amino acid residue interaction networks to identify conformational states of the
NaPi2b transporter.

Each trajectory frame is represented as a protein residue contact graph:
residues are nodes, residue-residue contacts are edges, and graph-level
features are used for clustering and interpretation. The full pipeline supports
Phase3A inspection, per-frame graph construction, graph feature extraction,
clustering, consensus network construction, cluster interpretation, and figure
generation.

## Local Data Layout

Raw data and generated outputs are not stored in Git. Put the Phase3A exports
locally at:

```text
data/phase3A/phase3aexports/
```

Expected Phase3A files:

```text
phase3a_manifest.json
dataset_manifest.json
residue_table_normal.csv
residue_table_tumor.csv
contact_edges_perframe_normal.parquet
contact_edges_perframe_tumor.parquet
protein_contact_edges_undirected_normal.csv
protein_contact_edges_undirected_tumor.csv
pos3d_normal.json
pos3d_tumor.json
normal_validation.json
tumor_validation.json
```

The pipeline writes tables to `results/tables/` and figures to
`results/figures/`. Both `data/` and `results/` are ignored by Git.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If your environment exposes Python as `python3` only, use `python3 -m pip`
instead.

## Configuration

Copy or adapt `configs/paths.example.yaml` if you want to keep local path
settings outside notebooks:

```yaml
phase3a_dir: data/phase3A/phase3aexports
phase3b_dir: data/phase3B/phase3boutputs
results_dir: results
default_condition: normal
```

## Running The Pipeline

The notebooks are intended to be run in order:

```text
notebooks/01_data_inspection.ipynb
notebooks/02_build_frame_graphs.ipynb
notebooks/03_graph_features.ipynb
notebooks/04_clustering.ipynb
notebooks/05_cluster_interpretation.ipynb
```

The notebooks should import functions from `src/napi2b_vkr/` rather than
duplicating pipeline logic.

Lightweight inspection:

```python
from napi2b_vkr.io import inspect_phase3a_dataset

report = inspect_phase3a_dataset("data/phase3A/phase3aexports", "normal")
report["parquet"]["schema"]
```

Or run the diagnostic script directly:

```bash
python3 scripts/inspect_phase3a.py --base-dir data/phase3A/phase3aexports --condition normal
```

Smoke and debug examples:

```bash
python3 scripts/build_frame_graphs_smoke.py --base-dir data/phase3A/phase3aexports --condition normal --max-frames 5
```

For graph feature calculation in a quick smoke mode:

```bash
python3 scripts/compute_graph_features.py --base-dir data/phase3A/phase3aexports --condition normal --max-frames 5 --output-dir results/tables
```

Main examples:

```bash
python3 scripts/compute_graph_features.py --base-dir data/phase3A/phase3aexports --condition normal --compute-centrality --output-dir results/tables
python3 scripts/compute_graph_features.py --base-dir data/phase3A/phase3aexports --condition tumor --compute-centrality --output-dir results/tables
```

```bash
python3 scripts/run_clustering.py --features-path results/tables/frame_features_normal.csv --condition normal --output-dir results/tables
python3 scripts/run_clustering.py --features-path results/tables/frame_features_tumor.csv --condition tumor --output-dir results/tables
```

```bash
python3 scripts/interpret_clusters.py --base-dir data/phase3A/phase3aexports --condition normal --labels-path results/tables/cluster_labels_normal.csv --features-path results/tables/frame_features_normal.csv --pca-path results/tables/pca_projection_normal.csv --threshold 0.5 --output-dir results
python3 scripts/interpret_clusters.py --base-dir data/phase3A/phase3aexports --condition tumor --labels-path results/tables/cluster_labels_tumor.csv --features-path results/tables/frame_features_tumor.csv --pca-path results/tables/pca_projection_tumor.csv --threshold 0.5 --output-dir results
```

```bash
python3 scripts/summarize_clustering.py --condition normal --scores-path results/tables/clustering_scores_normal.csv --labels-path results/tables/cluster_labels_normal.csv --features-path results/tables/frame_features_normal.csv --residue-centrality-path results/tables/residue_centrality_normal.csv
python3 scripts/summarize_clustering.py --condition tumor --scores-path results/tables/clustering_scores_tumor.csv --labels-path results/tables/cluster_labels_tumor.csv --features-path results/tables/frame_features_tumor.csv --residue-centrality-path results/tables/residue_centrality_tumor.csv
```

## Full Reproducible Run

```bash
python3 scripts/compute_graph_features.py --base-dir data/phase3A/phase3aexports --condition normal --output-dir results/tables
python3 scripts/compute_graph_features.py --base-dir data/phase3A/phase3aexports --condition tumor --output-dir results/tables

python3 scripts/run_clustering.py --features-path results/tables/frame_features_normal.csv --condition normal --output-dir results/tables
python3 scripts/run_clustering.py --features-path results/tables/frame_features_tumor.csv --condition tumor --output-dir results/tables

python3 scripts/interpret_clusters.py --base-dir data/phase3A/phase3aexports --condition normal --labels-path results/tables/cluster_labels_normal.csv --features-path results/tables/frame_features_normal.csv --pca-path results/tables/pca_projection_normal.csv --threshold 0.5 --output-dir results
python3 scripts/interpret_clusters.py --base-dir data/phase3A/phase3aexports --condition tumor --labels-path results/tables/cluster_labels_tumor.csv --features-path results/tables/frame_features_tumor.csv --pca-path results/tables/pca_projection_tumor.csv --threshold 0.5 --output-dir results
```

Observed full-run outcomes on the local Phase3A dataset:
- `normal`: 301 frames, selected clustering `kmeans`, `n_clusters=3`
- `tumor`: 301 frames, selected clustering `kmeans`, `n_clusters=2`

## Expected Outputs

Tables:

```text
results/tables/frame_features_{condition}.csv
results/tables/clustering_scores_{condition}.csv
results/tables/cluster_labels_{condition}.csv
results/tables/pca_projection_{condition}.csv
results/tables/cluster_summary_{condition}.csv
results/tables/consensus_graph_metrics_{condition}.csv
results/tables/residue_centrality_{condition}.csv
results/tables/region_involvement_{condition}.csv
```

Figures:

```text
results/figures/pca_clusters_{condition}.png
results/figures/cluster_sizes_{condition}.png
results/figures/feature_heatmap_{condition}.png
results/figures/consensus_network_cluster_*_{condition}.png
```

## Package Structure

```text
src/napi2b_vkr/
  __init__.py
  io.py
  graph_builder.py
  features.py
  clustering.py
  consensus.py
  plots.py
```

The current pipeline supports Phase3A inspection, per-frame graph
construction, graph feature extraction, clustering, consensus network
construction, cluster interpretation, and figure generation.

## Notes

- `data/` and `results/` are ignored by Git.
- Generated tables and figures are local artifacts.
- Full centrality calculation over all frame graphs can be expensive.
- Centrality-based biological interpretation is mainly computed on consensus graphs.
- Phase3B outputs are optional reference materials and are not required for the main pipeline.
