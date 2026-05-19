# NaPi2b VKR Graph Clustering

This repository contains the Python pipeline for MSc thesis work on clustering
amino acid residue interaction networks to identify conformational states of the
NaPi2b transporter.

Each trajectory frame is represented as a protein residue contact graph:
residues are nodes, residue-residue contacts are edges, and graph-level
features are used for clustering and interpretation.

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

For lightweight Phase3A checks without loading the full parquet data:

```python
from napi2b_vkr.io import inspect_phase3a_dataset

report = inspect_phase3a_dataset("data/phase3A/phase3aexports", "normal")
report["parquet"]["schema"]
```

Or run the diagnostic script directly:

```bash
python3 scripts/inspect_phase3a.py --base-dir data/phase3A/phase3aexports --condition normal
```

Full graph construction, clustering, and centrality calculations may be
expensive. Run them deliberately after the data diagnostics pass.

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

Current setup work focuses on project structure and Phase3A diagnostics. The
later notebooks will save frame features, clustering labels and scores,
consensus graph summaries, centrality tables, and figures under `results/`.
