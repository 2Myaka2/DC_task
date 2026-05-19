# Project instructions for Codex

## Project goal

This repository contains code for MSc thesis work:

"Clustering amino acid residue interaction networks to identify conformational states of the NaPi2b transporter."

Each trajectory frame is represented as a protein residue contact graph:
- nodes are amino acid residues;
- edges are residue-residue contacts;
- frames are clustered based on graph topology;
- clusters are interpreted as conformational states.

## Data policy

Do not commit raw data or generated results.

Ignored local directories:
- data/
- results/

Raw files are expected locally at:

data/phase3A/phase3aexports/

Important files:
- phase3a_manifest.json
- dataset_manifest.json
- residue_table_normal.csv
- residue_table_tumor.csv
- contact_edges_perframe_normal.parquet
- contact_edges_perframe_tumor.parquet
- protein_contact_edges_undirected_normal.csv
- protein_contact_edges_undirected_tumor.csv
- pos3d_normal.json
- pos3d_tumor.json
- normal_validation.json
- tumor_validation.json

Phase3B outputs are optional reference material and are expected locally at:

data/phase3B/phase3boutputs/

## Required source structure

Use this structure:

src/napi2b_vkr/
  __init__.py
  io.py
  graph_builder.py
  features.py
  clustering.py
  consensus.py
  plots.py

## Main pipeline

1. Inspect Phase3A files.
2. Build per-frame NetworkX graphs.
3. Compute graph-level features for every frame.
4. Cluster frames using KMeans and AgglomerativeClustering.
5. Select number of clusters using silhouette, Davies-Bouldin, and Calinski-Harabasz scores.
6. Build consensus graphs for clusters.
7. Interpret clusters using graph metrics, centralities, residues, and regions.
8. Save tables to results/tables/.
9. Save figures to results/figures/.

## Code requirements

- Use pathlib.Path.
- Use pandas, numpy, networkx, scikit-learn, scipy, matplotlib, seaborn, pyarrow, tqdm.
- Code should work for condition="normal" and condition="tumor".
- Do not hard-code absolute paths.
- Prefer functions over notebook-only code.
- Notebooks should call functions from src/napi2b_vkr.
- Handle unknown column names with diagnostics and clear errors.
- Never commit data files, parquet files, pt files, pkl files, npy files, html reports, pdf reports, or generated figures from data/archive.

## Expected notebooks

notebooks/01_data_inspection.ipynb
notebooks/02_build_frame_graphs.ipynb
notebooks/03_graph_features.ipynb
notebooks/04_clustering.ipynb
notebooks/05_cluster_interpretation.ipynb

Each notebook should be runnable from top to bottom.