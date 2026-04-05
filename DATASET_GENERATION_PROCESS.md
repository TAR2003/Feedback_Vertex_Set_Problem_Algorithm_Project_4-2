# Dataset Generation Process

This document explains the full data generation pipeline implemented in this repository, with a focus on:

1. `data/setup_benchmark_inputs.py`
2. `data/download_real_world.py`
3. `data/generate_synthetic.py`

It also describes how these scripts are consumed by GNN dataset generation (`gnn_model/dataset_gen.py`).

---

## 1) End-to-End Entry Point: `setup_benchmark_inputs.py`

`setup_benchmark_inputs.py` is the orchestration script that builds the benchmark inputs in one command.

### Primary responsibilities

- Optionally clean old data artifacts under `data/`.
- Generate real-world category buckets by calling `download_real_world.py`.
- Generate synthetic category buckets by calling `generate_synthetic.py`.
- Keep the pipeline reproducible through explicit seed forwarding.

### Important defaults

- `--total-undirected 100000`
- `--total-directed 100000`
- `--exact-ratio 0.5`
- `--seed 1337`
- `--family all`

### Cleaning policy

Before generation (unless `--no-clean` is passed), the script removes data items in `data/` while preserving:

- `data/pace2022`
- `data/__pycache__`
- script and docs files (`.py`, `.md`, `.gitkeep`)

This gives deterministic, clean dataset reconstruction while protecting source assets and PACE data.

### Execution model

The script launches two subprocesses in order:

1. `download_real_world.py` with shared totals/split/seed/family
2. `generate_synthetic.py` with the same configuration

The order ensures the real-world buckets are populated first, then synthetic categories fill the rest.

---

## 2) Real-World Bucket Builder: `download_real_world.py`

This script populates only the real-world category portions of the benchmark layout:

- `data/synthetic/undirected/exact_track/real_world/`
- `data/synthetic/undirected/heuristic_track/real_world/`
- `data/synthetic/directed/exact_track/real_world_ego/`
- `data/synthetic/directed/heuristic_track/real_world_ego/`

### Real-world allocation ratios

The script reserves fixed real-world fractions from global totals:

- Undirected real-world share: `20%`
- Directed real-world share: `30%`

Then each share is split by `--exact-ratio` into exact and heuristic tracks.

### Data sources and loader strategy

The loader is robust and best-effort. It attempts the following source families and degrades gracefully when packages or network are unavailable.

- NetworkX built-ins
- PyTorch Geometric datasets (Planetoid, Amazon, Coauthor, TU)
- Open Graph Benchmark (`ogbn-arxiv`)
- SNAP archives via URL download

A local cache is maintained under `.cache/real_graphs` to avoid repeated download costs.

### Normalization and slicing

Every loaded graph is normalized before writing:

- Largest connected component / weakly connected component extraction
- Node relabeling to compact integer IDs
- Optional BFS-induced sampling to match target size bands

Target size policy per track:

- Exact track: small graphs (`~10..35` nodes)
- Heuristic track: larger graphs (`~100..5000` nodes)

### Output format

Each output file is in benchmark edge-list format:

- `# format: edge_list_v1`
- `# directed: 0|1`
- `# source: <source-tag>`
- `p edge N M`
- then edge list `u v`

The `# source` metadata is especially useful for provenance and debugging.

### Fallback behavior

If no real dataset can be loaded, the script generates proxy graphs (BA/GN-style) so the pipeline remains operational.

---

## 3) Synthetic Category Builder: `generate_synthetic.py`

This script generates all category buckets (including synthetic proxies for real-world categories), split into exact and heuristic tracks.

### Output layout

`data/synthetic/<family>/<track>/<category>/*.txt`

Families:

- `undirected`
- `directed`

Tracks:

- `exact_track`
- `heuristic_track`

### Category weights

Undirected categories are distributed equally (20% each):

- `real_world`
- `scale_free`
- `small_world`
- `random_er`
- `grids_trees`

Directed categories use asymmetric distribution:

- `real_world_ego`: 30%
- `scale_free`: 20%
- `random_er`: 20%
- `directed_grids`: 15%
- `dags`: 15%

### Exact/heuristic split logic

For each category count, the script computes:

- `exact = int(total * exact_ratio)`
- `heuristic = total - exact`

This preserves category proportions while enforcing track split.

### Generators by category

The script uses category-specific graph generators and track-dependent size ranges.

Examples:

- Undirected scale-free: Barabasi-Albert
- Undirected small-world: Watts-Strogatz
- Undirected random: Erdos-Renyi
- Undirected grids/trees: alternating grid and random tree
- Directed scale-free: directed projection from scale-free multigraph
- Directed grids: oriented grid with occasional reverse edge injection
- Directed DAGs: GN model

### Reproducibility

Each bucket receives a deterministic derived seed:

- Global seed + deterministic hash of `family:track:category`

This allows exact regeneration of each bucket.

### Regeneration controls

- `--force` deletes existing `.txt` files in each bucket before regeneration
- Without `--force`, existing files are reused and only missing files are created

---

## 4) Track Semantics and Intended Solver Usage

The two-track design is solver-aware.

### Exact track

Purpose:

- Smaller graphs for exact algorithms and exhaustive comparison

Algorithms commonly run:

- BST
- IC
- MA
- KMA
- GNN-KMA variants

### Heuristic track

Purpose:

- Larger graphs intended for scalable heuristic/hybrid methods

Algorithms commonly run:

- MA
- KMA
- DKMA
- GNN-KMA family

---

## 5) Integration with GNN Dataset Creation (`gnn_model/dataset_gen.py`)

The benchmark generators produce `.txt` graph corpora. `gnn_model/dataset_gen.py` consumes those files and produces `.pt` graph objects.

### Labeling policy

- `exact_track`: labels from IC (exact) with timeout control
- `heuristic_track`: labels from KMA with MA-stage timeout and best-so-far return

### Additional safety features

- Subprocess timeout enforcement per graph for robust generation
- Invalid-solution verification and CSV status logging (`completed`, `timeout`, `invalid`)
- Track-aware progress accounting

This design keeps model training data aligned with benchmark data distribution and solver behavior.

---

## 6) Operational Recommendations

- Use `setup_benchmark_inputs.py` as the canonical generation entrypoint.
- Keep a fixed `--seed` for reproducible experiments.
- Use `--family` and reduced totals for quick smoke validation.
- Use `--force` only when intentional full refresh is needed.
- Preserve the same split logic between benchmark runs and GNN dataset generation to avoid train-test distribution drift.
