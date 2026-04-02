# FVS Benchmark Suite

This repository supports two-track benchmarking for Feedback Vertex Set (undirected + directed):

- exact_track: small graphs, run all algorithms
- heuristic_track: large graphs, run heuristic algorithms only

## Quick Start for

### 1) Build/download all benchmark inputs

```bash
python data/setup_benchmark_inputs.py --total-undirected 100000 --total-directed 100000
```

For a quick smoke run:

```bash
python data/setup_benchmark_inputs.py --total-undirected 100 --total-directed 100
```

Notes:

- The setup command preserves `data/pace2022`.
- Data is generated under `data/synthetic/` in family/track/category folders.

### 2) Run the full benchmark suite in one command

```bash
python experiments/run_benchmark_suite.py
```

Default suite behavior (`--profile requested`):

- Directed exact_track: `BST`, `IC`, `MA`, `KMA`, `GNN-KMA`
- Undirected heuristic_track: `MA`, `KMA`, `GNN-KMA`

Run every family+track combination:

```bash
python experiments/run_benchmark_suite.py --profile full
```

Full profile behavior:

- Undirected + directed exact_track: `BST`, `IC`, `MA`, `KMA`, `GNN-KMA`
- Undirected + directed heuristic_track: `MA`, `KMA`, `GNN-KMA`

## Resume And Skip Logic

The suite writes one CSV per `(family, track, algorithm)` in:

- `results/suite/`

Before running each algorithm, it checks the target CSV. If a file path already exists there, that file is skipped for that specific algorithm.

Example:

- If `test_001.txt` already has an `IC` row in `undirected_exact_track_IC.csv`, IC is not run again for that file.

Force rerun all files:

```bash
python experiments/run_benchmark_suite.py --rerun
```

## Useful Commands

Run only directed tasks:

```bash
python experiments/run_benchmark_suite.py --mode directed
```

Run with smaller evolutionary budget:

```bash
python experiments/run_benchmark_suite.py --pop 20 --gens 50 --quiet

# Tiny smoke run (1 file per task)
python experiments/run_benchmark_suite.py --max-files 1 --quiet
```

Run only heuristic tracks quickly (existing helper):

```bash
python experiments/run_heuristics_track.py
```

## CSV Output Schema

Suite CSV rows contain:

`file, file_path, n, m, family, track, algorithm, fvs_size, runtime_ms, valid, status, executed_at`

This guarantees each row has input file identity plus vertex/edge counts, runtime, and FVS size.

## GNN Dataset And Training Pipeline

Use this flow to generate `.pt` datasets with the same benchmark-style distribution and then train GNN weights.

### 1) Generate GNN training data (`.pt`)

```bash
# Full-size example
python gnn_model/dataset_gen.py --total-undirected 100000 --total-directed 100000

# Quick smoke generation
python gnn_model/dataset_gen.py --total-undirected 100 --total-directed 100

# Live-progress mode (recommended for long runs)
python -u gnn_model/dataset_gen.py \
	--total-undirected 100000 --total-directed 100000 \
	--progress-every 5 --max-nodes 300 --solver-mode ma
```

Distribution mirrors benchmark generators:

- Undirected: `real_world` 20%, plus `scale_free/small_world/random_er/grids_trees` at 20% each.
- Directed: `real_world_ego` 30%, `scale_free` 20%, `random_er` 20%, `directed_grids` 15%, `dags` 15%.
- Each category is split by `--exact-ratio` (default `0.5`) into `exact_track` and `heuristic_track`.

Generated files are saved under:

- `gnn_model/datasets/pt/undirected/.../*.pt`
- `gnn_model/datasets/pt/directed/.../*.pt`

Notes:

- `--progress-every` prints frequent progress updates per bucket.
- `--solver-mode ma` avoids long exact-label bottlenecks during large runs.
- `--max-nodes` caps graph size during generation to keep runtime predictable.

### 2) Train GNN models from generated `.pt`

```bash
# Train both models
python gnn_model/train.py --type both --epochs 100 --lr 0.001

# Train with custom validation split and model size
python gnn_model/train.py --type both --epochs 300 --hidden 128 --dropout 0.2 --val-ratio 0.2
```

Training loads all `.pt` files recursively from `gnn_model/datasets/pt/<type>/`.

Weights are saved to:

- `gnn_model/weights/undirected_fvs_gcn.pt`
- `gnn_model/weights/directed_fvs_gcn.pt`
