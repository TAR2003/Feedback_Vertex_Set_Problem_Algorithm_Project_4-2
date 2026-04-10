# Dataset Generation Process (Complete Technical Guide)

This document explains the repository's complete data-generation system in implementation order, with runtime behavior and operational intent.

Primary scripts:

1. data/setup_benchmark_inputs.py (orchestrator)
2. data/download_real_world.py (real-world bucket constructor)
3. data/generate_synthetic.py (synthetic bucket constructor)
4. gnn_model/dataset_gen.py (TXT-to-PT with solver labels)

## 1. Design Goals

The generation pipeline is designed around five goals:

1. Controlled distribution: fixed category proportions by family.
2. Track-aware difficulty: exact_track and heuristic_track are constructed separately.
3. Reproducibility: deterministic allocation and seed handling.
4. Restartability: safe reuse of existing files when regeneration is not required.
5. Benchmark/train alignment: PT datasets inherit the same family/track/category structure.

## 2. Output Contracts and Directory Layout

Benchmark graph corpus contract:

- data/synthetic/{family}/{track}/{category}/*.txt

Family values:

- undirected
- directed

Track values:

- exact_track
- heuristic_track

The downstream benchmark and PT pipelines assume this contract; changing it requires coordinated script updates.

## 3. Orchestration Layer: setup_benchmark_inputs.py

setup_benchmark_inputs.py is the entrypoint used for complete benchmark input preparation.

### 3.1 CLI and defaults

Key defaults:

- total-undirected: 100000
- total-directed: 100000
- exact-ratio: 0.5
- seed: 1337
- family: all

### 3.2 Cleaning behavior

Unless no-clean is specified, the script removes generated items in data while preserving:

- data/pace2022
- data/__pycache__
- source/document files with .py, .md, .gitkeep

This ensures a reproducible clean rebuild while protecting static assets.

### 3.3 Execution graph

The orchestrator performs two subprocess calls in order:

1. download_real_world.py
2. generate_synthetic.py

Both receive synchronized family, total counts, exact-ratio, and seed.

Operationally, this means real-world buckets are prepared first, then synthetic bucket generation completes all category plans.

## 4. Real-World Bucket Constructor: download_real_world.py

This script populates only the real-world categories:

- undirected/*/real_world
- directed/*/real_world_ego

### 4.1 Real-world budget allocation

Real-world slices are fixed fractions of family totals:

- undirected real-world share = 20%
- directed real-world share = 30%

Each share is split into exact and heuristic counts using exact-ratio.

### 4.2 Source loading strategy (best-effort)

The loader attempts multiple source classes in sequence and continues gracefully if some are unavailable.

Source families:

- NetworkX built-ins
- PyTorch Geometric datasets
- OGB node property datasets
- SNAP archives

Artifacts are cached under .cache/real_graphs to reduce repeated network/download costs.

### 4.3 Normalization and slicing rules

Before write, each candidate graph is normalized:

1. extract largest component (connected/weakly connected)
2. relabel to compact integer IDs
3. sample to track-specific target size if required

Track size intent:

- exact_track: small instances (roughly 10-35 nodes)
- heuristic_track: larger instances (roughly 100-5000 nodes)

### 4.4 Write format

Each .txt follows edge_list_v1 with metadata:

- format marker
- directed bit
- source tag
- p edge N M header
- normalized edge list

The source tag is useful when auditing provenance and debugging quality drift.

### 4.5 Degradation path

If external datasets cannot be loaded, the script emits proxy real-world-like graphs so the generation process remains operational.

## 5. Synthetic Constructor: generate_synthetic.py

This script generates all categories for both tracks and families according to weighted plans.

### 5.1 Category weights

Undirected categories are uniformly weighted (0.20 each):

- real_world
- scale_free
- small_world
- random_er
- grids_trees

Directed categories are nonuniform:

- real_world_ego 0.30
- scale_free 0.20
- random_er 0.20
- directed_grids 0.15
- dags 0.15

### 5.2 Count allocation algorithm

The script uses weighted integer allocation with remainder distribution by largest fractional part.

For each category:

- exact = floor(cat_total * exact_ratio)
- heuristic = cat_total - exact

This guarantees exact + heuristic = category total while preserving global proportions.

### 5.3 Graph model implementations

Examples by category and family:

- scale_free: Barabasi-Albert (undirected) / directed scale-free projection
- small_world: Watts-Strogatz
- random_er: Erdos-Renyi (directed and undirected variants)
- grids_trees: alternating grid and random tree construction
- directed_grids: oriented grid with occasional reverse-edge injection
- dags: GN DAG-style generator

The track controls graph-size ranges, not only algorithm labels.

### 5.4 Regeneration semantics

- force: removes bucket .txt files first, then regenerates
- non-force: existing files are reused and only missing files are generated

This behavior is crucial for long-running workflows and interrupted runs.

### 5.5 Deterministic seeding

Each bucket receives derived seed material from:

- global seed
- deterministic bucket key (family:track:category)

This makes bucket-level generation reproducible.

## 6. Graph File Compatibility Considerations

The generated TXT files are consumed by benchmark parsers in experiments/benchmark_undirected.py and experiments/benchmark_directed.py.

Compatibility properties preserved by generation:

- compact index space
- no self-loop emission in normalized edge list path
- consistent edge semantics by directed flag

## 7. Runtime Characteristics and Bottlenecks

Most expensive stages in practice:

1. external dataset loading/downloading in download_real_world.py
2. large-bucket synthetic generation under high totals
3. PT labeling stage in gnn_model/dataset_gen.py (solver calls dominate)

Operational recommendations:

- run small smoke totals first
- keep fixed seed during comparative experiments
- use force only when full replacement is required

## 8. PT Dataset Generation Integration (gnn_model/dataset_gen.py)

dataset_gen.py consumes the generated TXT hierarchy and produces PyG Data objects.

### 8.1 Label source policy

- exact_track labels: IC path (with timeout guard)
- heuristic_track labels: KMA path (with MA-stage timeout)

### 8.2 Reliability guards

- solver isolation via subprocess timeout where needed
- explicit FVS validity checks before save
- CSV status ledger for completed/timeout/invalid outcomes
- track/family/category metadata retained per sample

### 8.3 Variant output roots

- v1: gnn_model/datasets/pt
- v2: gnn_model/datasets/pt_v2
- v3: gnn_model/datasets/pt_v3

## 9. Practical Command Patterns

Full benchmark input build:

```bash
python data/setup_benchmark_inputs.py --total-undirected 100000 --total-directed 100000 --exact-ratio 0.5 --seed 1337
```

Quick smoke build:

```bash
python data/setup_benchmark_inputs.py --total-undirected 200 --total-directed 200 --no-clean
```

Family-specific generation:

```bash
python data/setup_benchmark_inputs.py --family directed --total-directed 5000 --total-undirected 0
```

## 10. Failure Modes and Troubleshooting

Typical issues and causes:

1. Empty real-world buckets: external dataset package/network unavailable.
2. Fewer files than requested: source caps and per-bucket availability constraints.
3. Slow generation: large totals + expensive categories + cold cache.
4. PT invalid records: solver timeout or invalid solution rejected by validator.

High-confidence recovery sequence:

1. run tiny totals to verify pipeline health
2. inspect bucket counts under data/synthetic
3. regenerate specific families with force only if needed
4. run dataset_gen with reduced timeout and small totals to validate label path

## 11. Summary

The repository implements a production-style research data pipeline:

- deterministic weighted planning,
- separate real-world and synthetic construction,
- strict track semantics,
- robust restart behavior,
- direct compatibility with benchmark and GNN training workflows.

If you understand the four scripts in this document, you understand the entire data lifecycle from raw graph construction to model-ready PT data.
