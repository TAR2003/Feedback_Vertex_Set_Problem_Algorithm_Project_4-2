# KMA Implementation Procedure

This document explains how KMA (Kernelized Memetic Algorithm) is implemented and executed in this repository.

Primary implementation locations:

- C++ undirected: `cpp_engine/src/undirected_algos/memetic_u.cpp`
- C++ directed: `cpp_engine/src/directed_algos/memetic_d.cpp`
- Python hybrid wrappers: `experiments/run_hybrid.py`
- Benchmark integration:
  - `experiments/benchmark_undirected.py`
  - `experiments/benchmark_directed.py`

---

## 1) Conceptual Pipeline

KMA combines two ideas:

1. Kernelization/reduction to shrink the graph while preserving optimality structure.
2. Memetic search (MA) on the reduced kernel graph.

Final solution is:

- forced vertices from reductions
- plus mapped kernel solution from MA/KMA

This gives better scalability than running MA directly on the full graph.

---

## 2) C++ KMA Path

Both undirected and directed C++ implementations follow the same overall structure.

## Undirected KMA (`solve_undirected_KMA`)

### Steps

1. Build graph from input edge list.
2. Run undirected kernelization (`kernelize_undirected`).
3. Record forced vertices selected by reduction rules.
4. Build mapping:
   - original vertex -> kernel index
   - kernel index -> original vertex
5. Create kernel edge list from active vertices.
6. Run MA on kernel graph (`solve_undirected_MA`).
7. Map kernel solution back to original IDs.
8. Union with forced set and deduplicate.

## Directed KMA (`solve_directed_KMA`)

The directed variant mirrors the undirected flow with directed graph structures and directed kernelization.

### Directed specifics

- Uses directed adjacency (`out_adj`).
- Uses directed kernel rules via `kernelize_directed`.
- Kernel MA stage calls `solve_directed_MA`.

Both implementations provide alias compatibility (`KME` aliases to `KMA`).

---

## 3) Python KMA Wrappers (`run_hybrid.py`)

`run_hybrid.py` provides repository-level KMA APIs that:

- compute timing by stage
- handle fallback if a specific binding is unavailable
- expose diagnostics for benchmark CSV integration

Key functions:

- `kma_solve_undirected(...)`
- `kma_solve_directed(...)`

### Stage accounting

Wrappers explicitly track:

- `kernelization_ms`
- `ma_ms`

This makes KMA behavior inspectable across benchmark runs.

### Fallback hierarchy

For both families, wrappers prefer:

1. native `solve_*_KMA`
2. legacy alias `solve_*_KME`
3. `solve_*_MA` as last resort

---

## 4) Timeout and Early-Stop Semantics

KMA in this codebase is configured through:

- `pop_size`
- `max_gens`
- `early_stop` (patience)
- `max_time_seconds` (hard bound for MA stage)

Important behavior:

- Kernelization runs before MA-stage timeout budget dominates runtime.
- Heuristic wrappers in benchmarks keep best-so-far behavior rather than discarding partial progress.

---

## 5) CLI and Benchmark Integration

KMA is exposed in both directed and undirected benchmark scripts and can be run:

- standalone (`--algo KMA`)
- as part of `ALL`
- as part of suite/pipeline scripts

Result CSVs record per-instance outcomes, including validity and runtime.

---

## 6) Why KMA Works Well Here

KMA is effective in this repository because:

- reduction removes many irrelevant vertices before expensive search,
- MA explores nontrivial combinations on a smaller kernel,
- final mapping guarantees compatibility with original graph indexing,
- solver wrappers unify behavior across C++ and Python orchestration.

In practice, this is the baseline hybrid-ready solver used by GNN-guided variants and by DKMA components.
