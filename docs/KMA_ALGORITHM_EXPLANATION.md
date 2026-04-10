# KMA Algorithm: Complete Explanation

This document explains how the KMA algorithm works in this repository. It covers the full pipeline for both undirected and directed graphs, including kernelization, forced vertices, the memetic refinement step, and the GNN-guided hybrid variants.

## 1. Problem Definition

KMA solves the Feedback Vertex Set (FVS) problem:

- Given a graph, find a set of vertices whose removal leaves the graph acyclic.
- For undirected graphs this is the FVS problem.
- For directed graphs this is the Directed Feedback Vertex Set (DFVS) problem.

`KMA` stands for Kernelized Memetic Algorithm:

- Kernelized: first reduce the graph by removing vertices that can be handled by safe reduction rules.
- Memetic Algorithm: then apply a heuristic evolutionary search to solve the reduced kernel.

KMA in this repository is implemented in `experiments/run_hybrid.py` and uses the C++ backend in `cpp_engine` for the heavy lifting.

## 2. High-Level KMA Pipeline

The KMA pipeline has three main phases:

1. Kernelization
2. Memetic Algorithm (MA) refinement
3. Solution reconstruction

For the hybrid variants (`GNN-KMA`, `GNN-KMA-2`, and `GNN-KMA-3`) there is an extra GNN inference phase between kernelization and MA.

### 2.1 Kernelization

The kernelization phase simplifies the graph while preserving the optimal FVS structure.

- If a vertex is forced into the FVS by an obvious rule, it is removed and recorded.
- If a vertex can be removed without changing the optimal solution, it is removed and the graph is rewired.
- The remaining graph is the kernel.

It returns:

- `k_n`: number of vertices in the kernel
- `k_edges`: edges of the kernel
- `forced`: original vertices that were forced into the solution
- `k_new_to_old`: mapping from kernel vertex index back to original vertex index

### 2.2 MA Refinement

The kernel is solved by a memetic algorithm implemented in C++.

- If the C++ engine exposes `solve_*_KMA`, KMA is used.
- Otherwise it falls back to `solve_*_KME` or `solve_*_MA`.
- The solver is called with population size, generation limit, early-stop, and a timeout.

For undirected graphs the function is `kma_solve_undirected`; for directed graphs it is `kma_solve_directed`.

### 2.3 Solution Reconstruction

The final solution is:

- all forced vertices from kernelization,
- plus the mapped vertices returned by the kernel MA solver.

This is returned as a sorted, deduplicated vertex set in the original graph indexing.

## 3. Kernelization Rules

The implementation contains separate rules for undirected and directed graphs.

### 3.1 Undirected Kernelization

The function `kernelize_undirected_graph(n, edges)` applies these rules:

- Self-loop rule:
  - If a vertex has an edge to itself, it is forced into the FVS.
- Degree-0 and degree-1 rule:
  - If a vertex has degree 0 or 1, it is removed from the graph because it cannot be part of a cycle.
- Degree-2 bypass rule:
  - If a vertex `v` has exactly two active neighbors `a` and `b`, and `a` and `b` are not already adjacent,
    then `v` can be removed and an edge between `a` and `b` is added.

During kernelization, the code repeatedly applies these rules until no further changes occur.

### 3.2 Directed Kernelization

The function `kernelize_directed_graph(n, edges)` uses stronger directed reduction rules:

- Self-loop rule:
  - If a vertex has a self-loop, it is forced into the DFVS.
- Source/Sink rule:
  - If a vertex has zero in-degree or zero out-degree, it is removed because it cannot be part of a directed cycle.
- Degree-1 bypass:
  - If a vertex has in-degree 1 or out-degree 1, it can be removed while connecting its predecessor(s) to its successor(s), preserving cycles.
- SCC reduction:
  - After these removals, only vertices that belong to nontrivial strongly connected components remain.
  - A nontrivial SCC is size > 1 or a self-loop.

This directed kernelization is designed to preserve all cycles that matter to the DFVS objective.

## 4. KMA Implementation Details

### 4.1 Core functions

In `experiments/run_hybrid.py` the core functions are:

- `kma_solve_undirected(...)`
- `kma_solve_directed(...)`

Both functions:

1. Run kernelization.
2. If the kernel is empty, return forced vertices.
3. Otherwise call the C++ KMA solver on the kernel.
4. Map kernel vertex indices back to the original graph.
5. Return the union of forced and mapped KMA vertices.

### 4.2 C++ solver dispatch

The code dispatches to the best available solver:

- `cpp_engine.solve_undirected_KMA`
- `cpp_engine.solve_undirected_KME`
- `cpp_engine.solve_undirected_MA`
- `cpp_engine.solve_directed_KMA`
- `cpp_engine.solve_directed_KME`
- `cpp_engine.solve_directed_MA`

If the dedicated `KMA` solver is unavailable, it uses the next capable fallback.

### 4.3 Timeout handling

- `kernelization` is timed but not interrupted.
- `max_time_seconds` is enforced inside the C++ solver during the MA stage.
- The solver returns the best solution found within the time budget.

### 4.4 Heuristic nature

KMA is heuristic for both undirected and directed graphs.

- The kernelization step is exact and safe.
- The MA solver is not guaranteed optimal, but it is effective on large kernels.
- The approach is practical for benchmark-heavy evaluation and hybrid GNN coupling.

## 5. GNN-KMA Hybrid Strategy

The repository also implements GNN-guided versions of KMA.

### 5.1 Why hybridize with a GNN?

- The GNN learns vertex-level patterns from solved instances.
- It predicts which vertices are most likely to belong to the FVS.
- Those predictions are used to bias the KMA solver toward better solutions.

### 5.2 Soft-hint coupling

The current hybrid design is precision-first and safe.

- The GNN produces per-vertex probabilities.
- Only vertices with probability above a threshold are hard-fixed.
- Hard-fixing is limited to at most 8% of the kernel.
- If no vertices are confident enough, the system runs pure KMA.

This avoids locking in too many false positives, which would worsen the solution.

### 5.3 Hybrid solver flow

For `GNN-KMA` and `GNN-KMA-2`:

1. Kernelize the original graph.
2. Run the GNN on the kernel and compute vertex probabilities.
3. Select high-confidence candidates with `_pick_gnn_candidates_from_probs`.
4. Remove those candidates from the kernel before KMA.
5. Run KMA on the reduced kernel.
6. Reconstruct solution as:
   - forced vertices from kernelization,
   - hard-fixed GNN vertices,
   - KMA output vertices mapped back to original indices.

For `GNN-KMA-3`, the same solver structure is used but with a deeper, attention-based model and richer features.

### 5.4 GNN models used

The solver supports three families of GNN models:

- `GNN-KMA` / `v1`: baseline GNN with 3-channel features.
- `GNN-KMA-2` / `v2`: improved structural features with RWSE, motif counts, and k-core data.
- `GNN-KMA-3` / `v3`: research-grade attention-based GNN with 16-channel features.

Each variant is loaded lazily and falls back gracefully if PyTorch, torch_geometric, or weights are unavailable.

### 5.5 Candidate selection

Candidate selection is handled by `_pick_gnn_candidates_from_probs(...)`.

- `threshold` defaults to 0.65.
- `min_fraction` is 0.5% of the kernel.
- `max_fraction` is 8% of the kernel.
- If too few vertices exceed the threshold, the function returns an empty set.

The precise logic is:

- If selected vertices > `max_k`, keep only the top `max_k` by confidence.
- If selected vertices < `min_k`, return an empty set.
- This ensures the GNN only hard-fixes when it is sufficiently confident.

### 5.6 Why the hybrid design works better

- Hard-fixing only very high-confidence vertices reduces the risk of bad decisions.
- The solver still has full freedom on the remaining kernel.
- This is a conservative use of learned information, preserving the KMA solver's strengths.

## 6. Running the Solver

The benchmark scripts and solver wrappers provide the main entry points.

### 6.1 Direct KMA

For directed graphs:

```bash
python experiments/benchmark_directed.py --algo KMA --test <graph-or-folder> --pop 20 --gens 100 --timeout 600
```

For undirected graphs:

```bash
python experiments/benchmark_undirected.py --algo KMA --test <graph-or-folder> --pop 20 --gens 100 --timeout 600
```

### 6.2 GNN-KMA

For directed graphs:

```bash
python experiments/benchmark_directed.py --algo GNN-KMA --test <graph-or-folder> --pop 20 --gens 100 --timeout 600 --gnn-threshold 0.2
```

For undirected graphs:

```bash
python experiments/benchmark_undirected.py --algo GNN-KMA --test <graph-or-folder> --pop 20 --gens 100 --timeout 600 --gnn-threshold 0.2
```

### 6.3 Parameters summary

- `--pop`: population size for MA/KMA and hybrid variants.
- `--gens`: maximum generations for MA/KMA.
- `--timeout`: hard wall-clock time limit for MA/KMA.
- `--gnn-threshold`: confidence threshold for GNN hard-fixing.
- `--gnn-hidden`: optional hidden dimension override when loading model weights.

## 7. Key Implementation Files

- `experiments/run_hybrid.py` — KMA solver wrappers, kernelization, GNN inference, and hybrid integration.
- `experiments/benchmark_directed.py` — benchmark driver for directed algorithms.
- `experiments/benchmark_undirected.py` — benchmark driver for undirected algorithms.
- `gnn_model/model_undirected.py` / `model_directed.py` — baseline GNN architecture.
- `gnn_model/model_undirected_v2.py` / `model_directed_v2.py` — v2 GNN architecture.
- `gnn_model/model_directed_v3.py` — v3 attention-based architecture.
- `gnn_model/feature_engineering_v2.py` / `feature_engineering_v3.py` — feature engineering for GNN training.
- `cpp_engine/` — performance-critical solver implementations.

## 8. Notes and Best Practices

- KMA itself is a strong heuristic baseline.
- The hybrid GNN-KMA variants are designed to improve solution quality without sacrificing safety.
- The current repository uses a conservative hard-fix threshold to avoid degrading pure KMA performance.
- If GNN weights are unavailable, the system automatically falls back to pure KMA.
- The solver is most effective when `cpp_engine` is compiled and available.

## 9. Recommended Reading

- `README.md` for project overview and benchmarking setup.
- `GNN_KMA_MODELS.md` for complete model and workflow explanation.
- `GNN_KMA_PROCESS_DETAILED.md` for the full GNN-KMA pipeline reference.

---

This file is intended to be the complete technical reference for how KMA and its GNN-guided variants work in this repository.
