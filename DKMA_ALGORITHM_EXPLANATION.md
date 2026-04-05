# DKMA Algorithm: Dynamic Kernelized Memetic Algorithm

This document describes DKMA and GNN-DKMA in this repository.

## 1. Core Idea

KMA applies kernelization once before the memetic algorithm (MA).
DKMA interleaves reduction and search repeatedly:

1. Evolve population for one MA generation.
2. Commit consensus vertices that appear in a threshold fraction of individuals.
3. Force committed vertices into the solution.
4. Re-kernelize the residual graph.
5. Continue MA on the smaller kernel.

This dynamic loop enables the opening-up effect where new reductions become available only after partial commitments.

References:
- [PACE22] Grossmann, Heuer, Schulz, Strash, IPEC 2022, doi:10.4230/LIPIcs.IPEC.2022.26
- [ESA22] Figiel, Froese, Nichterlein, Niedermeier, ESA 2022, doi:10.4230/LIPIcs.ESA.2022.53
- [LANGEDAL] DFVS PACE 2022 heuristic solver, doi:10.5281/zenodo.6630611

## 2. DKMA Pipeline

Top-level functions:
- `dkma_solve_undirected(...)`
- `dkma_solve_directed(...)`

Main stages:

1. Static kernelization (`kernelize_*_graph`).
2. Directed-only SHORTONE contractions (`_apply_shortone_rule`).
3. Initial population generation on the kernel.
4. Dynamic MA loop:
   - `_run_one_generation_ma`
   - `_commit_vertices_from_population`
   - `_dynamic_kernelize`
   - `_diversify_with_topological_ordering` (every 5 generations)
5. Post-loop gain local search (`_gain_local_search`) with 1-1 and 2-1 swaps.
6. Final validation with `_is_acyclic` on original graph, fallback to KMA if invalid.

## 3. Key Helpers

- `_commit_vertices_from_population(population, n_kernel, threshold)`:
  selects consensus vertices; caps commitment to top 30% by frequency if over-committing.

- `_dynamic_kernelize(...)`:
  removes committed vertices, runs kernelization again on residual graph, and updates index maps.

- `_is_acyclic(n, edges, removed_vertices, directed)`:
  directed uses iterative DFS back-edge detection; undirected uses union-find cycle detection.

- `_diversify_with_topological_ordering(...)`:
  directed uses randomized Kahn tie-breaking and cycle-breaking insertion.

- `_apply_shortone_rule(k_n, k_edges, directed)`:
  directed contraction pass inspired by Mount-Doom SHORTONE style reductions.

## 4. GNN-DKMA

Top-level functions:
- `gnn_dkma_solve_undirected(...)`
- `gnn_dkma_solve_directed(...)`

Flow:

1. Kernelize input graph.
2. Run GNN inference and hard-fix high-confidence vertices (`p >= gnn_threshold`).
3. Solve remaining kernel with DKMA.
4. Map all vertices back to original indexing and merge with forced vertices.

## 5. CLI Integration

Both benchmark drivers now support:

- `--algo DKMA`
- `--algo GNN-DKMA`
- `--commit-threshold`
- `--dynkern-every`
- `--no-gain-search`
- `--no-diversify`

Per-instance logging includes:

- `initial_kernel_size`
- `final_kernel_size`
- `n_dynamic_reductions`
- `solution_size`
- `time_seconds`

## 6. Correctness Safeguards

DKMA enforces:

- deduplicated final solution indices,
- forced vertices from static and dynamic reductions included,
- final acyclicity verification on original graph,
- fallback to KMA if validation fails.
