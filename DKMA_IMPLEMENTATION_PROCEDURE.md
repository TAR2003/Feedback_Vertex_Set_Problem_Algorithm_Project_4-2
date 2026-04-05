# DKMA Implementation Procedure (Full Runtime and Algorithm Guide)

This document covers the complete DKMA implementation as it exists in the repository, including control flow, helper mechanics, runtime safeguards, and practical tuning.

Main implementation file:

- experiments/run_hybrid.py

Public entry points:

- dkma_solve_undirected(...)
- dkma_solve_directed(...)

Shared engine:

- _dkma_solve_common(...)

## 1. DKMA Motivation and Position in Solver Stack

KMA performs one reduction pass and then MA search on the resulting kernel.

DKMA extends this by interleaving search and reduction repeatedly:

1. search generates population-level evidence,
2. consensus vertices are committed,
3. residual graph is re-kernelized,
4. search continues in new reduced space.

This dynamic loop can expose reduction opportunities that are invisible before partial search decisions are made.

## 2. DKMA State Model

Core mutable state inside _dkma_solve_common includes:

- current_k_n: current kernel size
- current_k_edges: current kernel edge set
- current_forced: original-graph forced vertices accumulated so far
- current_k_new_to_old: mapping from kernel IDs to original IDs
- population: current set of kernel-space candidate solutions
- best_solution: best kernel-space candidate
- best_original: mapped best candidate in original graph IDs
- stagnation counters and dynamic reduction count

Understanding these variables is key to understanding remapping correctness.

## 3. Step-by-Step DKMA Execution

### 3.1 Initial reduction stage

1. kernelize input graph (directed or undirected path)
2. collect forced vertices
3. build kernel-to-original mapping
4. directed path optionally applies additional SHORTONE-style contraction

If kernel becomes empty, DKMA returns forced set directly.

### 3.2 Population initialization stage

Initialization strategy is mixed:

- warm-start from KMA seed when budget permits
- random population fill for diversity
- duplicate removal and sorting by objective proxy (size)

This gives both quality and diversity without expensive full-generation bootstrapping.

### 3.3 Iterative dynamic loop

Loop continues while generation/time/patience constraints are satisfied.

Per-cycle operations:

1. one-generation MA-style evolution step
2. optional diversification injection
3. rank and validate candidates
4. update best solution if improved
5. at dynkern_every intervals, perform commit + re-kernelize + remap

### 3.4 Post-loop refinement

After loop exit:

- optional gain-based local search
- final acyclicity verification in original graph
- repair fallback if needed
- optional fallback to KMA on severe failure

## 4. Dynamic Components in Detail

### 4.1 Consensus commit logic

Helper: _commit_vertices_from_population.

Mechanism:

- count vertex frequency across population
- commit vertices above threshold fraction
- cap over-commit situations to avoid collapse

Primary control:

- commit_threshold (default ~0.6)

### 4.2 Dynamic re-kernelization logic

Helper: _dynamic_kernelize.

Mechanism:

1. remove committed kernel vertices
2. map committed picks to original forced set
3. kernelize residual graph
4. rebuild mapping layers
5. export old->new kernel index map for population remap

This is the critical correctness step that keeps solution semantics stable across changing kernel spaces.

### 4.3 Population remap

Helper: _remap_population.

Mechanism:

- transform each individual from old kernel IDs to new kernel IDs
- drop committed/deleted vertices
- refill with random candidates if diversity collapses

### 4.4 Diversification

Helper: _diversify_with_topological_ordering.

Purpose:

- prevent search collapse into single structural basin
- reintroduce exploration pressure periodically

### 4.5 Gain local search

Helper: _gain_local_search.

Implements swap-style local optimization (1-1 and 2-1 style moves) while preserving acyclicity constraints.

## 5. Runtime Safeguards and Failure Handling

DKMA has explicit reliability guards:

1. hard global wall-clock budget
2. short-budget fallback branch (KMA-first strategy)
3. final validity check in original graph
4. repair fallback (_greedy_acyclic_repair)
5. optional KMA fallback when final DKMA state is invalid

These safeguards are why DKMA remains usable in large unattended benchmark batches.

## 6. Short-Budget Branch (<= 30s)

When time budget is very small, full dynamic interleaving can become unstable or ineffective.

Implemented strategy:

1. run baseline KMA
2. if time remains, run diversified alternate shot
3. keep best valid result
4. apply non-worsening prune pass

This branch prioritizes robustness over dynamic sophistication for strict wall-time regimes.

## 7. Directed vs Undirected DKMA

Both wrappers call the same engine with directed flag.

Differences handled internally:

- directed/undirected kernelization routines
- directed/undirected acyclicity predicates
- directed-specific contraction behavior and edge semantics

The user-facing API remains consistent across families.

## 8. GNN-DKMA Coupling

run_hybrid.py includes:

- gnn_dkma_solve_undirected
- gnn_dkma_solve_directed

Pipeline:

1. initial kernelization
2. GNN probability inference on kernel
3. high-confidence hard-fix
4. DKMA on residual
5. map and union with forced/fixed vertices

This is DKMA with ML-guided initialization pressure.

## 9. Parameter Semantics and Tuning

Primary DKMA controls:

- pop_size
- max_gens
- early_stop
- max_time_seconds
- commit_threshold
- dynkern_every
- gain_search on/off
- diversify on/off

Tuning heuristics:

1. if runtime too high: reduce pop_size, max_gens; increase early_stop strictness
2. if quality unstable: increase pop_size; relax early_stop; keep diversification on
3. if over-committing hurts quality: raise commit_threshold or reduce commit frequency

## 10. Diagnostics and Interpretability

When diagnostics are requested, DKMA returns stage and structural metadata, including kernel size transitions and dynamic reduction count.

This makes DKMA behavior auditable and suitable for ablation studies.

## 11. Practical Summary

DKMA is the repository's advanced dynamic heuristic:

- starts from kernelized search like KMA,
- adds consensus-driven reduction interleaving,
- preserves correctness through remapping and validation guards,
- remains deployable in batch pipelines because of short-budget and fallback protections.

If KMA is the stable baseline, DKMA is the adaptive extension for harder or structurally diverse workloads.
