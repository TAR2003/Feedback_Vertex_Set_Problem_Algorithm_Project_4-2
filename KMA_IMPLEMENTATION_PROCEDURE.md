# KMA Implementation Procedure (Professional Deep-Dive)

This document explains exactly how KMA (Kernelized Memetic Algorithm) is implemented in this repository, from C++ core solver flow to Python runtime wrappers and benchmark integration.

Primary code anchors:

- cpp_engine/src/undirected_algos/memetic_u.cpp
- cpp_engine/src/directed_algos/memetic_d.cpp
- experiments/run_hybrid.py
- experiments/benchmark_undirected.py
- experiments/benchmark_directed.py

## 1. KMA in One Sentence

KMA = kernelization stage + memetic optimization stage + mapping back to original graph.

The kernelization stage reduces problem size while preserving correctness structure; the memetic stage optimizes on the reduced instance; mapping reconstructs the final answer in original vertex IDs.

## 2. Why KMA Exists in This Codebase

MA alone scales better than exact methods but still spends time on vertices that reductions can eliminate deterministically.

KMA addresses this by:

1. applying reduction rules first,
2. searching only on the active kernel,
3. reintroducing forced vertices afterwards.

This usually improves both wall-clock and solution quality stability for medium/large instances.

## 3. C++ Implementation Flow

### 3.1 Undirected path

Entry function: solve_undirected_KMA.

Execution steps:

1. Build UndirectedGraph from input edges.
2. Run kernelize_undirected.
3. Collect forced vertices produced by reductions.
4. Build old/new index mapping for active kernel vertices.
5. Build kernel edge list among active vertices.
6. Run solve_undirected_MA on kernel.
7. Map kernel solution back to original IDs.
8. Union mapped solution with forced set.
9. Sort and deduplicate output.

### 3.2 Directed path

Entry function: solve_directed_KMA.

The same structure is used with directed graph structures and directed kernelization:

- directed adjacency and cycle semantics
- kernelize_directed reductions
- solve_directed_MA for kernel optimization

### 3.3 Alias compatibility

Both memetic files expose KME aliases forwarding to KMA for backward compatibility with older call sites.

## 4. Python KMA Runtime Wrappers

run_hybrid.py contains operational wrappers:

- kma_solve_undirected
- kma_solve_directed

These wrappers are not just call-throughs; they provide stage timing and robust fallback dispatch.

### 4.1 Dispatch hierarchy

Wrapper preference order:

1. solve_*_KMA
2. solve_*_KME
3. solve_*_MA

This ensures portability across build variants where symbol exposure may differ.

### 4.2 Diagnostics support

Wrappers can return stage metrics with keys such as:

- kernelization_ms
- ma_ms

This allows benchmark scripts and research runs to distinguish reduction time from search time.

## 5. Runtime Semantics and Time Budgeting

KMA-related parameters:

- pop_size
- max_gens
- early_stop
- max_time_seconds

Important practical semantic:

- The expensive search stage is MA on kernel.
- Kernelization executes before MA and can significantly reduce effective runtime by shrinking kernel size.
- Timeouts in benchmark wrappers are designed to preserve best-so-far outputs where possible, rather than discarding progress.

## 6. Mapping and Correctness Intuition

KMA output has two components:

1. forced vertices from reductions
2. selected kernel vertices from MA

Mapping logic converts kernel vertex IDs back to original IDs and unions with forced set.

Because reductions are consistency-preserving and mapping is deterministic, the produced set is a valid candidate over original graph space.

## 7. Complexity and Performance Behavior

There is no single closed-form practical runtime because KMA behavior depends on:

- reduction effectiveness on specific graph family
- kernel size after reductions
- evolutionary convergence profile

Empirically in this codebase:

- strong reductions yield major speedups vs pure MA
- weakly reducible instances behave closer to MA runtime
- larger pop/gens improve search quality but increase runtime

## 8. Tuning Guidance

For throughput-focused runs:

- keep moderate pop_size
- cap max_gens and use early_stop
- enforce wall timeout

For quality-focused runs:

- increase pop_size and max_gens
- relax early_stop
- retain timeout safety for batch stability

For very large batch campaigns, always rely on CSV resume behavior in benchmark scripts.

## 9. Integration in Benchmark Tooling

KMA is first-class in benchmark scripts and pipeline runners:

- single algorithm mode (algo KMA)
- ALL profile comparisons
- suite and pipeline orchestration

Output rows include runtime, validity, status fields, allowing direct comparison with BST, IC, MA, DKMA, and GNN-guided variants.

## 10. Relationship to DKMA and GNN-KMA

KMA is the baseline hybrid backbone:

- DKMA extends KMA with dynamic reduction-search interleaving.
- GNN-KMA uses GNN priors plus KMA refinement.

In other words, understanding KMA implementation is mandatory before reasoning correctly about DKMA or GNN-assisted procedures in this repository.

## 11. Practical Takeaway

In this project, KMA is the default robust solver for medium/large practical workloads:

- significantly more scalable than exact methods,
- typically more stable than plain MA on reducible graphs,
- forms the operational bridge between pure heuristics and ML-guided hybrids.
