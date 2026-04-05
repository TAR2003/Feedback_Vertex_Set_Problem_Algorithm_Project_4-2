# IC, BST, and MA: Full Implementation and Runtime Guide

This document explains the three foundational solver families in the project at implementation depth:

1. BST (Bounded Search Tree)
2. IC (Iterative Compression)
3. MA (Memetic Algorithm)

Code anchors:

- cpp_engine/src/undirected_algos/exact_solver_u.cpp
- cpp_engine/src/directed_algos/exact_solver_d.cpp
- cpp_engine/src/undirected_algos/memetic_u.cpp
- cpp_engine/src/directed_algos/memetic_d.cpp
- experiments/benchmark_undirected.py
- experiments/benchmark_directed.py

## 1. Problem Context

For a graph G, FVS asks for a minimum vertex set S such that G - S is acyclic.

- undirected: G - S must be a forest
- directed: G - S must be a DAG

This repository contains exact and heuristic implementations for both settings, sharing a common C++ core with Python orchestration.

## 2. BST (Bounded Search Tree)

BST is a parameterized exact approach with iterative deepening on budget k.

High-level invariant:

- if a cycle exists, at least one cycle vertex must be selected

### 2.1 Undirected BST flow

Entry point: solve_undirected_BST.

Execution:

1. Build UndirectedGraph from edge list.
2. Iterate k from 0 to n.
3. Call recursive branch solver with graph copy and budget k.
4. In each recursive frame:
   - run kernelize_undirected (forced reductions, simplification)
   - if no cycle: accept
   - if cycle exists and k exhausted: reject branch
   - otherwise branch on cycle vertices

Implementation-level notes:

- Branching order is degree-descending for faster practical pruning.
- Graph is passed by value intentionally to isolate branch state.
- Result is deduplicated/sorted before return.

### 2.2 Directed BST flow

Entry point: solve_directed_BST.

Differences from undirected:

- cycle detection is directed (find_directed_cycle)
- directed kernelization path is used
- SCC pruning removes vertices outside nontrivial SCCs (cannot be on directed cycles)
- branching score uses in+out degree

Practical effect:

- directed SCC pruning greatly reduces search space on sparse real graphs.

### 2.3 BST runtime profile

BST is exact but exponential in parameterized budget behavior. In practice:

- excellent on small kernels and low-cycle density
- can become slow on larger/highly cyclic instances

Use BST as exact baseline and correctness reference rather than default large-scale solver.

## 3. IC (Iterative Compression)

IC is exact and often stronger than naive branching in practice.

Core idea:

- maintain a valid solution X while adding vertices incrementally
- repeatedly compress size |X|+1 solution back to size |X| if possible

### 3.1 Undirected IC structure

Main functions:

- solve_undirected_IC
- compress
- restricted_bst

Execution pattern:

1. Build incremental induced graph by ordered vertex insertion.
2. Maintain current FVS X.
3. For each insertion, add new vertex to X (trivial validity restore).
4. Attempt repeated compression:
   - enumerate subset Z of X
   - let Y = X - Z
   - require induced G[Y] to be acyclic
   - solve remaining part with restricted BST disallowing forbidden choices
   - return first valid compressed candidate
5. Run final redundancy cleanup by removal testing.

### 3.2 Why restricted BST exists

Compression imposes forbidden picks in some branches. restricted_bst enforces these constraints while preserving BST-style branching correctness.

### 3.3 Directed IC structure

Directed counterparts in exact_solver_d.cpp:

- solve_directed_IC
- compress_directed
- restricted_bst_directed

Directed changes:

- forest condition becomes DAG condition
- induced cycle check uses directed cycle routine
- directed kernelization/SCC pruning are integrated in restricted recursion

### 3.4 IC runtime profile

IC remains exact but typically offers better practical scaling than straightforward branching on many structured instances.

It is still not intended as the default for very large heuristic-track workloads.

## 4. MA (Memetic Algorithm)

MA is the scalable heuristic backbone and is implemented for both families.

Representation:

- binary chromosome over vertices
- 1 means selected into current FVS candidate

Core operators:

- greedy + random repaired initialization
- tournament parent selection
- uniform crossover
- bit mutation (approx 1/n)
- feasibility repair
- local search for redundancy removal

### 4.1 Undirected MA details

Entry: solve_undirected_MA in memetic_u.cpp.

Fitness is size-plus-penalty style:

- fitness = |candidate| + n * cycles_remaining

Behavioral outcomes:

- infeasible solutions are strongly penalized
- feasible smaller sets dominate ranking

Early stop is patience-based and wall-clock timeout is enforced.

### 4.2 Directed MA details

Entry: solve_directed_MA in memetic_d.cpp.

Directed-specific heuristics:

- seed preference uses min(in, out)
- repair heuristics use directed degree cues
- validity checks use directed cycle tests

Population and operator framework remains consistent with undirected version for implementation symmetry.

### 4.3 MA runtime profile

Compared to exact methods, MA scales significantly better and is suitable for large instances.

Trade-off:

- not guaranteed optimal
- quality depends on population, generations, and patience settings

## 5. Python Runtime Integration

benchmark_undirected.py and benchmark_directed.py provide execution wrappers around C++ solvers.

Important integration points:

- parser compatibility for benchmark graph formats
- timeout-aware execution for heuristic families
- result verification and CSV persistence
- algorithm mapping between CLI names and solver bindings

For exact algorithms, scripts call C++ exact entries directly. For heuristic/hybrid algorithms, wrappers maintain robust timeout and status reporting.

## 6. Recommended Solver Usage by Scenario

1. correctness baseline or small-instance optimality studies:
   BST and IC
2. medium/large benchmark throughput:
   MA and KMA
3. high-scale hybrid experiments:
   KMA, DKMA, and GNN-guided variants

## 7. Practical Parameter Guidance

For MA-like execution:

- increase pop_size for stability on heterogeneous graphs
- increase max_gens when runtime budget allows
- tune patience to avoid wasted tail generations
- always set hard timeout in large batch runs

For exact runs:

- limit to exact-track or explicitly small subsets
- use CSV resume mechanisms in benchmark scripts for long campaigns

## 8. Summary

BST and IC deliver exactness through branching/compression logic; MA delivers practical scale through evolutionary search with local refinement.

Together they form the algorithmic foundation on top of which KMA, DKMA, and GNN-hybrid procedures are built in this repository.
