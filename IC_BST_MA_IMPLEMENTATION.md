# IC, BST, and MA Implementation Details

This document explains how BST, IC, and MA are implemented in this repository for both undirected and directed FVS.

Core implementation files:

- Undirected exact solvers: `cpp_engine/src/undirected_algos/exact_solver_u.cpp`
- Directed exact solvers: `cpp_engine/src/directed_algos/exact_solver_d.cpp`
- Undirected MA: `cpp_engine/src/undirected_algos/memetic_u.cpp`
- Directed MA: `cpp_engine/src/directed_algos/memetic_d.cpp`
- Python runners and timeout wrappers:
  - `experiments/benchmark_undirected.py`
  - `experiments/benchmark_directed.py`

---

## 1) BST (Bounded Search Tree)

BST is implemented as a fixed-parameter branching algorithm with iterative deepening on solution size `k`.

## Undirected BST

Implemented in `solve_undirected_BST` + recursive helper in `exact_solver_u.cpp`.

### Undirected IC procedure

1. Build graph.
2. For `k = 0..n`, call recursive solver.
3. In recursion:
   - Apply kernelization and forced reductions first.
   - Find any cycle.
   - If no cycle: success.
   - If cycle exists and `k == 0`: fail branch.
   - Otherwise branch by selecting each cycle vertex into FVS.

### Important implementation details

- Kernelization is executed before branching to reduce state and prune infeasible branches early.
- Branch vertex ordering is degree-based (high degree first) as a practical speed heuristic.
- Returned solution is deduplicated and sorted before final return.

## Directed BST

Implemented in `solve_directed_BST` + recursive helper in `exact_solver_d.cpp`.

### Directed IC differences

- Cycles are directed cycles (`find_directed_cycle`).
- Directed kernelization is used.
- SCC pruning is applied to remove vertices not in nontrivial SCCs since they cannot lie on directed cycles.
- Branching priority uses total degree (`in + out`) for practical pruning.

---

## 2) IC (Iterative Compression)

IC is implemented as incremental graph growth with repeated compression of a size `k+1` FVS into size `k` when possible.

## Undirected IC

Implemented via:

- `solve_undirected_IC`
- `compress`
- `restricted_bst`

### Procedure

1. Order vertices (degree-descending in this implementation for practical quality).
2. Add vertices incrementally to current induced graph.
3. Maintain FVS `X` for current prefix; on each step push new vertex into `X`.
4. Attempt compression repeatedly:
   - Enumerate subsets `Z ⊆ X`.
   - Let `Y = X \ Z`.
   - Require `G[Y]` to be acyclic (forest condition).
   - Remove `Z`, then run restricted BST that may only pick from allowed non-forbidden vertices.
   - Return first valid compressed set.
5. Final cleanup tries removing redundant vertices from the output.

### Why restricted BST is needed

The compression subproblem forbids selecting some vertices (`X`) in certain branches, so the solver must enforce forbidden constraints while branching.

## Directed IC

Implemented in `exact_solver_d.cpp` as:

- `solve_directed_IC`
- `compress_directed`
- `restricted_bst_directed`

### Directed-specific differences

- Forest test becomes DAG test (`induced_has_dcycle`).
- SCC pruning and directed kernelization are integrated in restricted recursion.
- Same subset enumeration strategy (`Z` / `Y`) but with directed acyclicity constraints.

---

## 3) MA (Memetic Algorithm)

MA is implemented in C++ for both graph families.

## Common representation

An individual is a binary vector over vertices:

- `1` means vertex is selected in FVS
- `0` means not selected

## Common components

- Population initialization (greedy + random repaired solutions)
- Tournament selection
- Uniform crossover
- Bit-flip mutation (`~1/n` probability)
- Feasibility repair for cycle elimination
- Local search that removes redundant selected vertices
- Early stopping via patience
- Hard wall-clock timeout guard

## Undirected MA (`memetic_u.cpp`)

### Fitness

Fitness is based on solution size plus cycle penalty:

`fitness = |FVS| + n * cycles_remaining`

This strongly separates feasible and infeasible solutions while still preferring smaller feasible sets.

### Initialization

- Greedy seed uses high-degree removal ordering.
- Additional individuals are random, repaired, then locally improved.

### Search behavior

- Replaces worst individual when child improves quality.
- Tracks generation-best size and stops after no improvement for `patience` generations.

## Directed MA (`memetic_d.cpp`)

Directed variant keeps the same global structure with directed-specific scoring/repair cues.

### Directed heuristics

- Greedy seed uses `min(in_degree, out_degree)` to target directed-cycle participation.
- Repair often uses total directed degree to select influential removals.
- Directed local search validates against directed cycle existence.

---

## 4) Runtime Integration in Python

`benchmark_undirected.py` and `benchmark_directed.py` expose these algorithms through CLI and CSV logging.

### Practical behavior

- Exact algorithms (BST/IC) are run through C++ bindings directly.
- MA/KMA/GNN variants are wrapped with timeout logic.
- Output rows include runtime, size, validity, and status for benchmark tracking.

---

## 5) Complexity and Usage Guidance

- BST/IC are exact and can be expensive on large graphs.
- MA is heuristic and scales much better for large instances.
- In this codebase, exact-track runs include exact methods for quality baselines, while heuristic-track focuses on scalable methods.

For high-scale runs, prefer MA/KMA/DKMA families; use BST/IC for ground-truth style references and controlled-size experiments.
