# DKMA Implementation Procedure

This document describes how DKMA (Dynamic Kernelized Memetic Algorithm) is implemented in this repository.

Main implementation file:

- `experiments/run_hybrid.py`

Primary entry functions:

- `dkma_solve_undirected(...)`
- `dkma_solve_directed(...)`

Both route into the shared engine:

- `_dkma_solve_common(...)`

---

## 1) What Makes DKMA Different from KMA

KMA performs one kernelization pass followed by MA search.

DKMA extends this with dynamic reduction-search interleaving:

1. Initialize on an initial kernel.
2. Maintain a population of candidate solutions.
3. Periodically commit consensus vertices.
4. Re-kernelize the residual graph.
5. Remap population/solutions to the new kernel.
6. Continue search with updated reduced state.

This dynamic loop can open additional reduction opportunities during search.

---

## 2) DKMA Core Flow (`_dkma_solve_common`)

### Step A: Initial reduction

- Kernelize graph (directed or undirected path).
- Capture forced vertices and kernel mapping to original IDs.
- For directed path, SHORTONE-style contraction may be applied.

### Step B: Population initialization

Population uses:

- a warm-start KMA seed when time budget allows
- additional random individuals
- duplicate elimination and size ranking

### Step C: Iterative dynamic loop

For each generation (subject to time/patience):

1. Run one lightweight MA evolution step.
2. Optional diversification phase.
3. Evaluate/rank candidates.
4. Validate candidates on original graph acyclicity.
5. Update global best if improved.
6. Every `dynkern_every` generations:
   - compute consensus commitments from population
   - dynamically re-kernelize residual graph
   - remap population and best solution into new kernel index space

### Step D: Post-optimization

- Optional gain-based local search (`_gain_local_search`).
- Final acyclicity validation.
- Repair if needed (`_greedy_acyclic_repair`).
- Optional fallback to KMA if final validation fails.

---

## 3) Dynamic Components in Detail

## 3.1 Consensus commit

`_commit_vertices_from_population` computes per-vertex support count across population and commits vertices above threshold.

Controls:

- `commit_threshold` (default `0.6`)

Safety:

- if too many vertices meet threshold, commitment is capped (top-ranked subset only)

## 3.2 Dynamic re-kernelization

`_dynamic_kernelize`:

- removes committed kernel vertices
- re-kernelizes residual graph
- updates forced original vertices
- builds new kernel mapping
- provides old->new index remap for population transfer

## 3.3 Population remap

`_remap_population` rewrites individuals after kernel change and injects random individuals when needed to preserve diversity.

## 3.4 Diversification

`_diversify_with_topological_ordering` periodically injects diversity using topological-order-inspired perturbations.

## 3.5 Gain local search

`_gain_local_search` tries 1-1 and 2-1 swap improvements while preserving acyclicity.

---

## 4) Short-Budget Behavior

For low time budgets (`<= 30s`), DKMA intentionally uses a robust short-budget branch:

1. Run KMA baseline.
2. If time remains, run diversified second KMA shot.
3. Keep the better valid result.
4. Apply non-worsening prune refinement.

This avoids unstable long-loop behavior when budget is too small for full dynamic adaptation.

---

## 5) Directed and Undirected Variants

The same DKMA core handles both families via a `directed` flag.

Differences are handled internally through:

- directed vs undirected kernelization
- directed cycle checks vs undirected cycle checks
- directed-specific helper rules (including SHORTONE path)

Public wrappers expose identical argument structure for both graph families.

---

## 6) GNN-DKMA

`run_hybrid.py` also provides:

- `gnn_dkma_solve_undirected(...)`
- `gnn_dkma_solve_directed(...)`

Pipeline:

1. Initial kernelization
2. GNN probability inference
3. Hard-fix high-confidence kernel vertices
4. Run DKMA on reduced kernel
5. Map and union with forced/fixed sets

This mirrors DKMA’s dynamic backend while introducing GNN-guided initialization pressure.

---

## 7) Practical Tuning Knobs

Important DKMA controls exposed by benchmark CLI:

- `pop_size`
- `max_gens`
- `early_stop`
- `max_time_seconds`
- `commit_threshold`
- `dynkern_every`
- `gain_search` toggle
- `diversify` toggle

The defaults are conservative and tuned for stable benchmark operation.

---

## 8) Summary

DKMA in this repository is a dynamic, reduction-aware extension of KMA:

- search and reduction are interleaved, not one-shot,
- consensus commitments trigger structural graph simplification during search,
- remapping logic keeps optimization state consistent after each reduction,
- final safeguards ensure valid acyclic output even under strict runtime limits.
