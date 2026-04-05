# GNN-KMA Models (v1, v2, v3): Dataset, Training, and Solution Pipeline

This document explains all three GNN-KMA generations implemented in this repository, including:

1. Dataset generation
2. Model architectures and features
3. Training protocol
4. Inference and solver coupling with KMA/DKMA

Primary code files:

- Dataset generation: `gnn_model/dataset_gen.py`
- Training: `gnn_model/train.py`
- Hybrid solver runtime: `experiments/run_hybrid.py`
- Models:
  - `gnn_model/model_undirected.py`
  - `gnn_model/model_directed.py`
  - `gnn_model/model_undirected_v2.py`
  - `gnn_model/model_directed_v2.py`
  - `gnn_model/model_directed_v3.py` (contains directed + undirected v3 classes)
- Feature engineering:
  - `gnn_model/feature_engineering_v2.py`
  - `gnn_model/feature_engineering_v3.py`

---

## 1) Data Generation for GNN Training

## 1.1 Source distribution

`gnn_model/dataset_gen.py` generates `.pt` datasets from the same two-track benchmark distribution:

- `exact_track`
- `heuristic_track`

Across both families:

- undirected
- directed

This keeps training data aligned with benchmark workload composition.

## 1.2 Labeling strategy

Labels are solver-derived FVS membership vectors.

- Exact track labels:
  - produced with IC (exact style)
  - timeout-protected via subprocess wrapper
- Heuristic track labels:
  - produced with KMA-style solver path
  - timeout parameter routed to MA-stage budget

Every label is validated for acyclicity before dataset write.

## 1.3 Output format

Each sample is stored as PyG `Data`:

- `x` node features
- `edge_index`
- `y` binary node labels
- metadata (family/category/track/source)
- `fvs_size`

Output roots by variant:

- v1: `gnn_model/datasets/pt`
- v2: `gnn_model/datasets/pt_v2`
- v3: `gnn_model/datasets/pt_v3`

## 1.4 Robustness controls

The dataset generator includes:

- per-graph timeout isolation in subprocess
- invalid-label detection (`invalid FVS` safeguard)
- persistent CSV status tracking (`completed`, `timeout`, `invalid`)
- forced regeneration and root cleanup options

---

## 2) Model Generation v1

## 2.1 Features (v1)

Undirected v1 features are compact structural signals:

- normalized degree
- clustering coefficient
- normalized log-degree

Directed v1 uses directional counterparts:

- normalized in-degree
- normalized out-degree
- normalized min(in,out)

## 2.2 Architecture (v1)

### Undirected v1

`UndirectedFVSNet`:

- 3 message-passing layers (GraphSAGE when available)
- batch normalization + dropout
- MLP head
- output as 2-class log-softmax

### Directed v1

`DirectedFVSNet`:

- custom directed layer with separate incoming/outgoing aggregations
- 3 layers + BN + dropout + MLP
- output as 2-class log-softmax

## 2.3 Loss and training style (v1)

- weighted NLL loss to address class imbalance
- cosine LR schedule
- checkpoint selected by validation F1

---

## 3) Model Generation v2

v2 retains general architecture style but significantly expands feature expressiveness.

## 3.1 Features (v2)

`feature_engineering_v2.py` builds 11-channel features.

Added signals include:

- RWSE return probabilities (steps 2..5)
- motif-related counts (triangles, 4-cycles, 4-cliques)
- k-core number

Directed v2 keeps directed degree channels and computes motif/core on undirected projection where appropriate.

## 3.2 Architecture (v2)

### Undirected v2

`UndirectedFVSNetV2`:

- in_channels defaults to 11
- 3 GraphSAGE/manual conv layers
- same classifier style as v1

### Directed v2

`DirectedFVSNetV2`:

- in_channels defaults to 11
- directed in/out/self aggregation layers
- same classifier style as v1

## 3.3 Training style (v2)

- weighted NLL objective
- stratified train/val split by family/category in `train.py`
- saved weights:
  - `*_gcn_v2.pt`

---

## 4) Model Generation v3

v3 is the research-grade redesign.

## 4.1 Features (v3)

`feature_engineering_v3.py` computes 16-channel features.

Key additions and redesign:

- longer-horizon RWSE steps: `[2,3,4,6,8,12,16]`
- SCC-centric directed features
- cycle-score style interaction feature
- directional ratio channels
- reduced dependency on expensive low-signal motifs

Undirected adaptation maps SCC ideas to connected-component proxies.

## 4.2 Architecture (v3)

Implemented in `model_directed_v3.py`.

### Directed v3 (`DirectedFVSNetV3`)

- input projection
- 5 residual GATv2-style blocks
- separate forward/reverse edge message paths
- fusion projections per layer
- global context readout concatenated to node embeddings
- final MLP outputs one logit per node

### Undirected v3 (`UndirectedFVSNetV3`)

- same design philosophy with single undirected path
- residual GAT blocks + global context + single-logit output

## 4.3 Training style (v3)

`train.py` switches to v3-specialized protocol:

- `AsymmetricFVSLoss` (false positives penalized more than false negatives)
- warmup + cosine schedule
- gradient clipping
- primary validation metric: top-k precision at 8%
- checkpoint selection by `topk_precision` instead of F1

Saved weights:

- `undirected_fvs_gcn_v3.pt`
- `directed_fvs_gcn_v3.pt`

---

## 5) Runtime Inference and Solver Coupling

All runtime integration is in `experiments/run_hybrid.py`.

## 5.1 Inference execution

The runtime pipeline:

1. Load model lazily.
2. Auto-detect hidden dimension from checkpoint when possible.
3. Compute variant-appropriate features.
4. Run mini-batch node inference (`NeighborLoader`) where available.
5. Produce per-node probabilities.

## 5.2 Soft-hint coupling strategy

Current GNN-KMA coupling is precision-first soft-hint design.

Core rule:

- hard-fix only high-confidence nodes above threshold (default around 0.65)
- cap hard-fixes to a small kernel fraction (8%)
- if confidence is insufficient, fall back to pure KMA instead of risky hard-fixing

This prevents FP-heavy inflation seen in naive hard-fix approaches.

## 5.3 Solver variants

Available hybrid functions include:

- `gnn_KMA_solve_undirected` / `gnn_KMA_solve_directed` (v1)
- `gnn_KMA2_solve_undirected` / `gnn_KMA2_solve_directed` (v2)
- `gnn_KMA3_solve_undirected` / `gnn_KMA3_solve_directed` (v3)
- `gnn_dkma_*` variants for DKMA backend

Common flow:

1. kernelize graph
2. infer GNN probabilities on kernel
3. choose high-confidence candidates
4. run KMA/DKMA refinement
5. map solution back to original graph IDs
6. union with forced reductions

---

## 6) Training and Deployment Commands (Practical)

## 6.1 Generate PT datasets

Use variant-specific generation:

- v1: `--variant v1`
- v2: `--variant v2`
- v3: `--variant v3`

Adjust:

- track selection
- solver timeout
- KMA label parameters (`kma-pop`, `kma-gens`, `kma-early-stop`)

## 6.2 Train models

`gnn_model/train.py` supports:

- `--type undirected|directed|both`
- `--variant v1|v2|v3`
- shared hyperparameters (`epochs`, `lr`, `hidden`, `dropout`, `val-ratio`)
- v3-specific controls (`warmup-epochs`, `max-grad-norm`)

## 6.3 Use in benchmarks

`benchmark_undirected.py` and `benchmark_directed.py` can run:

- `GNN-KMA`
- `GNN-KMA-2`
- `GNN-KMA-3`
- `GNN-DKMA`

with timeout/threshold/hidden-dim controls and CSV logging.

---

## 7) Summary by Version

## v1

- compact features
- baseline hybrid guidance
- weighted-NLL training

## v2

- richer structural features (RWSE + motifs + core)
- improved guidance quality on difficult structures

## v3

- research-grade features + residual GAT architecture
- asymmetric training objective focused on false-positive control
- top-k precision optimized for hard-fix quality
- strongest alignment with soft-hint KMA coupling

Together, these three generations provide an incremental evolution from baseline GNN guidance to a robust, precision-controlled hybrid optimization pipeline.
