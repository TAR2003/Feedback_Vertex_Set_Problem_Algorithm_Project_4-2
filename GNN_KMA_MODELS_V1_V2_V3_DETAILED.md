# GNN-KMA v1/v2/v3: Full Dataset, Training, and Runtime Integration Guide

This document explains the complete GNN-KMA stack in this repository at implementation depth:

1. graph-to-PT dataset generation
2. v1/v2/v3 feature engineering differences
3. model architecture evolution
4. training objectives and checkpoint policies
5. runtime inference and coupling to KMA/DKMA

Core code files:

- gnn_model/dataset_gen.py
- gnn_model/train.py
- experiments/run_hybrid.py
- gnn_model/model_undirected.py
- gnn_model/model_directed.py
- gnn_model/model_undirected_v2.py
- gnn_model/model_directed_v2.py
- gnn_model/model_directed_v3.py
- gnn_model/feature_engineering_v2.py
- gnn_model/feature_engineering_v3.py

## 1. End-to-End Data-to-Solver View

The full lifecycle is:

1. TXT graph corpus is prepared in benchmark format.
2. dataset_gen.py computes node features and solver labels.
3. PT samples are saved with metadata for training and stratification.
4. train.py trains variant-specific models and writes weights.
5. run_hybrid.py loads weights and performs GNN-guided KMA/DKMA solving.

A key design rule in this repository is distribution consistency: benchmark data distribution and training data distribution are aligned by family/track/category.

## 2. PT Dataset Generation: gnn_model/dataset_gen.py

### 2.1 Track-aware label generation

Label source by track:

- exact_track: IC-based labels
- heuristic_track: KMA-based labels with timeout-bound MA stage

This mirrors benchmark role separation:

- exact track approximates high-fidelity supervision
- heuristic track reflects practical large-instance behavior

### 2.2 Reliability guarantees

dataset_gen.py includes multiple correctness guards:

1. subprocess timeout isolation for solver calls where needed
2. explicit validation that predicted removal set is acyclic
3. CSV status ledger for each source graph (completed, timeout, invalid)
4. deterministic source traversal and reproducible sampling behavior

### 2.3 PT sample contract

Each sample stores:

- x: node feature matrix
- edge_index: PyG COO edges
- y: node labels (binary FVS membership)
- fvs_size
- family, track, category, source_file, feature_set metadata

### 2.4 Variant data roots

- v1: gnn_model/datasets/pt
- v2: gnn_model/datasets/pt_v2
- v3: gnn_model/datasets/pt_v3

## 3. Feature Engineering Evolution

### 3.1 v1 features

Undirected v1 (compact local structure):

- normalized degree
- clustering coefficient
- normalized log-degree

Directed v1:

- normalized in-degree
- normalized out-degree
- normalized min(in,out)

Use case: fast baseline, low feature cost.

### 3.2 v2 features

v2 extends to richer structural channels (11 dims):

- degree-like channels (directed or undirected)
- RWSE short-range return channels
- motif-derived channels (triangles, 4-cycle, 4-clique style counts)
- core-number channel

Use case: stronger structural discrimination with moderate overhead.

### 3.3 v3 features

v3 moves to a 16-channel research-grade set with deeper structural context.

Highlights:

- longer RWSE horizon steps [2,3,4,6,8,12,16]
- SCC-aware channels for directed cycle structure
- cycle-score style interaction features
- directional ratio features
- reduced reliance on expensive low-yield motif channels

Use case: high-quality precision-first guidance for hybrid solving.

## 4. Model Architecture Evolution

### 4.1 v1 models

UndirectedFVSNet:

- 3-layer message passing (GraphSAGE when available)
- BN + dropout + MLP head
- 2-class log-softmax output

DirectedFVSNet:

- custom directed aggregation layer
- separate in/out/self aggregation terms
- 3 layers + BN + dropout + MLP
- 2-class log-softmax output

### 4.2 v2 models

Architecture style remains similar to v1 but input channels and representation quality are upgraded via v2 features.

- UndirectedFVSNetV2 default in_channels = 11
- DirectedFVSNetV2 default in_channels = 11

### 4.3 v3 models

Implemented in model_directed_v3.py with directed and undirected classes.

DirectedFVSNetV3 includes:

- input projection
- 5 residual GATv2-style blocks
- separate forward and reverse message paths
- per-layer fusion projections
- global readout context concatenated to node embedding
- single-logit output per node

UndirectedFVSNetV3 follows similar residual GAT design without direction split.

## 5. Training Pipeline: gnn_model/train.py

### 5.1 Dataset loading and cleaning

Before training, the script scans PT files and drops corrupted/invalid samples (shape mismatch, non-finite values, invalid edge_index bounds, etc.).

### 5.2 Splitting strategy

train.py uses stratified split behavior by graph metadata (family/category), reducing validation distribution mismatch.

Optional track-level subsampling is also supported via take-exact and take-heuristic.

### 5.3 v1/v2 objective and checkpointing

- weighted NLL-based training
- cosine schedule
- primary selection metric: validation F1

### 5.4 v3 objective and checkpointing

v3 training introduces specialized optimization choices:

- AsymmetricFVSLoss to penalize false positives more strongly
- warmup + cosine schedule
- gradient clipping
- primary validation metric: topk_precision@8%

Why this matters: false positives are expensive in hard-fix hybrid coupling, so v3 optimizes for precision where it directly affects solver quality.

## 6. Runtime Inference Engine: experiments/run_hybrid.py

### 6.1 Lazy loading and compatibility

run_hybrid.py lazily imports torch, model classes, and PyG loaders to keep startup cost low and allow graceful fallback when optional dependencies are missing.

### 6.2 Feature-model matching

Each solver variant calls matching inference path:

- v1 uses base feature functions and v1 weights
- v2 uses feature_engineering_v2 + v2 weights
- v3 uses feature_engineering_v3 + v3 weights (with fallback behavior where applicable)

### 6.3 Probability extraction

Inference uses sigmoid probabilities and supports robust handling of shape variants in output tensors.

## 7. Coupling to KMA/DKMA

### 7.1 Current KMA coupling design

The active coupling strategy is precision-first soft-hint:

1. kernelize graph
2. run GNN on kernel
3. hard-fix only high-confidence vertices
4. cap hard-fix ratio (around 8%)
5. run KMA on residual
6. map back and union with forced set

If confidence is insufficient, code falls back to pure KMA path rather than forcing uncertain vertices.

### 7.2 Why precision-first

In this design, false positives are high-cost because hard-fixed wrong vertices inflate final FVS size and cannot be removed later in that branch. Therefore, thresholding and caps are critical.

### 7.3 GNN-DKMA path

gnn_dkma variants apply the same probability-guided fixing principle before running DKMA on reduced kernel.

## 8. Variant Comparison Matrix

v1:

- lowest feature and model complexity
- fastest setup
- baseline quality

v2:

- richer structural features
- same family of training objective as v1
- stronger structural discrimination

v3:

- deepest feature/model redesign
- precision-oriented loss and metric selection
- strongest alignment with hybrid hard-fix risk profile

## 9. Operational Workflows

### 9.1 Generate PT datasets

Run one variant at a time to control compute:

- dataset_gen.py with variant v1/v2/v3
- tune solver timeout and KMA label parameters to match hardware budget

### 9.2 Train and validate

- train.py with matching variant and type
- use fixed seed for reproducibility
- monitor primary metric by variant (F1 for v1/v2, topk precision for v3)

### 9.3 Deploy in benchmark scripts

benchmark_undirected.py and benchmark_directed.py support GNN-KMA, GNN-KMA-2, GNN-KMA-3, and GNN-DKMA with threshold, timeout, and hidden-dim controls.

## 10. Common Failure Modes and Recovery

1. Missing weights: train.py must be run for target variant/type.
2. Feature mismatch: variant must match dataset root and model class.
3. Slow inference on very large kernels: reduce gnn-timeout or use fallback path.
4. Poor hybrid quality due to over-fixing: increase threshold or reduce fix fraction.

## 11. Summary

The repository implements an end-to-end, production-grade hybrid ML + combinatorial optimization stack:

- robust PT label generation,
- progressively stronger feature/model variants,
- variant-aware training and checkpoint selection,
- runtime coupling that explicitly manages false-positive risk.

v1 establishes baseline guidance, v2 improves structural representation, and v3 delivers precision-focused guidance optimized for real hybrid solver behavior.
