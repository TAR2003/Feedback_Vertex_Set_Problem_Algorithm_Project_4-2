# GNN-KMA Model Family: Complete Workflow and Implementation

This document explains the GNN-KMA pipeline in this repository from dataset generation to training, inference, and testing. It covers GNN-KMA variants 1, 2, and 3, including how the models work, what features they use, how they integrate with the C++ solver, and how to run the full pipeline.

## 1. Overview

`GNN-KMA` is a hybrid solver for the Feedback Vertex Set (FVS) problem that combines:

- a Graph Neural Network (GNN) for vertex scoring,
- kernelization to reduce the graph,
- a memetic algorithm (MA) to refine the remaining kernel,
- a C++ backend (`cpp_engine`) for fast kernel reduction and MA refinement.

The repository implements three main GNN variants:

- `GNN-KMA` / `v1` — baseline GNN-guided KMA.
- `GNN-KMA-2` / `v2` — improved structural features and stronger graph encoding.
- `GNN-KMA-3` / `v3` — research-grade model using GATv2, residual connections, and RWSE/SCC features.

## 2. Model Variants

### 2.1 GNN-KMA (v1)

#### Files

- `gnn_model/model_undirected.py`
- `gnn_model/model_directed.py`
- `gnn_model/train.py`
- `experiments/run_hybrid.py`

#### Architecture

- Undirected: 3-layer Graph Neural Network with GraphSAGE-style or manual GCN convolution.
- Directed: custom directed convolution with separate incoming and outgoing aggregation.
- Each model has 3 graph-convolutional layers followed by batch normalization, ReLU, dropout, and a small MLP head.
- Output is per-node logits for 2 classes: `not-in-FVS` and `in-FVS`.

#### Features

- Undirected input channels: 3 features per vertex.
- Directed input channels: 3 features per vertex.

These features in `model_undirected.py` and `model_directed.py` represent local degree and cycle-proxy signals.

#### Loss and training

- Uses `torch.nn.NLLLoss` with class weights computed by label imbalance.
- This is the baseline GNN training behavior for the earliest GNN-KMA variant.

### 2.2 GNN-KMA-2 (v2)

#### Files

- `gnn_model/model_undirected_v2.py`
- `gnn_model/model_directed_v2.py`
- `gnn_model/feature_engineering_v2.py`
- `gnn_model/dataset_gen.py`
- `gnn_model/train.py`
- `experiments/run_hybrid.py`

#### Architecture

- Same high-level 3-layer architecture as `v1`.
- Uses GraphSAGE convolution if `torch_geometric` is installed; otherwise a manual normalized GCN fallback.
- Each model still produces 2-class node logits with log-softmax.

#### Features

`v2` uses enriched structural features:

- 11 channels per node.
- Undirected features include normalized degree, clustering coefficient, log-degree, RWSE steps 2–5, triangle count, 4-cycle count, 4-clique count, and core number.
- Directed features include normalized in-degree, out-degree, log-degree ratios, RWSE on the directed graph, motif counts from the undirected projection, and k-core features.

This is implemented in `gnn_model/feature_engineering_v2.py`.

#### Why v2 is better

- More structural descriptors make the GNN better at identifying vertices that participate in cycles.
- RWSE encodes return probability in short random walks, which is strongly correlated with cycle participation.
- Motif counts and core numbers capture local subgraph structure that basic degree features miss.

### 2.3 GNN-KMA-3 (v3)

#### Files

- `gnn_model/model_directed_v3.py`
- `gnn_model/feature_engineering_v3.py`
- `gnn_model/train.py`
- `experiments/run_hybrid.py`

#### Architecture

`v3` is the research-grade model with the following advances:

- `GATv2Conv` attention layers replace hand-written aggregation when `torch_geometric` is available.
- Residual blocks prevent oversmoothing over multiple message-passing steps.
- Separate forward/reverse message passing for directed graphs.
- Global readout via mean pooling is concatenated with per-node embeddings.
- A deeper network: 5 residual GAT layers plus an MLP head.
- Output is a single logit per node; training uses `BCEWithLogitsLoss`.

#### Features

`v3` uses a 16-channel feature vector:

- Directed: normalized in-degree, normalized out-degree, min(in,out)/n, RWSE at steps [2,3,4,6,8,12,16], triangle score, core number, SCC size, nontrivial SCC flag, cycle score, and degree ratio.
- Undirected: similar RWSE and structural encodings adapted for undirected graphs.

This feature set is defined in `gnn_model/feature_engineering_v3.py`.

#### Special training choices

- `AsymmetricFVSLoss` penalizes false positives more heavily than false negatives.
- The validation metric is `topk_precision@8%`, which measures whether the top-ranked GNN predictions include actual FVS nodes.
- Warmup + cosine LR schedule improves stability for the deeper model.

### 2.4 Model Summary Table

| Variant | Model files | Feature set | Output | Training loss | Use in hybrid solver |
|---|---|---|---|---|---|
| GNN-KMA / v1 | `model_undirected.py`, `model_directed.py` | 3-channel degree/cycle proxies | 2-class logits | NLLLoss with class weights | baseline soft-hint KMA |
| GNN-KMA-2 / v2 | `model_undirected_v2.py`, `model_directed_v2.py` | 11-channel structural features, RWSE, motifs | 2-class logits | NLLLoss with class weights | improved soft-hint KMA |
| GNN-KMA-3 / v3 | `model_directed_v3.py` + undirected v3 | 16-channel RWSE + SCC + attention-friendly features | single logits | AsymmetricFVSLoss | research-grade KMA 3 |

## 3. Dataset Generation

### Source data

The GNN training pipeline uses graphs from the repository’s benchmark generation logic.

- Synthetic graphs are generated in `data/synthetic/`.
- Exact-track graphs are selected and exported to the GNN dataset.
- The dataset generator can also reuse `data/pace2022/` graphs.

### Generator script

`gnn_model/dataset_gen.py` is responsible for producing training examples.

- It reads graph files from `data/synthetic/<family>/exact_track/<category>/*.txt`.
- It labels each vertex using a solver (preferably `cpp_engine` for speed, fallback to Python if unavailable).
- It computes node features using either `feature_engineering_v2.py` or `feature_engineering_v3.py`.
- It saves each example as a PyTorch Geometric `.pt` file containing:
  - `data.x` — node features,
  - `data.edge_index` — edge index matrix,
  - `data.y` — ground-truth vertex labels,
  - `data.fvs_size` — the solution size,
  - optional metadata such as `data.family`.

### Output layout

Generated data is stored under:

- `gnn_model/datasets/pt/undirected/...`
- `gnn_model/datasets/pt/directed/...`

### Generation command examples

```bash
python gnn_model/dataset_gen.py --total-undirected 100000 --total-directed 100000
python gnn_model/dataset_gen.py --total-undirected 100 --total-directed 100
python -u gnn_model/dataset_gen.py --total-undirected 100000 --total-directed 100000 --progress-every 5 --max-nodes 300 --solver-mode ma
```

### Dataset distribution

The generator uses weighted categories matching the benchmark distribution.

- Undirected categories: `real_world`, `scale_free`, `small_world`, `random_er`, `grids_trees`.
- Directed categories: `real_world_ego`, `scale_free`, `random_er`, `directed_grids`, `dags`.
- A default `exact_ratio` is used to split graphs into exact and heuristic tracks.

## 4. Training

### Training script

The main training entrypoint is `gnn_model/train.py`.

It offers configuration for:

- graph type: `--type undirected`, `--type directed`, `--type both`
- model variant: `--variant v1`, `--variant v2`, `--variant v3`
- epochs, learning rate, batch size, hidden dimension, dropout, validation ratio, seed.

### Recommended commands

```bash
python gnn_model/train.py --type both --epochs 100 --lr 0.001
python gnn_model/train.py --type both --epochs 300 --hidden 128 --dropout 0.2 --val-ratio 0.2
python gnn_model/train.py --type directed --variant v3 --epochs 200 --hidden 128
```

### Output weights

Trained models are saved to `gnn_model/weights/` such as:

- `undirected_fvs_gcn.pt`
- `directed_fvs_gcn.pt`
- `undirected_fvs_gcn_v2.pt`
- `directed_fvs_gcn_v2.pt`
- `undirected_fvs_gcn_v3.pt`
- `directed_fvs_gcn_v3.pt`

### Training details

- `load_pt_dataset` loads all `.pt` files recursively.
- `stratified_split` holds out validation data per graph family to avoid distribution drift.
- `get_warmup_cosine_scheduler` applies linear warmup followed by cosine decay.
- `compute_metrics` tracks loss, accuracy, recall, F1, and for v3 the `topk_precision@8%` metric.

### Special loss for v3

`gnn_model/train.py` defines `AsymmetricFVSLoss`:

- penalizes false positives harder than false negatives,
- reduces the risk of hard-fixing incorrect vertices in the hybrid solver,
- is more appropriate for FVS because false positives permanently increase solution size.

## 5. Inference and Hybrid Solver Integration

### Core hybrid process

Most solver workflows use `experiments/run_hybrid.py`.

The high-level process is:

1. Read the input graph file and parse it into vertices and edges.
2. Kernelize the graph using `cpp_engine`.
3. Run the selected GNN model on the kernel graph.
4. Convert GNN logits into per-node probabilities.
5. Hard-fix high-confidence vertices above `gnn_threshold` (default `0.65`).
6. Solve the reduced kernel with KMA.
7. Combine forced vertices and KMA solution to produce the final FVS.

### What is kernelization?

Kernelization applies reduction rules that remove vertices that are safe to decide without full search:

- degree-0 and degree-1 removal,
- degree-2 bypass / path compression,
- directed-specific reductions for source/sink and SCC structure.

The result is a smaller kernel graph that preserves the minimum FVS size.

### GNN candidate selection

The hybrid solver does not accept all GNN predictions blindly. It uses a precision-first design:

- only vertices with probability ≥ `gnn_threshold` are hard-fixed,
- only a small percentage of the kernel may be fixed (default `max_fix_fraction=0.08` = 8%),
- if no vertex meets confidence criteria, the solver falls back to pure KMA,
- the remaining kernel is solved by the C++ memetic algorithm.

This avoids catastrophic failure from too many false-positive predictions.

### Models used in inference

`run_hybrid.py` lazily loads the model variant:

- `v1` uses `gnn_model.model_undirected` / `gnn_model.model_directed`
- `v2` uses `gnn_model.model_undirected_v2` / `gnn_model.model_directed_v2`
- `v3` uses `gnn_model.model_directed_v3` and an undirected v3 variant

For undirected `v3`, if weights are missing or the model fails, the code falls back to the `v2` inference path.

### GNN-KMA solver functions

Key hybrid solver exports in `experiments/run_hybrid.py`:

- `gnn_KMA_solve_undirected`
- `gnn_KMA_solve_directed`
- `gnn_KMA2_solve_undirected`
- `gnn_KMA2_solve_directed`
- `gnn_KMA3_solve_undirected`
- `gnn_KMA3_solve_directed`

Each performs:

- `kernelize_*_graph`
- `run_gnn_*_probs`
- `_soft_hint_gnn_kernel_solve`
- final solution reconstruction and metric collection

### Soft-hint design

The `_soft_hint_gnn_kernel_solve` helper implements the GNN + kernel solver coupling.
It takes raw GNN probabilities and only hard-fixes the most confident nodes.
This is the core of the current GNN-KMA design.

## 6. C++ Backend and `cpp_engine`

### Role of C++ code

The C++ engine in `cpp_engine/` implements the core combinatorial algorithms:

- kernelization rules for undirected and directed graphs,
- KMA / memetic algorithm refinement,
- fast solution-checking and solver orchestration.

### Build directories

The repository contains compiled build directories for multiple platforms:

- `cpp_engine/build-win/`
- `cpp_engine/build-linux/`
- `cpp_engine/build-macos/`
- `cpp_engine/build/`

`run_hybrid.py` and `gnn_model/dataset_gen.py` attempt to load `cpp_engine` from these directories first.

### Usage

If `cpp_engine` is available, the scripts use the C++ solver for kernelization and KMA.
If not, Python fallback paths exist but are slower.

### Build scripts

Use the repository’s build scripts if you need to compile the engine:

- Windows: `build_cpp.ps1`
- Linux/macOS: `build_cpp.sh`

## 7. Running Experiments and Tests

### Benchmark scripts

Main experiment entrypoints include:

- `python experiments/run_benchmark_suite.py`
- `python experiments/benchmark_directed.py --algo GNN-KMA --test <graph>`
- `python experiments/benchmark_undirected.py --algo GNN-KMA --test <graph>`
- `python experiments/run_heuristics_track.py`
- `python experiments/run_ablation.py`

`run_benchmark_suite.py` defaults to:

- exact track: `BST`, `IC`, `MA`, `KMA`, `GNN-KMA`
- heuristic track: `MA`, `KMA`, `GNN-KMA`

### Running a single GNN-KMA solve

```bash
python experiments/run_hybrid.py --graph path/to/graph.txt --type undirected --pop 100 --gens 400 --gnn-threshold 0.65
```

### Running tests

There is a dedicated unit test file for the GNN/KMA components:

```bash
python -m pytest tests/test_gnn_components.py -v
python tests/test_gnn_components.py
```

### Important experiment options

Typical options supported by benchmark scripts include:

- `--pop` — population size for MA/KMA/GNN-KMA variants.
- `--gens` — maximum generations.
- `--timeout` — wall-clock timeout in seconds.
- `--gnn-threshold` — GNN probability threshold for candidate hard-fix.
- `--gnn-hidden` — optional override for hidden dimension when loading weights.

## 8. End-to-End Process Summary

### Step 1: Generate graphs

Use the synthetic benchmark generator or existing `data/pace2022` graphs.

```bash
python data/setup_benchmark_inputs.py --total-undirected 100 --total-directed 100
```

### Step 2: Generate GNN training dataset

```bash
python gnn_model/dataset_gen.py --total-undirected 100 --total-directed 100
```

This creates `.pt` files with node features, edges, and FVS labels.

### Step 3: Train the model

```bash
python gnn_model/train.py --type both --epochs 100 --lr 0.001
```

This generates weights in `gnn_model/weights/`.

### Step 4: Run hybrid inference

```bash
python experiments/run_hybrid.py --graph data/synthetic/.../graph.txt --type undirected
```

The solver kernelizes the graph, scores vertices with the GNN, hard-fixes high-confidence candidates, and then runs KMA on the remainder.

### Step 5: Evaluate

Use benchmark scripts like `experiments/benchmark_undirected.py` and `experiments/benchmark_directed.py` to compare solution size and runtime across `MA`, `KMA`, `GNN-KMA`, and `GNN-KMA-2`.

## 9. Notes and Practical Tips

- `GNN-KMA` is a soft-hint hybrid: the network suggests likely FVS vertices, but the final solver still optimizes the reduced kernel.
- `GNN-KMA-2` is the stronger production variant for this repository, because it uses richer features and the same stable solver coupling.
- `GNN-KMA-3` is research-grade and should be used when `torch_geometric` and the trained `v3` weights are available.
- Keep `gnn_threshold` conservative (`>= 0.65`) to avoid too many false-positive hard-fixes.
- Use the C++ `cpp_engine` build if you need performance; Python fallback is available but slow.

## 10. File Map for GNN-KMA

- `gnn_model/dataset_gen.py` — generate `.pt` datasets.
- `gnn_model/train.py` — train GNNs, save weights.
- `gnn_model/model_undirected.py` / `model_directed.py` — v1 models.
- `gnn_model/model_undirected_v2.py` / `model_directed_v2.py` — v2 models.
- `gnn_model/model_directed_v3.py` — v3 models.
- `gnn_model/feature_engineering_v2.py` — v2 feature extraction.
- `gnn_model/feature_engineering_v3.py` — v3 feature extraction.
- `experiments/run_hybrid.py` — hybrid solver integration and GNN-KMA chaining.
- `cpp_engine/` — C++ kernelization and memetic algorithm backend.

---

This file should contain the full path from dataset generation to training and solver implementation for GNN-KMA variants 1, 2, and 3 in the repository.
