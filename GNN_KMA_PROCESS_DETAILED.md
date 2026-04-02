# GNN-KMA Process: Complete Technical Reference

This document explains the full GNN-KMA pipeline in this repository, from graph sources to `.pt` creation, training, inference, and benchmark output.

It is based on the current code in:
- `gnn_model/dataset_gen.py`
- `gnn_model/train.py`
- `gnn_model/model_directed.py`
- `gnn_model/model_undirected.py`
- `experiments/run_hybrid.py`
- `experiments/benchmark_directed.py`

## 1. End-to-End Flow

1. Source graphs are taken from `data/synthetic/<family>/exact_track/<category>/*.txt`.
2. `gnn_model/dataset_gen.py` parses each graph, normalizes edges, computes node features, solves labels using IC, and saves one `.pt` file per graph.
3. `gnn_model/train.py` loads all `.pt` recursively, splits into train/val, trains GNN models, and saves weights.
4. `experiments/run_hybrid.py` loads weights, kernelizes graph, runs GNN on kernel, picks candidates, applies candidate-guided reduction, then runs KMA.
5. `experiments/benchmark_directed.py` calls GNN-KMA per graph (with timeout), verifies DFVS validity, and writes CSV rows.

## 2. Data Source Stage (Exact Inputs)

### 2.1 Input roots and families

- Root: `data/synthetic`
- Family choices: `undirected`, `directed`
- Track used for GNN dataset generation: `exact_track` only

### 2.2 Exact categories

- Undirected categories:
- `real_world`
- `scale_free`
- `small_world`
- `random_er`
- `grids_trees`

- Directed categories:
- `real_world_ego`
- `scale_free`
- `random_er`
- `directed_grids`
- `dags`

### 2.3 File selection

- Dataset generation reads only `*.txt` files from each exact category folder.
- Missing category folders are skipped with warning.

## 3. Dataset Generation Stage (`dataset_gen.py`)

## 3.1 Main command

```bash
python gnn_model/dataset_gen.py --family all --progress-every 10
```

CLI options:
- `--family {all,undirected,directed}` (default `all`)
- `--force` deletes existing `.pt` in each bucket before regeneration
- `--clean-root` removes entire PT output root before regeneration
- `--progress-every N` prints bucket progress every `N` created files

### 3.2 Parser and normalization

For each source file:
- Parses edge-list text lines and optional `p ...` header for `n` hint.
- Ignores comment lines starting with `#`, `%`, or `c `.
- Converts 1-indexed vertex ids to 0-index if min vertex id is 1.
- Normalizes edges:
- Directed: keeps `(u, v)` orientation, removes self-loops.
- Undirected: canonicalizes edge order `(min(u,v), max(u,v))`, removes duplicates/self-loops.

### 3.3 Labeling with IC solver

Ground-truth labels are produced with IC:
- Undirected: `cpp_engine.solve_undirected_IC(n, edges)`
- Directed: `cpp_engine.solve_directed_IC(n, edges)`

Timeout behavior:
- Each graph is solved in a subprocess via `fork` context.
- Timeout per graph: `SOLVER_TIMEOUT_SECONDS = 60`.
- Timeout graphs are skipped and not saved as `.pt`.

Fallback if `cpp_engine` import fails:
- Uses Python fallback solvers (`solve_undirected`, `solve_directed`) which are slower and approximate.

### 3.4 Feature construction

Undirected features (`x` with 3 channels per node):
- `degree_norm = degree(v) / max(n-1,1)`
- `clustering_coeff(v)`
- `log_degree_norm = log(degree(v)+1) / log(n+1)`

Directed features (`x` with 3 channels per node):
- `in_deg_norm = in_degree(v) / max(n-1,1)`
- `out_deg_norm = out_degree(v) / max(n-1,1)`
- `min_deg_norm = min(in_degree(v), out_degree(v)) / max(n-1,1)`

### 3.5 Label tensor and edge index

- `y` is binary class tensor (`torch.long`):
- `0` means not in FVS
- `1` means in FVS (IC-labeled)

- `edge_index`:
- Undirected: both directions are stored (`ei` plus `ei.flip(0)`).
- Directed: stored as-is, one directed edge per entry.

### 3.6 What is stored in each `.pt`

Each file is a `torch_geometric.data.Data` object with:
- `data.x`
- `data.edge_index`
- `data.y`
- `data.fvs_size` (number of IC-selected vertices)
- `data.family` (`undirected` or `directed`)
- `data.track` (always `exact_track`)
- `data.category`
- `data.source_file` (original `.txt` filename)

### 3.7 Output layout and mapping

Output root:
- `gnn_model/datasets/pt`

Path mapping:
- Input file:
- `data/synthetic/<family>/exact_track/<category>/<name>.txt`
- Output file:
- `gnn_model/datasets/pt/<family>/exact_track/<category>/<name>.pt`

### 3.8 Re-run behavior and cleanup

- Existing `.pt` with matching stem are reused unless `--force` is used.
- Non-source stale `.pt` files are trimmed from bucket.
- `heuristic_track` outputs under `gnn_model/datasets/pt/<family>/heuristic_track` are removed by `_remove_heuristic_outputs`.

## 4. Training Stage (`train.py`)

### 4.1 Main command

```bash
python gnn_model/train.py --type directed --epochs 100 --lr 0.001 --hidden 64 --dropout 0.3
```

CLI options:
- `--type {undirected,directed,both}` default `both`
- `--epochs` default `100`
- `--lr` default `0.001`
- `--hidden` default `64`
- `--dropout` default `0.3`
- `--val-ratio` default `0.2`
- `--log-every` default `1`
- `--data-root` default `gnn_model/datasets/pt`

### 4.2 Dataset loading

- Loads all `.pt` recursively from:
- `data_root/undirected` when training undirected
- `data_root/directed` when training directed

- Uses `torch.load(..., weights_only=False)` for each file.

### 4.3 Train/val split

- Uses in-place Python `random.shuffle` on dataset list.
- Split index: `int(len(dataset) * (1 - val_ratio))`.
- No fixed random seed is set in `train.py`.

### 4.4 Model architectures

Undirected model (`UndirectedFVSNet`):
- 3 graph conv layers (GraphSAGE when PyG available, manual GCN fallback otherwise)
- BN + ReLU + Dropout between layers
- MLP head to 2 classes
- Output uses `log_softmax`

Directed model (`DirectedFVSNet`):
- Custom `DirectedConvLayer` with separate incoming/outgoing aggregation
- 3 directed conv layers with BN/ReLU/Dropout
- MLP head to 2 classes
- Output uses `log_softmax`

### 4.5 Loss and class weighting

- Loss: `nn.NLLLoss(weight=class_weights)`
- Class weights computed from each graph labels:
- `w_class = n / (2 * class_count)`
- Returns `[w_other, w_fvs]`

### 4.6 Optimizer and scheduler

- Optimizer: `Adam(lr=args.lr, weight_decay=1e-4)`
- Scheduler: `CosineAnnealingLR(T_max=epochs, eta_min=1e-5)`

### 4.7 Validation, metrics, and early stopping

- Validation runs every 5 epochs (and final epoch).
- Metrics tracked: loss, accuracy, precision, recall, F1.
- Model checkpoint criterion: best validation F1.
- Early stopping:
- `patience = 20` validation checks without F1 improvement.

### 4.8 Weight outputs

- Undirected weights: `gnn_model/weights/undirected_fvs_gcn.pt`
- Directed weights: `gnn_model/weights/directed_fvs_gcn.pt`

Saved object format in training code:
- `state_dict` only (raw `torch.save(model.state_dict(), path)`).

## 5. Inference + Hybrid Solve Stage (`run_hybrid.py`)

### 5.1 Main command

```bash
python experiments/run_hybrid.py --graph <graph-file> --type directed --pop 100 --gens 500 --threshold 0.2
```

CLI options:
- `--graph` single graph file path
- `--type {undirected,directed}` default `undirected`
- `--pop` default `60`
- `--gens` default `300`
- `--threshold` default `0.2`
- `--gnn-hidden` optional hidden override
- `--compare` run pure KMA baseline too

Important:
- `--graph` is for one file, not a directory.

### 5.2 Weight loading and hidden dimension handling

Weights used:
- Undirected: `gnn_model/weights/undirected_fvs_gcn.pt`
- Directed: `gnn_model/weights/directed_fvs_gcn.pt`

Checkpoint compatibility logic:
- Accepts raw `state_dict` or wrapped dictionaries with keys `state_dict` or `model_state_dict`.
- Infers `hidden_dim` from checkpoint tensor shapes when possible.
- Builds model with inferred hidden size before loading weights.

### 5.3 Kernelization before GNN

The solver kernelizes first, then runs GNN on kernel graph.

Undirected kernelization:
- removes degree 0/1 vertices
- degree-2 bypass rule
- self-loop vertices are forced into solution

Directed kernelization:
- applies directed reduction rules on in/out degrees
- removes vertices not in non-trivial SCCs
- self-loop vertices are forced

Outputs of kernelization:
- `kernel_n`, `kernel_edges`, `forced`, `new_to_old`

### 5.4 GNN feature extraction at inference

Inference feature formulas match dataset generation formulas exactly for both graph types.

### 5.5 Candidate selection policy

After model logits:
- Compute positive class probabilities: `pos_probs = logits.exp()[:, 1]`.

Selection in `_pick_gnn_candidates_from_probs`:
- Primary: all vertices with `p >= threshold`.
- If none selected: top-k fallback where `k = ceil(1% * n)` (minimum 1).
- If too many selected: cap to top-k where `k = ceil(15% * n)`.

Returned mode labels:
- `threshold`
- `topk-fallback`
- `threshold-capped`

### 5.6 How GNN now guides KMA

For both undirected and directed:
1. Build `fixed_kernel` from GNN candidates.
2. Remove fixed vertices from kernel graph.
3. Reindex remaining kernel graph.
4. Run KMA on reduced graph.
5. Merge solution parts:
- `forced` from kernel rules
- `fixed_kernel` from GNN
- reduced KMA result mapped back through reindexing
6. Map kernel indices to original graph using `new_to_old`.

So final result is a union of forced + GNN-fixed + KMA-refined vertices.

### 5.7 Fallback behavior

If PyTorch/models/weights are unavailable:
- GNN returns `None`.
- Pipeline falls back to pure KMA path.

If GNN is available but predicts no fixed hints:
- Runs standard KMA on full kernel graph.

## 6. Directed Benchmark Stage (`benchmark_directed.py`)

### 6.1 Main command

```bash
python experiments/benchmark_directed.py --algo GNN-KMA --test data/pace2022/
```

Current CLI options include:
- `--algo`
- `--test`
- `--results-dir`
- `--output`
- `--pop`
- `--gens`
- `--quiet`

### 6.2 Important limitation

`benchmark_directed.py` does not currently expose GNN-specific CLI args such as:
- `--gnn-threshold`
- `--gnn-hidden`

That is why commands like:

```bash
python experiments/benchmark_directed.py --algo GNN-KMA --test data/pace2022/ --gnn-threshold 0.2
```

fail with argument parsing error.

### 6.3 Timeout behavior in benchmark

Dynamic per-graph timeout for each algorithm run:
- `n <= 50`: 100s
- `50 < n <= 200`: 500s
- `n > 200`: 600s

### 6.4 Result persistence and schema

Per algorithm CSV path:
- `results/directed_<ALGO>.csv`

Row schema for single-algorithm CSV rows:
- `file`
- `n`
- `m`
- `FVS_size`
- `runtime` (seconds)
- `validity`

Behavior:
- Appends one row immediately after each file+algorithm run.
- Skips rerun if same filename already exists in that algorithm CSV.

### 6.5 DFVS validity check

`verify_dfvs` removes proposed FVS vertices and checks acyclicity by DFS coloring:
- WHITE (0), GRAY (1), BLACK (2)
- Detects back-edge in residual graph.

## 7. What Is Marked, Stored, and Mapped

Label marking:
- Nodes in IC solution are marked `1` in `data.y`; all others `0`.

Metadata marking in each `.pt`:
- `family`, `track`, `category`, `source_file`, `fvs_size`.

Mapping in kernel stage:
- `new_to_old` maps kernel index to original graph index.
- Additional reduced graph reindex map is created after fixing GNN candidates.

Final map chain (conceptual):
- reduced-solver index -> kernel index -> original vertex index.

## 8. Recommended Operational Sequence

1. Build/verify `cpp_engine` import works.
2. Generate `.pt` from exact inputs:

```bash
python gnn_model/dataset_gen.py --family all --force
```

3. Train model weights:

```bash
python gnn_model/train.py --type both --epochs 100 --hidden 64
```

4. Test single-file hybrid directly (with explicit threshold):

```bash
python experiments/run_hybrid.py --graph data/pace2022/h_001 --type directed --threshold 0.2 --pop 100 --gens 500
```

5. Run benchmark batch with GNN-KMA:

```bash
python experiments/benchmark_directed.py --algo GNN-KMA --test data/pace2022/
```

## 9. Common Failure Cases and Meanings

- `Weights not found ...`:
- Training weights were not produced in `gnn_model/weights/`.

- `PyTorch/GNN not available`:
- Missing runtime dependencies or import path issue.

- `solver exceeded 60s` during dataset generation:
- IC labeling timed out for that graph; graph is skipped.

- `unrecognized arguments: --gnn-threshold` in benchmark command:
- `benchmark_directed.py` does not support that CLI flag currently.

## 10. Current Defaults Snapshot

Dataset generation defaults:
- family: `all`
- progress-every: `10`
- solver timeout per graph: `60s`

Training defaults:
- type: `both`
- epochs: `100`
- lr: `0.001`
- hidden: `64`
- dropout: `0.3`
- val-ratio: `0.2`

Hybrid (`run_hybrid.py`) defaults:
- pop: `60`
- gens: `300`
- threshold: `0.2`

Directed benchmark defaults:
- pop: `50`
- gens: `200`
- timeout by graph size as listed above

