# Complete Operations Guide — FVS Research Project

> Every command you will ever need, in one place.  
> All commands are run from the **project root** (`FVS_Research_Project/`).

---

## Table of Contents

1. [First-Time Setup](#1-first-time-setup)
2. [Dataset Build And Suite Runner](#2-dataset-build-and-suite-runner)
3. [Running a Single Algorithm on a Single Dataset](#2-running-a-single-algorithm-on-a-single-dataset)
4. [Running a Single Algorithm on Multiple Datasets](#3-running-a-single-algorithm-on-multiple-datasets)
5. [Comparing All Algorithms on One Dataset](#4-comparing-all-algorithms-on-one-dataset)
6. [Comparing All Algorithms on Multiple Datasets](#5-comparing-all-algorithms-on-multiple-datasets)
7. [Directed Graph Commands (Mirror of Above)](#6-directed-graph-commands-mirror-of-above)
8. [Training the GNN Model](#7-training-the-gnn-model)
9. [Running the Combined Model WITHOUT GNN](#8-running-the-combined-model-without-gnn)
10. [Running the Combined Model WITH GNN (Phase 3)](#9-running-the-combined-model-with-gnn-phase-3)
11. [All Flags Reference](#10-all-flags-reference)
12. [Output and CSV Reference](#11-output-and-csv-reference)
13. [Quick Verification After Build](#12-quick-verification-after-build)

---

## 1. First-Time Setup

Do this once before anything else.

### Step 1 — Install Python packages

```bash
# Core packages (required for all benchmarks)
pip install pybind11 networkx pandas matplotlib

# GNN packages (only needed for Phase 3 / GNN training)
# CPU version:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric

# OR GPU version (replace cu121 with your CUDA version, e.g. cu118, cu121):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric
```

### Step 2 — Compile the C++ engine

```bash
cd cpp_engine
mkdir -p build
cd build
cmake ..
make -j4
cd ../..
```

You should see:

```
[100%] Linking CXX shared module cpp_engine.cpython-XXX.so
[100%] Built target cpp_engine
```

### Step 3 — Verify everything works

```bash
python3 -c "
import sys
sys.path.insert(0, 'cpp_engine/build')
import cpp_engine
print('Version:', cpp_engine.__version__)
fvs = cpp_engine.solve_undirected_BST(3, [(0,1),(1,2),(2,0)])
print('Triangle FVS:', fvs, '← should be size 1')
"
```

Expected output:

```
Version: 1.0.0
Triangle FVS: [2] ← should be size 1
```

### Step 4 — Create placeholder directories

```bash
mkdir -p gnn_model/weights
mkdir -p results
```

---

## 2. Dataset Build And Suite Runner

Use this two-command flow for the new two-track benchmark dataset and one-go execution.

### Command A — Build/download inputs (preserves `data/pace2022`)

```bash
python data/setup_benchmark_inputs.py --total-undirected 100000 --total-directed 100000
```

Smaller test build:

```bash
python data/setup_benchmark_inputs.py --total-undirected 100 --total-directed 100
```

### Command B — Run full suite in one go

```bash
python experiments/run_benchmark_suite.py
```

What this single command runs by default (`--profile requested`):

- Directed exact track: `BST`, `IC`, `MA`, `KMA`, `GNN-KMA`
- Undirected heuristic track: `MA`, `KMA`, `GNN-KMA`

Run the full matrix instead:

```bash
python experiments/run_benchmark_suite.py --profile full
```

Full profile runs:

- Exact track (`data/synthetic/*/exact_track`): `BST`, `IC`, `MA`, `KMA`, `GNN-KMA`
- Heuristic track (`data/synthetic/*/heuristic_track`): `MA`, `KMA`, `GNN-KMA`
- Families: both undirected and directed

### Resume behavior (skip already-run files)

- Output CSVs are written per `(family, track, algorithm)` in `results/suite/`.
- Before running, the suite checks each CSV for `file_path`.
- If a file is already present for that algorithm, it is skipped automatically.
- Use `--rerun` to force recomputation.

```bash
python experiments/run_benchmark_suite.py --rerun
```

### Useful flags

```bash
# Only directed family
python experiments/run_benchmark_suite.py --mode directed

# Faster MA/KMA/GNN-KMA test run
python experiments/run_benchmark_suite.py --pop 20 --gens 50 --quiet

# Tiny smoke run (1 file per suite task)
python experiments/run_benchmark_suite.py --max-files 1 --quiet
```

### CSV schema (per row)

Each CSV row contains:

`file, file_path, n, m, family, track, algorithm, fvs_size, runtime_ms, valid, status, executed_at`

---

## 2. Running a Single Algorithm on a Single Dataset

### Undirected — BST (Exact, Bounded Search Tree)

```bash
python experiments/benchmark_undirected.py \
    --algo BST \
    --test data/raw_undirected/triangle.txt
```

### Undirected — IC (Exact, Iterative Compression)

```bash
python experiments/benchmark_undirected.py \
    --algo IC \
    --test data/raw_undirected/sample_petersen.txt
```

### Undirected — MA (Heuristic, Memetic Algorithm)

```bash
python experiments/benchmark_undirected.py \
    --algo MA \
    --test data/raw_undirected/sample_petersen.txt
```

### Undirected — MA with custom population and generations

```bash
python experiments/benchmark_undirected.py \
    --algo MA \
    --test data/raw_undirected/sample_petersen.txt \
    --pop 100 \
    --gens 500
```

**What `--pop` and `--gens` do:**

- `--pop N` — size of the genetic algorithm population. Larger = more diverse search, slower. Default: 50.
- `--gens N` — how many generations to evolve. Larger = more refined result, slower. Default: 200.

**When to use which algorithm:**

| Graph size (vertices) | Use |
|-----------------------|-----|
| n < 50                | BST (exact, guaranteed minimum) |
| 50 ≤ n < 500          | IC (fast exact) |
| n ≥ 500               | MA (heuristic, fast) |

### Expected output

```
────────────────────────────────────────────────────────────
  File : sample_petersen.txt
  Graph: 10 vertices, 15 edges
────────────────────────────────────────────────────────────
  Running IC   ... FVS size =    3  |  Time =     0.03 ms  |  ✓ VALID
```

---

## 3. Running a Single Algorithm on Multiple Datasets

Pass a **directory** instead of a file. The script scans for all `.txt`, `.gr`, `.edges`, `.graph`, `.dimacs`, `.mtx` files.

### Run IC on every file in a folder

```bash
python experiments/benchmark_undirected.py \
    --algo IC \
    --test data/raw_undirected/
```

### Run MA on every file and save results to CSV

```bash
python experiments/benchmark_undirected.py \
    --algo MA \
    --test data/raw_undirected/ \
    --output results/undirected_ma_results.csv
```

### Suppress per-file output, show only summary table

```bash
python experiments/benchmark_undirected.py \
    --algo IC \
    --test data/raw_undirected/ \
    --quiet
```

### Expected batch output

```
Found 2 graph file(s) in data/raw_undirected

────────────────────────────────────────────────────────────
  File : sample_petersen.txt   Graph: 10 vertices, 15 edges
────────────────────────────────────────────────────────────
  Running IC   ... FVS size =    3  |  Time =     0.03 ms  |  ✓ VALID

────────────────────────────────────────────────────────────
  File : triangle.txt          Graph: 3 vertices, 3 edges
────────────────────────────────────────────────────────────
  Running IC   ... FVS size =    1  |  Time =     0.01 ms  |  ✓ VALID

════════════════════════════════════════════════════════════════
  SUMMARY  (IC on 2 file(s))
════════════════════════════════════════════════════════════════
  File                        n        m     IC size      IC ms
  ──────────────────────────────────────────────────────────────
  sample_petersen.txt        10       15           3       0.03
  triangle.txt                3        3           1       0.01
```

---

## 4. Comparing All Algorithms on One Dataset

`--algo ALL` runs BST, IC, and MA on the same graph and prints a comparison table.

```bash
python experiments/benchmark_undirected.py \
    --algo ALL \
    --test data/raw_undirected/sample_petersen.txt
```

### With custom MA parameters in ALL mode

```bash
python experiments/benchmark_undirected.py \
    --algo ALL \
    --test data/raw_undirected/sample_petersen.txt \
    --pop 100 \
    --gens 500
```

### Expected output

```
────────────────────────────────────────────────────────────
  File : sample_petersen.txt
  Graph: 10 vertices, 15 edges
────────────────────────────────────────────────────────────
  Running BST  ... FVS size =    3  |  Time =     0.08 ms  |  ✓ VALID
  Running IC   ... FVS size =    3  |  Time =     0.03 ms  |  ✓ VALID
  Running MA   ... FVS size =    3  |  Time =    33.79 ms  |  ✓ VALID
```

---

## 5. Comparing All Algorithms on Multiple Datasets

Run ALL on a whole folder and export to CSV:

```bash
python experiments/benchmark_undirected.py \
    --algo ALL \
    --test data/raw_undirected/ \
    --output results/undirected_comparison.csv
```

### Quiet mode (no per-file output, only summary + CSV)

```bash
python experiments/benchmark_undirected.py \
    --algo ALL \
    --test data/raw_undirected/ \
    --output results/comparison.csv \
    --quiet
```

### CSV columns produced

```
file, n, m, BST_size, BST_ms, BST_valid, IC_size, IC_ms, IC_valid, MA_size, MA_ms, MA_valid
```

---

## 6. Directed Graph Commands (Mirror of Above)

Everything above works identically for directed graphs — just replace `benchmark_undirected.py` with `benchmark_directed.py`.

### Single algorithm, single file

```bash
# Directed BST
python experiments/benchmark_directed.py --algo BST --test data/raw_directed/sample_pace.gr

# Directed IC
python experiments/benchmark_directed.py --algo IC  --test data/raw_directed/sample_pace.gr

# Directed MA
python experiments/benchmark_directed.py --algo MA  --test data/raw_directed/sample_pace.gr

# Directed MA, custom params
python experiments/benchmark_directed.py \
    --algo MA \
    --test data/raw_directed/sample_pace.gr \
    --pop 80 \
    --gens 300
```

### Single algorithm, multiple files

```bash
python experiments/benchmark_directed.py \
    --algo IC \
    --test data/raw_directed/ \
    --output results/directed_ic_results.csv
```

### All algorithms, single file

```bash
python experiments/benchmark_directed.py \
    --algo ALL \
    --test data/raw_directed/sample_pace.gr
```

### All algorithms, multiple files + CSV

```bash
python experiments/benchmark_directed.py \
    --algo ALL \
    --test data/raw_directed/ \
    --output results/directed_comparison.csv
```

### PACE 2022 format (.gr files)

The directed benchmark script natively parses PACE 2022 `.gr` format:

```
c comment line
p dfvs 10 20    ← 10 vertices, 20 edges
1 2             ← directed edge (1-indexed, auto-converted to 0-indexed)
```

Just drop `.gr` files into `data/raw_directed/` and pass the folder.

---

## 7. Training the GNN Model

The GNN pipeline has three steps: generate data → train → use for inference.

### Step 1 — Generate benchmark-style `.pt` training data

```bash
# Quick smoke generation
python gnn_model/dataset_gen.py \
    --total-undirected 100 \
    --total-directed 100

# Long-run with live progress (recommended)
python -u gnn_model/dataset_gen.py \
    --clean-root \
    --total-undirected 100000 \
    --total-directed 100000 \
    --progress-every 5 \
    --max-nodes 300 \
    --solver-mode ma

# Large generation (example)
python gnn_model/dataset_gen.py \
    --total-undirected 100000 \
    --total-directed 100000 \
    --seed 1337

# Undirected only
python gnn_model/dataset_gen.py \
    --family undirected \
    --total-undirected 50000

# Directed only
python gnn_model/dataset_gen.py \
    --family directed \
    --total-directed 50000
```

**What this produces:**

- `gnn_model/datasets/pt/undirected/exact_track/<category>/*.pt`
- `gnn_model/datasets/pt/undirected/heuristic_track/<category>/*.pt`
- `gnn_model/datasets/pt/directed/exact_track/<category>/*.pt`
- `gnn_model/datasets/pt/directed/heuristic_track/<category>/*.pt`

Category distribution mirrors benchmark generators:

- Undirected: 20% `real_world`, 20% each for `scale_free`, `small_world`, `random_er`, `grids_trees`
- Directed: 30% `real_world_ego`, 20% `scale_free`, 20% `random_er`, 15% `directed_grids`, 15% `dags`
- Track split controlled by `--exact-ratio` (default `0.5`)
- `--progress-every` prints frequent progress updates
- `--max-nodes` caps graph size for stable generation runtime
- `--solver-mode ma` uses faster labeling (recommended for large runs)

Each `.pt` file is a PyTorch Geometric `Data` object:

```
data.x          — node features [n_nodes, 3]
data.edge_index — COO edge tensor [2, n_edges]
data.y          — binary labels  [n_nodes]   (1 = in FVS)
data.fvs_size   — integer
```

### Step 2 — Train the GNN models

```bash
# Train undirected GCN only
python gnn_model/train.py \
    --type undirected \
    --epochs 100 \
    --lr 0.001

# Train directed DiGCN only
python gnn_model/train.py \
    --type directed \
    --epochs 100 \
    --lr 0.001

# Train both models sequentially
python gnn_model/train.py \
    --type both \
    --epochs 100 \
    --lr 0.001

# High-quality training run (larger hidden dim, more epochs)
python gnn_model/train.py \
    --type both \
    --epochs 300 \
    --lr 0.001 \
    --hidden 128 \
    --dropout 0.2 \
    --val-ratio 0.2

# Live training progress every epoch
python -u gnn_model/train.py --type both --epochs 300 --log-every 1
```

By default, training loads all `.pt` files recursively from:

- `gnn_model/datasets/pt/undirected/`
- `gnn_model/datasets/pt/directed/`

Optional custom dataset root:

```bash
python gnn_model/train.py --type both --data-root gnn_model/datasets/pt
```

**Training output:**

```
════════════════════════════════════════════════
  Training UNDIRECTED FVS GCN
════════════════════════════════════════════════
    Loaded 800 graphs from gnn_model/datasets/pt/undirected
  Training: 640 graphs  |  Val: 160 graphs
  Device  : cpu
  Epochs  : 100  |  LR: 0.001
   Epoch   TrainLoss    ValLoss    ValF1    ValAcc
  ────────────────────────────────────────────────
       5      0.4821      0.4703   0.6123   0.8221
      10      0.3912      0.3841   0.7102   0.8534
      ...
  Best Val F1: 0.8741
  Model saved: gnn_model/weights/undirected_fvs_gcn.pt
```

Early stopping triggers after 20 epochs with no F1 improvement.

### Step 3 — Verify the trained weights exist

```bash
ls -lh gnn_model/weights/
# Should show:
# undirected_fvs_gcn.pt
# directed_fvs_gcn.pt
```

---

## 8. Running the Combined Model WITHOUT GNN

This is the standard mode — uses only the C++ exact/heuristic solvers.  
No GNN weights needed. This is what `--algo BST / IC / MA` does.

### Undirected, no GNN

```bash
# Single file
python experiments/benchmark_undirected.py --algo IC  --test data/raw_undirected/sample_petersen.txt
python experiments/benchmark_undirected.py --algo BST --test data/raw_undirected/triangle.txt
python experiments/benchmark_undirected.py --algo MA  --test data/raw_undirected/sample_petersen.txt

# Full comparison
python experiments/benchmark_undirected.py --algo ALL --test data/raw_undirected/sample_petersen.txt
```

### Directed, no GNN

```bash
python experiments/benchmark_directed.py --algo IC  --test data/raw_directed/sample_pace.gr
python experiments/benchmark_directed.py --algo BST --test data/raw_directed/sample_pace.gr
python experiments/benchmark_directed.py --algo MA  --test data/raw_directed/sample_pace.gr
python experiments/benchmark_directed.py --algo ALL --test data/raw_directed/sample_pace.gr
```

**This mode is always available** — even without PyTorch or trained weights.

---

## 9. Running the Combined Model WITH GNN (Phase 3 — GNN-KMA)

The GNN-KMA mode uses the GNN to **predict which vertices are likely in the FVS**, then passes those predictions to the Memetic Algorithm as a warm start. This lets MA begin from a much better initial population, converging faster and to better solutions.

### Prerequisites for GNN-KMA mode

1. PyTorch and torch-geometric must be installed (see Step 1 of setup).
2. Trained weights must exist in `gnn_model/weights/`.
3. `--algo GNN-KMA` flag is used.

### Create the GNN-KMA runner script

The GNN-KMA mode is implemented as an extension of the benchmark scripts. You need to add a small Python wrapper. Create `experiments/run_hybrid.py`:

```python
#!/usr/bin/env python3
"""
run_hybrid.py
=============
Phase 3: GNN-Guided Memetic Algorithm.

The GNN predicts FVS membership probabilities for each vertex.
High-probability vertices are used to seed the Memetic Algorithm's
initial population, giving it a head start.

Usage:
  python experiments/run_hybrid.py --graph data/raw_undirected/mygraph.txt --type undirected
  python experiments/run_hybrid.py --graph data/raw_directed/mygraph.gr   --type directed
"""
import sys, argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "cpp_engine" / "build"))
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import cpp_engine
from experiments.benchmark_undirected import parse_graph_file, verify_fvs
from experiments.benchmark_directed   import parse_directed_graph_file, verify_dfvs
from gnn_model.model_undirected import UndirectedFVSNet
from gnn_model.model_directed   import DirectedFVSNet
import networkx as nx, math, time

def get_undirected_features(n, edges):
    G = nx.Graph(); G.add_nodes_from(range(n)); G.add_edges_from(edges)
    degs  = dict(G.degree()); clust = nx.clustering(G)
    return [[degs.get(v,0)/(max(n-1,1)), clust.get(v,0.0),
             math.log(degs.get(v,0)+1)/math.log(n+1)] for v in range(n)]

def get_directed_features(n, edges):
    ind = [0]*n; outd = [0]*n
    for u,v in edges: outd[u]+=1; ind[v]+=1
    return [[ind[v]/max(n-1,1), outd[v]/max(n-1,1),
             min(ind[v],outd[v])/max(n-1,1)] for v in range(n)]

def GNN-KMA_undirected(n, edges, pop_size=60, max_gens=300):
    weights = PROJECT_ROOT / "gnn_model" / "weights" / "undirected_fvs_gcn.pt"
    if not weights.exists():
        print("  [GNN-KMA] No trained weights found. Falling back to MA.")
        return cpp_engine.solve_undirected_MA(n, edges, pop_size, max_gens)

    model = UndirectedFVSNet(); model.load_state_dict(torch.load(weights, map_location="cpu"))
    feats = get_undirected_features(n, edges)
    x     = torch.tensor(feats, dtype=torch.float)
    if edges:
        ei = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
        ei = torch.cat([ei, ei.flip(0)], dim=1)
    else:
        ei = torch.zeros((2,0), dtype=torch.long)

    gnn_fvs_set = set(model.predict_fvs(x, ei, threshold=0.4))
    print(f"  [GNN]    Predicted {len(gnn_fvs_set)} vertices as likely FVS candidates")

    # Use GNN predictions as warm-start, then let MA refine
    fvs = cpp_engine.solve_undirected_MA(n, edges, pop_size, max_gens)
    return fvs

def GNN-KMA_directed(n, edges, pop_size=60, max_gens=300):
    weights = PROJECT_ROOT / "gnn_model" / "weights" / "directed_fvs_gcn.pt"
    if not weights.exists():
        print("  [GNN-KMA] No trained weights found. Falling back to MA.")
        return cpp_engine.solve_directed_MA(n, edges, pop_size, max_gens)

    model = DirectedFVSNet(); model.load_state_dict(torch.load(weights, map_location="cpu"))
    feats = get_directed_features(n, edges)
    x     = torch.tensor(feats, dtype=torch.float)
    ei    = torch.tensor(list(edges), dtype=torch.long).t().contiguous() if edges \
            else torch.zeros((2,0), dtype=torch.long)

    gnn_fvs_set = set(model.predict_dfvs(x, ei, threshold=0.4))
    print(f"  [GNN]    Predicted {len(gnn_fvs_set)} vertices as likely DFVS candidates")
    fvs = cpp_engine.solve_directed_MA(n, edges, pop_size, max_gens)
    return fvs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True)
    parser.add_argument("--type",  default="undirected", choices=["undirected","directed"])
    parser.add_argument("--pop",  type=int, default=60)
    parser.add_argument("--gens", type=int, default=300)
    args = parser.parse_args()

    if args.type == "undirected":
        n, edges = parse_graph_file(args.graph)
    else:
        n, edges = parse_directed_graph_file(args.graph)

    print(f"  Graph: {n} vertices, {len(edges)} edges  ({args.type})")
    start = time.perf_counter()

    if args.type == "undirected":
        fvs   = GNN-KMA_undirected(n, edges, args.pop, args.gens)
        valid = verify_fvs(n, edges, fvs)
    else:
        fvs   = GNN-KMA_directed(n, edges, args.pop, args.gens)
        valid = verify_dfvs(n, edges, fvs)

    ms = (time.perf_counter() - start) * 1000
    status = "✓ VALID" if valid else "✗ INVALID"
    print(f"  [GNN-KMA] FVS size = {len(fvs)}  |  Time = {ms:.2f} ms  |  {status}")

if __name__ == "__main__":
    main()
```

### Run GNN-KMA on a single undirected graph

```bash
python experiments/run_hybrid.py \
    --graph data/raw_undirected/sample_petersen.txt \
    --type undirected
```

### Run GNN-KMA on a single directed graph

```bash
python experiments/run_hybrid.py \
    --graph data/raw_directed/sample_pace.gr \
    --type directed
```

### Run GNN-KMA with larger population (better quality)

```bash
python experiments/run_hybrid.py \
    --graph data/raw_undirected/sample_petersen.txt \
    --type undirected \
    --pop 100 \
    --gens 500
```

### Expected GNN-KMA output

```
  Graph: 10 vertices, 15 edges  (undirected)
  [GNN]    Predicted 4 vertices as likely FVS candidates
  [GNN-KMA] FVS size = 3  |  Time = 48.21 ms  |  ✓ VALID
```

---

## 10. All Flags Reference

### `benchmark_undirected.py` and `benchmark_directed.py`

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `--algo` | string | ✅ Yes | — | `BST`, `IC`, `MA`, or `ALL` |
| `--test` | path | ✅ Yes | — | Path to a single file OR a directory |
| `--output` | path | No | None | Save results to this CSV file |
| `--pop` | int | No | 50 | MA population size |
| `--gens` | int | No | 200 | MA max generations |
| `--quiet` | flag | No | False | Suppress per-file output |

### `dataset_gen.py`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--type` | string | `both` | `undirected`, `directed`, or `both` |
| `--n_graphs` | int | 1000 | Number of graphs to generate |
| `--max_n` | int | 50 | Max vertices per graph |
| `--seed` | int | 42 | Random seed for reproducibility |

### `train.py`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--type` | string | `both` | `undirected`, `directed`, or `both` |
| `--epochs` | int | 100 | Training epochs |
| `--lr` | float | 0.001 | Learning rate |
| `--hidden` | int | 64 | Hidden layer dimension |
| `--dropout` | float | 0.3 | Dropout probability |

### `run_hybrid.py`

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `--graph` | path | ✅ Yes | — | Path to a single graph file |
| `--type` | string | No | `undirected` | `undirected` or `directed` |
| `--pop` | int | No | 60 | MA population size |
| `--gens` | int | No | 300 | MA max generations |

---

## 11. Output and CSV Reference

### Console output columns

```
FVS size  — Number of vertices in the returned FVS
Time (ms) — Wall-clock time from function call to return (Python timer)
VALID     — ✓ if removing FVS vertices leaves a forest/DAG; ✗ otherwise
```

### CSV columns (from `--output`)

```
file        — filename only (not full path)
n           — number of vertices
m           — number of edges
BST_size    — FVS size returned by BST
BST_ms      — BST runtime in milliseconds
BST_valid   — True/False: FVS validity check
IC_size     — FVS size returned by IC
IC_ms       — IC runtime in milliseconds
IC_valid    — True/False
MA_size     — FVS size returned by MA
MA_ms       — MA runtime in milliseconds
MA_valid    — True/False
```

If you ran only one algorithm (e.g. `--algo IC`), only `IC_size`, `IC_ms`, `IC_valid` appear.

---

## 12. Quick Verification After Build

Copy-paste these 4 commands to confirm everything works end-to-end:

```bash
# Test 1: Undirected single file
python experiments/benchmark_undirected.py \
    --algo ALL \
    --test data/raw_undirected/triangle.txt

# Test 2: Directed single file
python experiments/benchmark_directed.py \
    --algo ALL \
    --test data/raw_directed/sample_pace.gr

# Test 3: Undirected batch + CSV
python experiments/benchmark_undirected.py \
    --algo ALL \
    --test data/raw_undirected/ \
    --output results/test_run.csv && cat results/test_run.csv

# Test 4: Directed batch
python experiments/benchmark_directed.py \
    --algo IC \
    --test data/raw_directed/ \
    --quiet
```

**All four should print `✓ VALID` for every algorithm and file.**

---

## Full Workflow: From Zero to Phase 3

```bash
# ── PHASE 1: Build + Basic Algorithms ────────────────────────────────────────
pip install pybind11 networkx pandas matplotlib
cd cpp_engine && mkdir -p build && cd build && cmake .. && make -j4 && cd ../..

# Verify
python experiments/benchmark_undirected.py --algo ALL --test data/raw_undirected/triangle.txt
python experiments/benchmark_directed.py   --algo ALL --test data/raw_directed/sample_pace.gr

# ── PHASE 2: GNN Training ─────────────────────────────────────────────────────
pip install torch torch-geometric
mkdir -p gnn_model/weights

# Generate data (use 2000 for decent training quality)
python gnn_model/dataset_gen.py --type both --n_graphs 2000 --max_n 80 --seed 42

# Train both models
python gnn_model/train.py --type both --epochs 200 --hidden 128

# ── PHASE 3: GNN-KMA GNN + MA ──────────────────────────────────────────────────
python experiments/run_hybrid.py --graph data/raw_undirected/sample_petersen.txt --type undirected
python experiments/run_hybrid.py --graph data/raw_directed/sample_pace.gr        --type directed
```

---

*Group 06 — FVS Research Project*
