# Feedback Vertex Set Project: Exact, Heuristic, and GNN-Hybrid Solvers

This repository is a full research and engineering workspace for the Feedback Vertex Set (FVS) problem, covering:

- Undirected FVS and Directed FVS (DFVS)
- Exact algorithms (BST, IC)
- Heuristic and hybrid algorithms (MA, KMA, DKMA, GNN-KMA variants)
- Synthetic and real-world benchmark data generation
- End-to-end benchmarking pipelines and result analysis
- GNN dataset generation, training, and hybrid inference integration
- C++ core engine with Python orchestration

---

## 1. What This Codebase Contains

At a high level, the project has four layers:

1. C++ solver engine (`cpp_engine/`)
2. Python benchmark and orchestration scripts (`experiments/`, top-level scripts)
3. Dataset generation pipeline (`data/` and `gnn_model/dataset_gen.py`)
4. GNN models and training pipeline (`gnn_model/`)

Important top-level files/folders:

- `build_engine.py` - one-command C++ build + install helper
- `requirements.txt` - Python dependencies for benchmarks + GNN pipeline
- `data/` - benchmark graph generation scripts and datasets
- `experiments/` - benchmark runners and experiment suites
- `gnn_model/` - feature engineering, model definitions, training, dataset-to-PT conversion
- `results/`, `directed_results/`, `paceresults/` - benchmark outputs and evaluation artifacts
- `tests/` - unit tests (currently includes GNN component tests)

---

## 2. Algorithm Families Implemented

### Exact

- `BST` - Bounded Search Tree
- `IC` - Iterative Compression

### Heuristic / Hybrid

- `MA` - Memetic Algorithm
- `KMA` - Kernelized Memetic Algorithm
- `DKMA` - Dynamic Kernelized Memetic Algorithm
- `GNN-KMA` (v1, v2, v3)
- `GNN-DKMA`

Core solver implementations live primarily in:

- `cpp_engine/src/undirected_algos/`
- `cpp_engine/src/directed_algos/`
- `experiments/run_hybrid.py` (hybrid and DKMA orchestration)

---

## 3. Two-Track Benchmark Design

The project uses a two-track benchmark layout:

- `exact_track` - smaller graphs, includes exact algorithms
- `heuristic_track` - larger graphs, focuses on scalable heuristic/hybrid methods

Canonical layout:

`data/synthetic/<family>/<track>/<category>/*.txt`

Families:

- `undirected`
- `directed`

---

## 4. Environment Setup

### 4.1 Python dependencies

```bash
python -m pip install -r requirements.txt
```

### 4.2 Build C++ engine

Recommended:

```bash
python build_engine.py
```

Optional flags:

- `--clean`
- `--build-type Debug|Release|RelWithDebInfo|MinSizeRel`
- `--jobs N`
- `--no-install-deps`

Build outputs are generated under `cpp_engine/build-*` and installed module artifacts are made available to benchmark scripts.

---

## 5. Data Generation Workflow

### 5.1 One-command benchmark input generation

```bash
python data/setup_benchmark_inputs.py --total-undirected 100000 --total-directed 100000
```

This orchestrates:

1. `data/download_real_world.py`
2. `data/generate_synthetic.py`

Quick smoke run:

```bash
python data/setup_benchmark_inputs.py --total-undirected 100 --total-directed 100
```

### 5.2 Real-world and synthetic generators

- `download_real_world.py` populates real-world category slices from available datasets (with graceful fallback).
- `generate_synthetic.py` fills category buckets using graph-model generators.

### 5.3 Detailed documentation

See:

- `DATASET_GENERATION_PROCESS.md`

---

## 6. Running Benchmarks

### 6.1 Full suite runner

```bash
python experiments/run_benchmark_suite.py
```

Profiles:

- `--profile requested` (default)
- `--profile full`

Useful options:

- `--mode all|undirected|directed`
- `--pop`, `--gens`, `--timeout`, `--earlystop`
- `--rerun`
- `--max-files` for quick checks

### 6.2 Single-family benchmark scripts

Undirected:

```bash
python experiments/benchmark_undirected.py --algo KMA --test data/synthetic/undirected
```

Directed:

```bash
python experiments/benchmark_directed.py --algo KMA --test data/synthetic/directed
```

### 6.3 Unified pipeline

`experiments/run_pipeline.py` supports existing-data validation, subset preparation, and benchmark execution by track/family.

---

## 7. GNN Dataset Generation and Training

### 7.1 Build PT datasets

```bash
python gnn_model/dataset_gen.py --family all --track both --variant v1
python gnn_model/dataset_gen.py --family all --track both --variant v2
python gnn_model/dataset_gen.py --family all --track both --variant v3
```

Controls include:

- `--solver-timeout`
- `--kma-pop`, `--kma-gens`, `--kma-early-stop`
- `--force`, `--clean-root`

### 7.2 Train models

```bash
python gnn_model/train.py --type both --variant v1 --epochs 100
python gnn_model/train.py --type both --variant v2 --epochs 100
python gnn_model/train.py --type both --variant v3 --epochs 200
```

Key training controls:

- `--hidden`, `--dropout`, `--lr`, `--val-ratio`
- `--seed`
- v3-specific: `--warmup-epochs`, `--max-grad-norm`

### 7.3 Weight outputs

Saved in `gnn_model/weights/` as variant-specific files, for example:

- `undirected_fvs_gcn.pt`, `directed_fvs_gcn.pt`
- `undirected_fvs_gcn_v2.pt`, `directed_fvs_gcn_v2.pt`
- `undirected_fvs_gcn_v3.pt`, `directed_fvs_gcn_v3.pt`

---

## 8. Hybrid Solvers (GNN-KMA and DKMA)

`experiments/run_hybrid.py` is the central hybrid runtime module.

Implemented capabilities include:

- pure KMA wrappers (directed/undirected)
- DKMA (dynamic kernelized) directed/undirected
- GNN-KMA v1/v2/v3
- GNN-DKMA

Current coupling design is precision-first soft-hint integration:

- high-confidence GNN candidates may be hard-fixed
- hard-fix fraction is capped
- fallback to pure KMA when confidence is insufficient

This design avoids false-positive inflation and keeps hybrid outputs stable.

---

## 9. Result Files and Evaluation

Outputs are produced in CSV form across folders such as:

- `results/`
- `results/suite/`
- `directed_results/`
- `paceresults/`

Common fields include:

- input identity (`file`, `file_path`)
- graph size (`n`, `m`)
- algorithm and track/family tags
- `fvs_size`, `runtime_ms`, validity/status flags

Evaluation and analysis scripts are available in result folders and `IC-test/`.

---

## 10. Documentation Index

Detailed algorithm/process documents in this repository:

- `DATASET_GENERATION_PROCESS.md`
- `IC_BST_MA_IMPLEMENTATION.md`
- `KMA_IMPLEMENTATION_PROCEDURE.md`
- `DKMA_IMPLEMENTATION_PROCEDURE.md`
- `GNN_KMA_MODELS_V1_V2_V3_DETAILED.md`
- `KMA_ALGORITHM_EXPLANATION.md`
- `DKMA_ALGORITHM_EXPLANATION.md`
- `GNN_KMA_MODELS.md`
- `GNN_KMA_PROCESS_DETAILED.md`
- `operations_guide.md`

---

## 11. Testing

Current test module:

- `tests/test_gnn_components.py`

Run tests with:

```bash
pytest -q
```

---

## 12. Suggested Workflows

### Fast local smoke workflow

1. Build engine: `python build_engine.py`
2. Generate tiny data: `python data/setup_benchmark_inputs.py --total-undirected 100 --total-directed 100`
3. Run tiny suite: `python experiments/run_benchmark_suite.py --max-files 1 --quiet`

### Full benchmark workflow

1. Generate full dataset corpora
2. Run suite (`run_benchmark_suite.py`)
3. Analyze CSV outputs and plots

### Full hybrid/GNN workflow

1. Generate PT datasets (`dataset_gen.py`)
2. Train v1/v2/v3 models (`train.py`)
3. Benchmark GNN-KMA variants via benchmark scripts or `run_pipeline.py`
4. Compare against MA/KMA/DKMA baselines

---

## 13. Notes

- The project is designed for deterministic and restartable experimentation.
- Most long-running commands expose timeout and early-stop controls.
- Hybrid pipelines include robust fallback paths when optional components (weights, packages, specific bindings) are unavailable.

For implementation-level details, use the dedicated markdown guides listed in section 10.
