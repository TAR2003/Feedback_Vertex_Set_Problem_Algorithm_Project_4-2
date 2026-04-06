# Feedback_Vertex_Set_Problem_Algorithm_Project_4-2

Comprehensive framework for solving and studying Feedback Vertex Set (FVS) and Directed FVS (DFVS), with exact algorithms, scalable heuristics, and GNN-guided hybrid solvers.

## 1. Project Scope

This repository is both a research platform and an operations-ready benchmarking toolkit.

It includes:

- exact solvers (BST, IC)
- evolutionary solvers (MA, KMA, DKMA)
- hybrid ML + combinatorial solvers (GNN-KMA v1/v2/v3, GNN-DKMA)
- synthetic and real-world data generation
- repeatable benchmark pipelines with resume semantics
- PT dataset generation, model training, and weight deployment

## 2. Architecture Overview

The codebase has four layers:

1. C++ algorithm engine
2. Python benchmarking/orchestration layer
3. data generation and preprocessing layer
4. GNN dataset/training/inference layer

### 2.1 C++ engine

Located in cpp_engine, with separate directed and undirected solver modules.

### 2.2 Python orchestration

Located in experiments and top-level scripts. It provides:

- parser adapters
- timeout and batch execution wrappers
- CSV persistence and resume behavior
- suite/pipeline runners

### 2.3 Data pipeline

Located in data and gnn_model/dataset_gen.py.

### 2.4 GNN stack

Located in gnn_model, including:

- feature engineering
- model definitions
- train loop
- runtime inference integration

## 3. Repository Highlights

Key files and why they matter:

- build_engine.py: reproducible build+install entrypoint for C++ module
- requirements.txt: pinned/runtime dependencies for both solver and GNN paths
- data/setup_benchmark_inputs.py: one-command benchmark input orchestrator
- experiments/run_benchmark_suite.py: large-scale batch benchmark runner
- experiments/run_pipeline.py: unified existing-data pipeline
- experiments/run_hybrid.py: hybrid runtime core (KMA, DKMA, GNN coupling)
- gnn_model/dataset_gen.py: TXT-to-PT generation with solver supervision
- gnn_model/train.py: variant-aware training and checkpointing

## 4. Algorithms Implemented

### 4.1 Exact

- BST (Bounded Search Tree)
- IC (Iterative Compression)

### 4.2 Heuristic and hybrid

- MA (Memetic Algorithm)
- KMA (Kernelized Memetic Algorithm)
- DKMA (Dynamic Kernelized Memetic Algorithm)
- GNN-KMA v1
- GNN-KMA v2
- GNN-KMA v3
- GNN-DKMA

## 5. Data Model and Benchmark Tracks

Canonical corpus layout:

- data/synthetic/{family}/{track}/{category}/*.txt

Families:

- undirected
- directed

Tracks:

- exact_track
- heuristic_track

Track intent:

- exact_track: small/controlled workloads where exact methods are feasible
- heuristic_track: larger workloads for scalable heuristics/hybrids

## 6. Environment Setup

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Build C++ engine:

```bash
python build_engine.py
```

Common build options:

- clean
- build-type
- jobs
- no-install-deps

## 7. Data Generation Workflows

### 7.1 Full benchmark input generation

```bash
python data/setup_benchmark_inputs.py --total-undirected 100000 --total-directed 100000 --exact-ratio 0.5 --seed 1337
```

### 7.2 Smoke generation

```bash
python data/setup_benchmark_inputs.py --total-undirected 100 --total-directed 100
```

### 7.3 Family-limited generation

```bash
python data/setup_benchmark_inputs.py --family directed --total-directed 5000 --total-undirected 0
```

Detailed process guide:

- DATASET_GENERATION_PROCESS.md

## 8. Benchmark Execution

### 8.1 Full suite

```bash
python experiments/run_benchmark_suite.py
```

Useful switches:

- profile requested or full
- mode all/undirected/directed
- pop, gens, timeout, earlystop
- rerun
- max-files for smoke batches

### 8.2 Single benchmark scripts

Undirected:

```bash
python experiments/benchmark_undirected.py --algo KMA --test data/synthetic/undirected
```

Directed:

```bash
python experiments/benchmark_directed.py --algo KMA --test data/synthetic/directed
```

### 8.3 Unified pipeline mode

run_pipeline.py supports validation, synthetic subset preparation, and benchmark dispatch in one command path.

## 9. GNN Dataset and Training Pipeline

### 9.1 PT generation by variant

```bash
python gnn_model/dataset_gen.py --family all --track both --variant v1
python gnn_model/dataset_gen.py --family all --track both --variant v2
python gnn_model/dataset_gen.py --family all --track both --variant v3
```

Important controls:

- solver-timeout
- kma-pop, kma-gens, kma-early-stop
- force, clean-root

### 9.2 Training

```bash
python gnn_model/train.py --type both --variant v1 --epochs 100
python gnn_model/train.py --type both --variant v2 --epochs 100
python gnn_model/train.py --type both --variant v3 --epochs 200
```

v3-specific controls:

- warmup-epochs
- max-grad-norm

### 9.3 Weights

Weights are saved under gnn_model/weights with variant-specific filenames (base, v2, v3).

## 10. Hybrid Runtime

experiments/run_hybrid.py provides:

- KMA wrappers
- DKMA wrappers
- GNN-KMA v1/v2/v3 wrappers
- GNN-DKMA wrappers

Current coupling policy is precision-first soft-hint:

- only high-confidence candidates are hard-fixed
- hard-fix ratio is capped
- pure KMA fallback path is used when confidence is insufficient

## 11. Outputs and Evaluation Artifacts

Benchmark CSV outputs are written to folders such as:

- results
- results/suite
- directed_results
- paceresults

Typical fields include:

- file identity
- graph size
- algorithm/family/track tags
- fvs_size, runtime, valid, status

Analysis scripts for comparison and plotting are included in result and IC-test folders.

## 12. Testing

Current test focus includes GNN component tests:

- tests/test_gnn_components.py

Run:

```bash
pytest -q
```

## 13. Documentation Index

Primary deep-dive docs:

- DATASET_GENERATION_PROCESS.md
- IC_BST_MA_IMPLEMENTATION.md
- KMA_IMPLEMENTATION_PROCEDURE.md
- DKMA_IMPLEMENTATION_PROCEDURE.md
- GNN_KMA_MODELS_V1_V2_V3_DETAILED.md

Additional repository references:

- KMA_ALGORITHM_EXPLANATION.md
- DKMA_ALGORITHM_EXPLANATION.md
- GNN_KMA_MODELS.md
- GNN_KMA_PROCESS_DETAILED.md
- operations_guide.md

## 14. Recommended Usage Patterns

### 14.1 Fast local smoke

1. build engine
2. generate small corpus
3. run suite with max-files 1

### 14.2 Full benchmark campaign

1. generate full corpora
2. run run_benchmark_suite.py with resume semantics
3. aggregate and analyze CSV outputs

### 14.3 Full hybrid campaign

1. generate PT datasets
2. train v1/v2/v3
3. run benchmark scripts for GNN variants
4. compare against MA/KMA/DKMA baselines

## 15. Operational Notes

- Pipelines are designed for deterministic and restartable execution.
- Timeouts and early-stop settings are available on long-running solver paths.
- Most hybrid paths include fallback logic to keep batch runs robust under partial dependency availability.

For implementation-level understanding, use the deep-dive documents in Section 13.
