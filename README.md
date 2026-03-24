# Feedback Vertex Set Problem — CSE 462 Research Project

A comprehensive research implementation comparing multiple exact and heuristic algorithms for solving the **Feedback Vertex Set (FVS)** problem, a fundamental NP-hard optimization problem in graph theory.

## Overview

This project implements and experimentally evaluates three FVS algorithms:
- **Iterative Compression (IC)** — Exact FPT algorithm, O(5^k × k × n²) worst-case
- **Kernelization + Bounded Search Tree (KBST)** — Exact algorithm with preprocessing, O(4^k × n²)
- **Memetic Genetic Algorithm (MEMETIC)** — Heuristic for large instances (n > 1000)

Across **10 comprehensive experiments** generating **12 publication-quality figures**, this project validates correctness, solution quality, runtime scalability, and robustness on synthetic and real-world networks.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [Algorithms](#algorithms)
- [Experiments](#experiments)
- [Output & Results](#output--results)
- [Development Notes](#development-notes)

---

## Quick Start

```bash
# Clone or navigate to project directory
cd /path/to/Feedback_Vertex_Set_Problem_Algorithm_Project_4-2

# Install dependencies
pip install -r requirements.txt

# Run full pipeline (generates data, runs all 10 experiments, creates 12 figures)
python main.py

# Run with options:
python main.py --quick              # QUICK_MODE: only instances with n ≤ 200
python main.py --tiny               # TINY_MODE: only 30 smallest instances
python main.py --exp EXP3           # Run only experiment 3 (runtime scalability)
python main.py --download-only      # Generate/download datasets (no experiments)
python main.py --plots-only         # Generate plots from existing results
```

---

## Installation

### Requirements
- **Python 3.10+**
- **Dependencies** (see `requirements.txt`):
  - `networkx>=3.0` — Graph manipulation
  - `numpy>=1.24` — Numerical computing
  - `pandas>=2.0` — Data analysis
  - `scipy>=1.10` — Scientific computing
  - `matplotlib>=3.7`, `seaborn>=0.12` — Visualization
  - `requests>=2.28` — Download real-world graphs
  - `tqdm>=4.65` — Progress bars
  - `psutil>=5.9` — System monitoring
  - `tabulate>=0.9` — Formatted output

### Setup

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import networkx, numpy, pandas; print('✓ All dependencies installed')"
```

---

## Project Structure

```
Feedback_Vertex_Set_Problem_Algorithm_Project_4-2/
│
├── main.py                 # Single entry point — orchestrates all experiments
├── plot.py                 # Generate 12 figures from results/report.csv
├── requirements.txt        # Python dependencies
├── .gitignore              # Version control exclusions (data/ excluded)
├── README.md               # This file
│
├── algorithms/             # FVS Algorithm implementations
│   ├── __init__.py
│   ├── base.py            # Abstract FVSSolver base class
│   ├── brute_force.py     # Exhaustive enumeration (n ≤ 20)
│   ├── iterative_compression.py  # Algorithm 1: IC (exact, FPT)
│   ├── kernelization_bst.py      # Algorithm 2: KBST (exact)
│   └── memetic_ga.py             # Algorithm 3: Memetic GA (heuristic)
│
├── experiments/            # 10 Research Experiments
│   ├── __init__.py
│   ├── runner.py           # Master orchestrator with checkpointing
│   ├── exp1_correctness.py       # Validate all algorithms produce valid FVS
│   ├── exp2_solution_quality.py  # Compare FVS sizes (n ≤ 200)
│   ├── exp3_runtime_scalability.py    # Wall-clock time vs. instance size
│   ├── exp4_pareto.py            # Quality-runtime trade-off analysis
│   ├── exp5_structure_sensitivity.py  # Performance by graph type
│   ├── exp6_ga_parameters.py     # Hyperparameter grid search (240 runs)
│   ├── exp7_convergence.py       # GA convergence profiles
│   ├── exp8_optimality_gap.py    # Gap between heuristic and optimal
│   ├── exp9_realworld.py         # Real-world network performance
│   └── exp10_robustness.py       # Noise/perturbation robustness
│
├── analysis/               # Results Analysis & Reporting
│   ├── __init__.py
│   ├── report_writer.py    # Thread-safe CSV writer for results/report.csv
│   └── statistics.py       # Statistical tests (Friedman, post-hoc)
│
├── data/                   # Data Handling (generated/downloaded at runtime)
│   ├── __init__.py
│   ├── downloader.py       # Fetch real-world networks (social, infrastructure)
│   ├── generator.py        # Generate synthetic graphs (ER, BA, Grid, etc.)
│   ├── validator.py        # FVS validation & cycle detection
│   ├── synthetic/          # Auto-generated: 40+ synthetic instances (GraphML)
│   └── real_world/         # Auto-downloaded: 5+ real-world networks (GraphML)
│
├── results/                # Experiment outputs (auto-created)
│   ├── report.csv          # Master results table (18 columns)
│   ├── exp{N}_*.json       # Experiment-specific intermediate data
│   └── run.log             # Detailed execution log
│
├── figures/                # Publication-ready visualizations (auto-created)
│   ├── fig1_correctness_validation.png
│   ├── fig2_solution_quality_comparison.png
│   ├── fig3_runtime_scalability.png
│   ├── fig4_pareto_frontier.png
│   ├── fig5_structure_sensitivity.png
│   ├── fig6_ga_hyperparameter_heatmap.png
│   ├── fig7_convergence_profiles.png
│   ├── fig8_optimality_gap_analysis.png
│   ├── fig9_realworld_performance.png
│   ├── fig10_robustness_perturbation.png
│   ├── fig11_instance_size_distribution.png
│   └── fig12_algorithm_comparison_summary.png
│
└── checkpoint_1_presentation/  # Presentation materials (LaTeX/PDF)
```

---

## Usage Guide

### 1. Running the Full Pipeline

```bash
python main.py
```

This will:
1. **Generate 40+ synthetic graphs** (ER, BA, Grid, Watts-Strogatz, Cycle-Heavy)
2. **Download 5+ real-world networks** (social/infrastructure networks)
3. **Execute all 10 experiments** (1000+ algorithm runs)
4. **Checkpoint progress** in `results/report.csv` (resume-safe)
5. **Generate experiment JSON summaries** in `results/`
6. **Generate 12 figures** in `figures/` directory

**Execution time:** ~2–4 hours (depending on hardware and QUICK_MODE)

### 2. Quick Mode (Reduced Dataset)

```bash
python main.py --quick
```

Runs only instances with **n ≤ 200 vertices**, reducing runtime to ~30 minutes.
Ideal for development & debugging.

### 3. Tiny Mode (Minimal Dataset)

```bash
python main.py --tiny
```

Runs only the **30 smallest instances** (by vertex count n).
**Fastest execution path** — completes in ~3–5 minutes.
Perfect for quick testing and validation during development.

**Customizing TINY_MODE:**
To change the number of instances, edit `main.py` and modify:
```python
TINY_MODE_COUNT = 30  # Change this value to run fewer/more instances
```

**Note:** `--tiny` automatically enables `--quick` mode under the hood.

### 4. Run Individual Experiments

```bash
python main.py --exp EXP1            # Correctness validation
python main.py --exp EXP3            # Runtime scalability
python main.py --exp EXP6            # GA hyperparameter study
python main.py --exp EXP9            # Real-world networks
```

### 5. Generate Plots Only

```bash
python main.py --plots-only
```

Reads existing `results/report.csv` and generates 12 figures without re-running algorithms.

### 6. Generate/Download Data Only

```bash
python main.py --download-only
```

Creates synthetic and real-world instances without running experiments.

### 7. Resume Interrupted Run

All experiments checkpoint to `results/report.csv`. If execution is interrupted, simply re-run:

```bash
python main.py  # Automatically skips completed (experiment, instance, algorithm, run) keys
```

---

## Algorithms

### Algorithm 1: Iterative Compression (IC)

**Type:** Exact, Fixed-Parameter Tractable (FPT)  
**Time Complexity:** O(5^k × k × n²)  
**Space Complexity:** O(n²)  
**Best For:** Small–medium instances (n ≤ 200)

**Key Idea:**
- Process vertices one at a time, maintaining a "compression set" F
- When |F| exceeds k, invoke compression subroutine to reduce F
- Guaranteed to find **minimum FVS**

**Location:** [algorithms/iterative_compression.py](algorithms/iterative_compression.py)

### Algorithm 2: Kernelization + Bounded Search Tree (KBST)

**Type:** Exact, combines preprocessing + branching  
**Time Complexity:** O(4^k × n²) after kernelization  
**Space Complexity:** O(n²)  
**Best For:** Medium instances (n ≤ 500)

**Key Idea:**
- Kernelization rules reduce graph to ≤ k² + k vertices before branching
- Bounded search tree explores remaining solution space efficiently
- Guaranteed to find **minimum FVS**

**Location:** [algorithms/kernelization_bst.py](algorithms/kernelization_bst.py)

### Algorithm 3: Memetic Genetic Algorithm (MEMETIC)

**Type:** Heuristic (approximate)  
**Time Complexity:** O(G × P × n²) per generation  
**Space Complexity:** O(P × n)  
**Best For:** Large instances (n > 1000)

**Key Idea:**
- Combines population-based search (GA) with local hill-climbing (memetic)
- Encoding: Permutation of vertices
- Decoding: Greedy left-to-right scan (include vertex in FVS only if cycles remain)
- Configurable hyperparameters:
  - `population_size` (default: 100)
  - `max_generations` (default: 200)
  - `mutation_rate` (default: 0.05)
  - `crossover_rate` (default: 0.8)
  - `local_search_iterations` (default: 50)

**Location:** [algorithms/memetic_ga.py](algorithms/memetic_ga.py)

---

## Experiments

### EXP1: Correctness Validation
**Purpose:** Verify all algorithms produce valid FVS  
**Scope:** All instances with n ≤ 50  
**Metrics:** Valid FVS (binary), execution time  
**Output:** Validation pass/fail for each algorithm

### EXP2: Solution Quality Comparison
**Purpose:** Compare FVS sizes across algorithms  
**Scope:** Instances with n ≤ 200  
**Metrics:** FVS size, approximation ratio, optimality gap  
**Statistical Test:** Friedman rank-sum test + post-hoc analysis

### EXP3: Runtime Scalability
**Purpose:** Measure wall-clock time vs. instance size  
**Scope:**
  - IC: n ≤ 200
  - KBST: n ≤ 500
  - MEMETIC: all n
**Metrics:** CPU time, wall-clock time vs. n (log-log plots)

### EXP4: Pareto Analysis
**Purpose:** Quality-runtime trade-off (which algorithm best for given time budget?)  
**Scope:** n ∈ {20, 50, 100, 200}  
**Metrics:** FVS size vs. runtime under fixed time budgets (1s, 10s, 60s)

### EXP5: Graph Structure Sensitivity
**Purpose:** Does performance vary by graph type?  
**Scope:** Compare ER, BA, Grid, Watts-Strogatz, Cycle-Heavy graphs  
**Metrics:** Algorithm performance ranking by graph family  
**Insight:** Identify which algorithms excel on specific topologies

### EXP6: GA Hyperparameter Sensitivity
**Purpose:** Grid search for optimal MEMETIC configuration  
**Scope:** 4×4×3×5 = 240 configuration runs on 5 representative instances  
**Parameters:** Population size, generations, mutation rate, local search intensity  
**Output:** Heatmap showing parameter impact on solution quality

### EXP7: Convergence Profiles
**Purpose:** Track GA fitness improvement over generations  
**Scope:** 10 medium instances (n ≤ 100)  
**Metrics:** Best fitness per generation (convergence curves)

### EXP8: Optimality Gap Analysis
**Purpose:** Quantify heuristic solution quality vs. guarantees  
**Scope:** Compare MEMETIC to IC/KBST optimal solutions (n ≤ 200)  
**Metrics:** % optimality gap, distribution analysis

### EXP9: Real-World Network Validation
**Purpose:** Evaluate algorithms on realistic (social/infrastructure) networks  
**Scope:** 5+ real-world instances (downloaded or synthetic proxy)  
**Metrics:** FVS size, runtime, solution quality vs. network properties

### EXP10: Robustness & Perturbation Analysis
**Purpose:** How do algorithms perform under noise/edge perturbations?  
**Scope:** Gradually add/remove edges; measure FVS stability  
**Metrics:** Solution size variance, algorithm sensitivity

---

## Output & Results

### 1. results/report.csv
**Master results table** with **18 columns**:
| Column | Description |
|--------|-------------|
| `experiment_id` | EXP1–EXP10 |
| `instance_id` | Graph name (e.g., "ER_n100_p0.3_seed1") |
| `graph_type` | ER, BA, Grid, WS, CycleHeavy, RealWorld |
| `n_vertices` | Number of vertices |
| `n_edges` | Number of edges |
| `graph_density` | Edge density (m / (n²/2)) |
| `algorithm` | IC, KBST, MEMETIC, BRUTE_FORCE |
| `run_number` | Run index (for repeated experiments) |
| `fvs_size` | Size of found FVS |
| `optimal_fvs_size` | Known optimal (if available) |
| `approximation_ratio` | fvs_size / optimal |
| `optimality_gap_pct` | 100 × (fvs_size - optimal) / optimal |
| `wall_time_sec` | Actual runtime |
| `cpu_time_sec` | CPU utilization time |
| `peak_memory_mb` | Peak RAM used (MB) |
| `is_valid_solution` | Valid FVS (true/false) |
| `notes` | Algorithm-specific notes |
| `timestamp` | ISO 8601 execution timestamp |

**Example rows:**
```
EXP1,ER_n50_p0.3_seed1,ER,50,375,0.31,IC,1,12,12,1.00,0.00,0.523,0.502,45.2,true,"Optimal found",2024-03-15T10:23:45Z
EXP3,BA_n500_m3_seed1,BA,500,1497,0.01,MEMETIC,1,87,unknown,unknown,unknown,3.456,3.201,256.8,true,"Converged",2024-03-15T10:25:12Z
```

### 2. results/exp{N}_*.json
**Experiment-specific intermediate data:**
- `exp3_scaling_data.json` — Runtime vs. n for each algorithm
- `exp4_pareto.json` — Quality-time Pareto fronts
- `exp6_ga_param_study.json` — Hyperparameter grid results
- `exp7_convergence.json` — Generation-by-generation GA fitness
- `exp8_gap_analysis.json` — Optimality gap statistics

### 3. results/run.log
**Detailed execution log** with timestamps:
```
[2024-03-15 10:00:00,123] [INFO] Starting FVS Research Pipeline...
[2024-03-15 10:00:05,456] [INFO] Generating synthetic graphs...
[2024-03-15 10:01:30,789] [INFO] Generated 40 instances in data/synthetic/
[2024-03-15 10:01:35,012] [INFO] Downloading real-world networks...
[2024-03-15 10:02:45,234] [INFO] EXP1 (Correctness): 150 runs, 3 algorithms
...
```

### 4. figures/ Directory
**12 publication-ready PNG + PDF figures:**

| Figure | Description |
|--------|------------|
| **fig1** | Correctness validation across all algorithms |
| **fig2** | FVS size comparison (IC vs. KBST vs. MEMETIC) |
| **fig3** | Runtime scaling: log-log plots, all algorithms |
| **fig4** | Pareto frontier: quality vs. runtime |
| **fig5** | Algorithm performance by graph type (heatmap) |
| **fig6** | GA hyperparameter sensitivity (heatmap) |
| **fig7** | GA convergence profiles (fitness over generations) |
| **fig8** | Optimality gap distribution (violin plots) |
| **fig9** | Real-world network performance breakdown |
| **fig10** | Robustness under perturbation (line plots) |
| **fig11** | Instance size distribution (histogram) |
| **fig12** | Algorithm comparison summary (table + bar chart) |

**Format:** 300 DPI PNG + vector PDF for publications

---

## Data Handling

### Synthetic Graphs (Auto-Generated)
Generated on first run in `data/synthetic/`:

| Type | Count | Parameters |
|------|-------|-----------|
| Erdős-Rényi (ER) | 35 | n ∈ {10, 20, 50, 100, 200, 500, 1000}, p ∈ {0.1, ..., 0.9} |
| Barabási-Albert (BA) | 21 | n ∈ {10, 20, 50, 100, 200, 500, 1000}, m ∈ {2, 3, 5} |
| Grid Graph | 5 | Sizes: (3,3), (5,5), (10,10), (20,20), (30,30) |
| Watts-Strogatz (WS) | 45 | n ∈ {20, 50, 100, 200, 500}, k ∈ {4, 6, 8}, b ∈ {0.1, 0.3, 0.5} |
| Cycle-Heavy (CH) | 12 | 4 density levels × 3 patterns |

**Format:** GraphML (XML), human-readable  
**Storage:** ~50 MB total  
**Quick Mode:** n ≤ 200 only (~8 MB)

### Real-World Graphs (Downloaded)
Downloaded on first run to `data/real_world/`:
- Karate Club (n=34, social)
- Dolphin Social Network (n=62, social)
- US Political Blogs (n=1490, social)
- Power Grid (n≈5000, infrastructure)
- Other small networks from NetworkX/Graph500

**Format:** GraphML  
**Storage:** ~30 MB total  
**Fallback:** If download fails, synthetic proxy graphs with similar properties are used

### Git Ignore Policy
**All data automatically excluded from version control:**

```gitignore
# Data files (generated/downloaded at runtime)
/data/synthetic/
/data/real_world/

# Results & logs (experiment outputs)
/results/
/figures/

# Python artifacts
__pycache__/
*.pyc
*.pyo
*.egg-info/

# Virtual environments
/venv/
```

**Why?**
- Graphs are **reproducibly generated** → no need to commit
- Experiments are **deterministic** (fixed random seeds)
- Users can regenerate data by running `python main.py --download-only`
- Repository stays **lightweight** (no large binary files)

---

## Development Notes

### Key Design Patterns

**1. Abstract Base Class (`FVSSolver`)**
```python
class FVSSolver(ABC):
    @abstractmethod
    def solve(self, graph: nx.Graph, k=None) -> tuple[set, dict]:
        """Returns (fvs_set, info_dict)"""
```
All algorithms inherit from this, ensuring consistent interface.

**2. Checkpointing with report.csv**
- `runner.load_done_set()` loads completed (experiment, instance, algorithm, run) keys
- Experiments skip already-finished runs
- Safe to interrupt and resume

**3. Thread-Safe Reporting**
- `ReportWriter` uses file locks for thread-safe CSV writes
- Partial results preserved if process crashes

**4. Random Seed Control**
- All generators use fixed seeds (reproducible)
- GA algorithms accept `random_seed` parameter
- Enables deterministic repeated runs for statistical tests

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `FVS_QUICK_MODE` | Limit to n ≤ 200 | "1" (enabled) |
| `FVS_PROJECT_DIR` | Project root (auto-detected) | `Path(__file__).parent` |

### Logging Levels

```python
# In any module:
import logging
logger = logging.getLogger(__name__)
logger.info("Message")      # General info
logger.warning("Alert")     # Potential issues
logger.error("Problem")     # Failure (continues)
logger.critical("Fatal")    # Severe error (may exit)
```

All logs written to both `stdout` and `results/run.log`.

### Adding a New Experiment

1. Create `experiments/exp{N}_{name}.py`
2. Define `run(config: dict, report_writer: ReportWriter, done_set: set) -> None`
3. Register in `main.py`'s experiment loader
4. Checkpoint to `report_writer` using `write_row()`

Example:
```python
def run(config, report_writer, done_set):
    for instance_id, graph in config["all_instances"]:
        result, info = my_algorithm.solve(graph)
        report_writer.write_row(
            experiment_id="EXP11",
            instance_id=instance_id,
            algorithm="MyAlgo",
            fvs_size=len(result),
            wall_time_sec=info['time_sec'],
            # ... other fields
        )
```

### Profiling & Debugging

**CPU Profile:**
```bash
python -m cProfile -s cumtime main.py --exp EXP1 > profile.txt
```

**Memory Profile:**
```bash
pip install memory-profiler
python -m memory_profiler plot.py
```

**Verbose Logging:**
```python
# In main.py, change log level:
logging.basicConfig(level=logging.DEBUG)  # Instead of INFO
```

---

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|------------|
| **CPU** | 2 cores | 8+ cores |
| **RAM** | 4 GB | 16+ GB |
| **Disk** | 1 GB free | 5+ GB free |
| **Python** | 3.10 | 3.11+ |
| **OS** | Linux/macOS/Windows | Linux (tested on Ubuntu 22.04) |

**Runtime Estimates:**
- **Full pipeline:** 2–4 hours
- **QUICK_MODE:** 30–45 minutes
- **Single EXP:** 5–30 minutes (depends on EXP)

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'algorithms'"
**Solution:**
```bash
# Ensure you're running from project root:
cd /path/to/Feedback_Vertex_Set_Problem_Algorithm_Project_4-2
python main.py
```

### Issue: "Experiment already done" (stuck on checkpoint)
**Solution:**
```bash
# Delete checkpoint to re-run:
rm results/report.csv

# Then re-run:
python main.py
```

### Issue: Timeout on large instances
**Solution:**
```bash
# Use QUICK_MODE:
python main.py --quick

# Or skip specific algorithm:
# (Manually edit experiments to skip KBST if n > 500)
```

### Issue: Real-world graphs fail to download
**Solution:**
```bash
# Automatically falls back to synthetic proxies.
# Check results/run.log for details:
grep "Download" results/run.log
```

---

## Citation & References

If using this project for research, cite as:

```bibtex
@misc{fvs_cse462_2024,
  title={Feedback Vertex Set: Algorithm Implementation \& Comparative Analysis},
  author={[Your Name]},
  year={2024},
  institution={CSE 462 Research Project},
  note={Available at: /path/to/repo}
}
```

**Key References:**
- Even et al., "A fast algorithm for solving unbounded knapsack problems" (IC algorithm)
- Bodlaender & Thomassé, "Graph Minors and Parameterized Complexity" (Kernelization)
- Moscato & Norman, "A memetic approach to combinatorial optimization" (GA framework)

---

## Contact & Support

For issues, clarifications, or feature requests:
1. Check `results/run.log` for detailed error messages
2. Review this README's troubleshooting section
3. Inspect `results/report.csv` for incomplete runs

---

## License & Acknowledgments

**Project:** CSE 462 Research Project  
**Semester:** Spring 2024  
**Institution:** [Your Institution]

**Acknowledgments:**
- NetworkX developers for graph manipulation library
- Matplotlib/Seaborn communities for visualization
- Research advisors and peer reviewers

---

**Last Updated:** March 2024  
**Version:** 1.0  
**Status:** Complete & Production-Ready ✓
