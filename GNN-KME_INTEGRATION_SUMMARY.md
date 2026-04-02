# GNN-KMA Algorithm Integration Summary

## Overview
Successfully integrated the GNN-KMA (GNN-guided Memetic Algorithm) into both `benchmark_directed.py` and `benchmark_undirected.py` scripts. The GNN-KMA algorithm can now be run directly from the benchmark CLI just like BST, IC, and MA algorithms.

## Changes Made

### 1. **benchmark_directed.py** ✓
- **Added import**: `gnn_kme_solve_directed` from `experiments.run_hybrid` (with graceful fallback)
- **Updated ALGO_MAP_D**: Now includes GNN-KMA as a selectable algorithm option
- **Modified run_directed_algorithm()**: Added conditional handling for GNN-KMA algorithm
- **Updated argparse**: Added "GNN-KMA" to choices list
- **Updated docstring**: Added examples showing how to run GNN-KMA
- **Updated summary printing**: Now prints GNN-KMA results alongside BST, IC, MA

**New supported commands:**
```bash
# Run GNN-KMA on a single file
python experiments/benchmark_directed.py --algo GNN-KMA --test data/raw_directed/pace001.gr

# Run GNN-KMA with custom parameters
python experiments/benchmark_directed.py --algo GNN-KMA --test data/raw_directed/pace001.gr --pop 100 --gens 400

# Run GNN-KMA on all files in a folder
python experiments/benchmark_directed.py --algo GNN-KMA --test data/raw_directed/ --output results_GNN-KMA.csv

# Compare ALL algorithms (including GNN-KMA)
python experiments/benchmark_directed.py --algo ALL --test data/raw_directed/ --output results_comparison.csv
```

### 2. **benchmark_undirected.py** ✓
- **Added import**: `gnn_kme_solve_undirected` from `experiments.run_hybrid` (with graceful fallback)
- **Updated ALGO_MAP**: Now includes GNN-KMA as a selectable algorithm option
- **Modified run_algorithm()**: Added conditional handling for GNN-KMA algorithm
- **Updated argparse**: Added "GNN-KMA" to choices list
- **Updated docstring**: Added examples showing how to run GNN-KMA
- **Updated summary printing**: Now prints GNN-KMA results alongside BST, IC, MA

**New supported commands:**
```bash
# Run GNN-KMA on a single file
python experiments/benchmark_undirected.py --algo GNN-KMA --test data/raw_undirected/graph01.txt

# Run GNN-KMA with custom parameters
python experiments/benchmark_undirected.py --algo GNN-KMA --test data/raw_undirected/graph01.txt --pop 100 --gens 400

# Run GNN-KMA on all files in a folder
python experiments/benchmark_undirected.py --algo GNN-KMA --test data/raw_undirected/ --output results_GNN-KMA.csv

# Compare ALL algorithms (including GNN-KMA)
python experiments/benchmark_undirected.py --algo ALL --test data/raw_undirected/ --output results_comparison.csv
```

## How It Works

### Algorithm Selection
- **GNN-KMA**: Uses the `gnn_kme_solve_directed()` or `gnn_kme_solve_undirected()` function from `run_hybrid.py`
- **Graceful degradation**: If the GNN weights are missing, automatically falls back to pure MA
- **Consistent timing**: The elapsed time includes both GNN inference and MA refinement

### Integration Approach
The GNN-KMA algorithm is integrated using a wrapper approach:
1. When `--algo GNN-KMA` is selected, the code imports and calls the GNN-KMA solver function
2. The GNN-KMA solver performs:
   - GNN inference to predict FVS candidates
   - MA refinement using the GNN predictions as guidance
3. Results are collected and reported in the same format as other algorithms

### Parameter Support
- `--pop`: Population size for MA (default: 50)
- `--gens`: Maximum generations for MA (default: 200)
- Both parameters apply to the MA phase of the GNN-KMA algorithm

## Usage Examples

### Example 1: Run GNN-KMA on PACE dataset (Directed)
```bash
python experiments/benchmark_directed.py --algo GNN-KMA --test data/pace2022/ --output pace_GNN-KMA_results.csv
```

### Example 2: Compare all algorithms on undirected graphs
```bash
python experiments/benchmark_undirected.py --algo ALL --test data/raw_undirected/ --output all_algos_comparison.csv
```

### Example 3: Run GNN-KMA with custom MA parameters
```bash
python experiments/benchmark_directed.py --algo GNN-KMA --test data/raw_directed/ --pop 150 --gens 500
```

### Example 4: Batch run GNN-KMA with quiet output (file/CSV only)
```bash
python experiments/benchmark_undirected.py --algo GNN-KMA --test data/raw_undirected/ --output results.csv --quiet
```

## Technical Details

### GNN-KMA Algorithm Workflow
1. **Graph Parsing**: Supports PACE .gr format, METIS, and edge-list formats
2. **Graph Type Detection**: Automatically handles directed vs. undirected graphs
3. **GNN Inference Phase**:
   - Loads trained GNN model (if available)
   - Computes per-vertex features (degree, clustering, etc.)
   - Predicts likelihood of vertices being in FVS
4. **MA Refinement Phase**:
   - Uses GNN predictions to warm-start MA population
   - Runs memetic algorithm for population_size and max_generations
   - Falls back to pure MA if GNN is unavailable
5. **Validation**: Uses the same verification functions as other algorithms

### Performance Characteristics
- **Time**: Includes GNN inference time (if models available) + MA execution time
- **Solution Quality**: Typically better than pure MA, worse than exact algorithms (BST, IC)
- **Scalability**: Handles large graphs (10k+ vertices) unlike exact algorithms

## Testing Recommendations

1. **Small test**: Run on a single PACE file
   ```bash
   python experiments/benchmark_directed.py --algo GNN-KMA --test data/pace2022/h_001
   ```

2. **Comparison**: Run ALL algorithms on a few instances to see GNN-KMA's relative performance
   ```bash
   python experiments/benchmark_directed.py --algo ALL --test data/pace2022/ --quiet
   ```

3. **Batch processing**: Run on full PACE dataset with output to CSV
   ```bash
   python experiments/benchmark_directed.py --algo GNN-KMA --test data/pace2022/ --output pace_GNN-KMA_full.csv
   ```

## Notes

- **GNN Weights**: The GNN-KMA algorithm requires trained GNN weights. If not available:
  - It will print a message about skipping the GNN step
  - It will automatically fall back to pure MA
  - The script will NOT crash - it degrades gracefully

- **Consistency**: GNN-KMA results are reported in the same CSV/table format as other algorithms
- **ALL comparison**: Using `--algo ALL` now runs 4 algorithms instead of 3 (BST, IC, MA, GNN-KMA)
- **Backward compatible**: Existing benchmark commands still work exactly as before

## Files Modified
1. `experiments/benchmark_directed.py` - Added GNN-KMA support for directed graphs
2. `experiments/benchmark_undirected.py` - Added GNN-KMA support for undirected graphs

## No Changes Required To
- `experiments/run_hybrid.py` - Already has complete GNN-KMA solver implementation
- `experiments/test_pace_all.py` - Can now use benchmark scripts for GNN-KMA instead
