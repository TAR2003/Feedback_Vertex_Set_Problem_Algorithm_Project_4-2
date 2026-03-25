# Feedback Vertex Set Algorithms: Comprehensive Analysis of `results/report.csv`

## 1. Executive Summary

This report analyzes the benchmark results in `results/report.csv` for the three FVS algorithms:
- `IC` (Iterative Compression)
- `KBST` (Kernelization + BST)
- `MEMETIC` (Memetic Genetic Algorithm)

### High-level verdict

1. `KBST` is the best overall algorithm in this dataset: it is the fastest on most instances and has the best approximation quality among the three on ground-truth cases.
2. `IC` is usually fast but has severe runtime outliers on a small number of hard cycle-heavy instances.
3. `MEMETIC` is consistently much slower (often by orders of magnitude) and generally produces worse FVS sizes in this dataset, despite always returning valid solutions.
4. There are strong signs that all three algorithms frequently fall back to near-trivial large FVS solutions (often deleting all vertices), which is a major quality concern.

## 2. Data Scope and Coverage

### Dataset size

- Total CSV rows: `796`
- Rows for the 3 target algorithms (`IC`, `KBST`, `MEMETIC`): `763`

### Per-algorithm row counts

- `IC`: `251`
- `KBST`: `251`
- `MEMETIC`: `261`

Reason for mismatch: `EXP7` has only `MEMETIC` rows (10 extra rows), and experiment coverage is not perfectly symmetric.

### Experiment coverage (three algorithms)

- `EXP1`: `201` rows, `67` instances
- `EXP2`: `219` rows, `73` instances
- `EXP3`: `219` rows, `73` instances
- `EXP8` (ground-truth comparison): `99` rows, `33` instances
- `EXP9`: `15` rows, `5` instances
- `EXP7`: `10` rows, `10` instances (MEMETIC only)

### Instance diversity

- Graph types present: `BA`, `CycleHeavy`, `ER`, `Grid`, `Tree`, `WS`, `realworld`
- Vertex count range: `5` to `127`

## 3. Correctness and Validity

### Constraint validity

- Invalid solutions (`is_valid_solution != True`) across the three algorithms: `0`

Interpretation:
- All algorithms satisfy the FVS validity check in this dataset.
- This is a strong positive signal for implementation correctness of the cycle-breaking constraint.

### Important caveat

Validity alone is not enough. A very large FVS (including deleting almost everything) can still be valid. Quality must be judged using `fvs_size` and ground-truth comparisons (EXP8).

## 4. Solution Quality (Ground Truth in EXP8)

`EXP8` is the most important quality experiment because it includes `optimal_fvs_size` and approximation metrics.

### Approximation ratio (lower is better, 1.0 is optimal)

- `IC`: mean `1.9826`, median `1.9091`, max `3.3333`
- `KBST`: mean `1.8164`, median `1.6667`, max `3.3333`
- `MEMETIC`: mean `2.6498`, median `2.2222`, max `10.0000`

### Optimality gap % (lower is better)

- `IC`: mean `98.26%`, median `90.91%`, max `233.33%`
- `KBST`: mean `81.64%`, median `66.67%`, max `233.33%`
- `MEMETIC`: mean `164.98%`, median `122.22%`, max `900.00%`

### Exact-optimal hit rate in EXP8 (`approximation_ratio == 1.0`)

- `IC`: `18.18%`
- `KBST`: `33.33%`
- `MEMETIC`: `21.21%`

### Overshoot over optimum (`fvs_size - optimal_fvs_size`)

- `IC`: mean `5.94`, median `6`, max `14`
- `KBST`: mean `5.61`, median `6`, max `14`
- `MEMETIC`: mean `7.52`, median `8`, max `18`

### Quality threshold counts in EXP8

- `IC`: `<=1.1` ratio: 6/33, `<=1.5`: 11/33, `>2.0`: 13/33
- `KBST`: `<=1.1` ratio: 11/33, `<=1.5`: 16/33, `>2.0`: 11/33
- `MEMETIC`: `<=1.1` ratio: 7/33, `<=1.5`: 10/33, `>2.0`: 17/33

### Worst observed quality failures (EXP8)

- `MEMETIC` on `ER_n20_p0.1_seed1`: ratio `10.0`, gap `900%` (`fvs_size=20`, optimum=2)
- `MEMETIC` on `BA_n20_m2_seed1`: ratio `6.6667`, gap `566.67%`
- `MEMETIC` on `ER_n10_p0.3_seed1` and `BA_n10_m2_seed1`: ratio `5.0`, gap `400%`
- `MEMETIC` on `Grid_3x3`: ratio `4.5`, gap `350%`

Interpretation:
- Quality is not just slightly suboptimal; in many cases it is dramatically suboptimal.
- `KBST` is the strongest of the three on quality, but still far from ideal on many instances.

## 5. Runtime Performance

## Global runtime summary (`wall_time_sec`)

- `IC`: mean `12.7808`, median `0.001092`, max `1032.3576`
- `KBST`: mean `0.001816`, median `0.000538`, max `0.020069`
- `MEMETIC`: mean `83.6632`, median `24.3702`, max `1042.5922`

Interpretation:
- `KBST` is consistently fast and stable.
- `IC` is usually very fast (tiny median), but has rare catastrophic outliers that inflate mean runtime massively.
- `MEMETIC` is systematically expensive: even median runtime is large.

### Pairwise speed ratios (same experiment + same instance)

- Median `IC / KBST`: `2.05x` slower
- Median `MEMETIC / KBST`: `25,261x` slower
- Median `MEMETIC / IC`: `15,962x` slower

The means are much larger due to extreme outliers.

### Fastest-algorithm share (across comparable groups)

- `KBST`: `90.42%`
- `IC`: `5.75%`
- `MEMETIC`: `3.83%`

### Largest runtime outliers

Top outliers include:
- `MEMETIC` on `ER_n50_p0.9_seed1`: `1042.59s`
- `MEMETIC` on `ER_n50_p0.9_seed1` (another run): `1039.32s`
- `IC` on `CycleHeavy_n50_density50_random_overlay`: `1032.36s`
- `IC` on same cycle-heavy instance in other runs: `1016.08s`, `1001.81s`
- `MEMETIC` on `ER_n50_p0.7_seed1`: around `814-815s`
- `MEMETIC` on `realworld_les_miserables`: around `528-534s`

## 6. Memory and CPU Usage

### CPU time trends

CPU time closely tracks wall-time behavior:
- `KBST`: tiny CPU times, stable
- `IC`: generally tiny with outlier spikes
- `MEMETIC`: very high CPU for large/hard instances

### Peak memory summary (`peak_memory_mb`)

- `IC`: mean `0.0060`, median `0.0`, max `1.085`
- `KBST`: mean `0.00016`, median `0.0`, max `0.029`
- `MEMETIC`: mean `0.0408`, median `0.0`, max `5.427`

Caution:
- Many memory entries are `0.000`, so memory logging appears coarse or below reporting resolution for many runs.
- Relative comparison still indicates `MEMETIC` tends to use more memory.

## 7. Structural Behavior by Graph Family

Using mean `fvs_size / n_vertices` across all available runs:

- `Tree`: near 0 (good; trees should need FVS size 0)
- `Grid`: KBST best (`~0.30`), IC moderate (`~0.43`), MEMETIC at `1.00` (worst)
- `CycleHeavy`: KBST (`~0.52`) better than IC (`~0.62`), MEMETIC at `1.00`
- `WS`: all algorithms at `1.00` (very concerning; all vertices removed on average)
- `BA`, `ER`, `realworld`: generally very high fractions, often near 1.0

Interpretation:
- Trees are handled correctly.
- For many non-tree families, especially `WS` and several dense/random instances, solutions degrade toward deleting almost all vertices.

## 8. Degenerate-Solution Pattern: Deleting All Vertices

A critical signal is the frequency of `fvs_size == n_vertices`.

Across all runs:
- `IC`: `153/251` = `61.0%`
- `KBST`: `150/251` = `59.8%`
- `MEMETIC`: `205/261` = `78.5%`

In `EXP8` only:
- `IC`: `20/33` = `60.6%`
- `KBST`: `20/33` = `60.6%`
- `MEMETIC`: `26/33` = `78.8%`

Interpretation:
- This is likely the central quality problem in this project version.
- The algorithms are often finding valid but overly large feedback sets, frequently the trivial maximum-size pattern.

## 9. Stability and Reproducibility Across Repeated Runs

For instances common to `EXP1`, `EXP2`, `EXP3` (67 shared instances):

- `fvs_size` consistency by instance is perfect for all three algorithms (`67/67` exact match each).
- Runtime variability (coefficient of variation):
  - `IC`: mean CV `0.062`
  - `KBST`: mean CV `0.077`
  - `MEMETIC`: mean CV `0.351`

Interpretation:
- Output size is deterministic/stable in these repeated experiments.
- `MEMETIC` runtime is much more variable than IC/KBST.

## 10. Data and Metric Anomalies

### One missing EXP8 approximation ratio

- `exp8_missing_ratio_rows = 1`
- This corresponds to a case with `optimal_fvs_size = 0` (division-by-zero situation for ratio).

This is not necessarily algorithm failure; it is a metric-definition edge case. A robust reporting rule should explicitly handle `optimal = 0`.

### Symmetry caveat

- Not all experiments have equal algorithm/instance coverage (`MEMETIC` has extra EXP7 rows).
- Cross-experiment aggregate means should be interpreted with this in mind.

## 11. What Looks Right vs What Looks Wrong

## What looks right

1. All three algorithms produce valid FVS solutions (`is_valid_solution=True` in all analyzed rows).
2. Tree handling is correct (near-zero FVS where expected).
3. `KBST` runtime behavior is excellent and highly practical.
4. Repeated runs show stable `fvs_size` outputs.

## What looks wrong or concerning

1. Approximation quality is weak overall in EXP8, with very large optimality gaps.
2. Frequent `fvs_size == n_vertices` suggests many solutions are close to trivial over-deletion.
3. `MEMETIC` is much slower and usually worse in quality in this dataset.
4. `IC` has severe runtime blowups on specific cycle-heavy instances.
5. Quality on `WS`, `CycleHeavy`, and parts of `BA/ER` appears structurally poor.

## 12. Algorithm-by-Algorithm Assessment

### IC

- Strengths: generally fast, deterministic output sizes, valid solutions.
- Weaknesses: substantial quality gaps in EXP8; catastrophic runtime outliers on hard cycle-heavy cases.
- Practical status: usable with caution; needs outlier mitigation and quality improvement.

### KBST

- Strengths: fastest in ~90% of comparable runs, best quality among the three in EXP8, stable.
- Weaknesses: still often far from optimal and also shows many all-vertex solutions.
- Practical status: best baseline in current project; strongest production candidate among three.

### MEMETIC

- Strengths: valid outputs and stable `fvs_size` across repeated runs.
- Weaknesses: very high runtime, high runtime variability, worst quality profile, highest rate of trivial/all-vertex outcomes.
- Practical status: currently not competitive; likely needs major retuning or redesign.

## 13. Recommended Next Checks (to verify root cause)

1. Verify objective function and penalty terms to ensure minimization of `fvs_size` is strongly enforced versus merely validity.
2. Add an explicit anti-triviality guard (e.g., reject/penalize solutions with very large vertex-removal fraction unless justified).
3. Audit stopping criteria and neighborhood operators in MEMETIC; tune for quality-first behavior before long runtime.
4. For IC, isolate the cycle-heavy outlier path and profile recursion/branching depth to prevent 1000+ second spikes.
5. In reporting code, handle `optimal_fvs_size = 0` explicitly to avoid missing approximation metrics.
6. Add per-graph-family targeted tests (especially WS and CycleHeavy) with expected quality thresholds.

## 14. Final Conclusion

From this CSV, the main issue is not validity, but **solution quality** and **runtime practicality**.

- If you need one algorithm to show as the current best: choose `KBST`.
- If you need to explain what is failing: many outputs are valid but too large, often approaching the trivial full-vertex deletion pattern.
- If you need to justify future work: focus on quality optimization first (especially reducing all-vertex outcomes), then on IC outlier control and MEMETIC redesign/tuning.
