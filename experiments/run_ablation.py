#!/usr/bin/env python3
"""
run_ablation.py
===============
Ablation study runner for GNN-KMA variants.

Runs KMA, GNN-KMA, GNN-KMA-2, and (optionally) GNN-KMA-3 on the same set
of test graphs and computes comparative statistics.

Usage:
  python experiments/run_ablation.py \
      --test data/raw_directed/ \
      --algos "KMA,GNN-KMA,GNN-KMA-2" \
      --thresholds "0.65" \
      --pop 50 --gens 200 \
      --output results/ablation/

Output CSV columns for each algorithm:
  algo, threshold, mean_fvs, std_fvs, median_fvs, mean_runtime_ms,
  pct_beats_kma, mean_improvement_vs_kma, validity_rate, n_instances

Reference: Research-grade ablation design from Part 8 of GNN-KMA overhaul.
"""

import argparse
import csv
import os
import sys
import time
import statistics
from pathlib import Path
from typing import List, Dict, Optional, Tuple

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for candidate in ("build", "build-linux", "build-macos", "build-win"):
    sys.path.insert(0, str(PROJECT_ROOT / "cpp_engine" / candidate))
sys.path.insert(0, str(SCRIPT_DIR))

try:
    import cpp_engine
    HAS_CPP_ENGINE = True
except ImportError:
    HAS_CPP_ENGINE = False

from run_hybrid import (
    kma_solve_directed,
    gnn_KMA_solve_directed,
    gnn_KMA2_solve_directed,
    gnn_KMA3_solve_directed,
)
from benchmark_directed import (
    parse_directed_graph_file,
    collect_graph_files,
    verify_dfvs,
)


def run_algorithm(
    algo: str,
    n: int,
    edges: List[Tuple[int, int]],
    pop_size: int,
    max_gens: int,
    threshold: float,
    timeout: int,
    early_stop: int,
) -> Tuple[Optional[List[int]], float, bool]:
    """
    Run one algorithm on one graph.

    Returns:
        (fvs, elapsed_ms, valid)
    """
    start = time.perf_counter()
    try:
        if algo == "KMA":
            fvs = kma_solve_directed(n, edges, pop_size, max_gens,
                                     max_time_seconds=timeout, early_stop=early_stop)
        elif algo == "GNN-KMA":
            fvs = gnn_KMA_solve_directed(n, edges, pop_size, max_gens,
                                          gnn_threshold=threshold,
                                          max_time_seconds=timeout, early_stop=early_stop)
        elif algo == "GNN-KMA-2":
            fvs = gnn_KMA2_solve_directed(n, edges, pop_size, max_gens,
                                           gnn_threshold=threshold,
                                           max_time_seconds=timeout, early_stop=early_stop)
        elif algo == "GNN-KMA-3":
            fvs = gnn_KMA3_solve_directed(n, edges, pop_size, max_gens,
                                           gnn_threshold=threshold,
                                           max_time_seconds=timeout, early_stop=early_stop)
        else:
            raise ValueError(f"Unknown algo: {algo}")
    except Exception as ex:
        print(f"    [{algo}] ERROR: {ex}")
        return None, (time.perf_counter() - start) * 1000.0, False

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    valid = verify_dfvs(n, edges, fvs) if fvs is not None else False
    return fvs, elapsed_ms, valid


def ablation_study(
    test_files: List[str],
    algos: List[str],
    thresholds: List[float],
    pop_size: int,
    max_gens: int,
    timeout: int,
    early_stop: int,
    output_dir: Path,
) -> None:
    """
    Run ablation study across all (algo, threshold) configs.

    For each config:
        - Run on all test graphs
        - Compute KMA baseline for each graph (used as reference)
        - Compute comparative statistics vs KMA

    Results saved to:
        output_dir/ablation_instance.csv   — per-instance results
        output_dir/ablation_summary.csv    — aggregated statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    instance_path = output_dir / "ablation_instance.csv"
    summary_path  = output_dir / "ablation_summary.csv"

    configs = []
    for algo in algos:
        if algo == "KMA":
            configs.append(("KMA", 0.0))  # threshold unused for KMA
        else:
            for thr in thresholds:
                configs.append((algo, thr))

    print(f"\n{'═' * 70}")
    print(f"  Ablation Study")
    print(f"  Test instances : {len(test_files)}")
    print(f"  Configurations : {len(configs)}")
    print(f"  Algorithms     : {algos}")
    print(f"  Thresholds     : {thresholds}")
    print(f"{'═' * 70}")

    # Pre-compute KMA baseline for all graphs
    kma_results: Dict[str, int] = {}
    print("\n  [1/2] Computing KMA baseline...")
    for fpath in test_files:
        fname = Path(fpath).name
        try:
            n, edges = parse_directed_graph_file(fpath)
        except Exception as ex:
            print(f"    [SKIP] {fname}: {ex}")
            continue
        fvs, _, valid = run_algorithm("KMA", n, edges, pop_size, max_gens, 0.0, timeout, early_stop)
        if fvs is not None and valid:
            kma_results[fname] = len(fvs)
            print(f"    KMA  {fname:40s}  FVS={len(fvs):4d}")

    # Run all configs
    print(f"\n  [2/2] Running {len(configs)} algo-threshold configs...")
    all_instance_rows: List[Dict] = []

    for algo, thr in configs:
        label = f"{algo}@{thr}" if algo != "KMA" else "KMA"
        fvs_sizes, runtimes, improvements, beats_kma = [], [], [], []
        valid_count = 0

        for fpath in test_files:
            fname = Path(fpath).name
            kma_size = kma_results.get(fname)
            if kma_size is None:
                continue

            try:
                n, edges = parse_directed_graph_file(fpath)
            except Exception:
                continue

            fvs, elapsed_ms, valid = run_algorithm(
                algo, n, edges, pop_size, max_gens, thr, timeout, early_stop
            )
            if fvs is None or not valid:
                print(f"    [{label}] {fname}: FAILED")
                continue

            size = len(fvs)
            impv = kma_size - size        # positive = beats KMA
            fvs_sizes.append(size)
            runtimes.append(elapsed_ms)
            improvements.append(impv)
            beats_kma.append(1 if impv > 0 else 0)
            valid_count += 1

            all_instance_rows.append({
                "algo": algo,
                "threshold": thr,
                "file": fname,
                "n": n,
                "m": len(edges),
                "fvs_size": size,
                "kma_size": kma_size,
                "improvement_vs_kma": impv,
                "runtime_ms": round(elapsed_ms, 2),
                "valid": valid,
            })
            print(f"    [{label}] {fname:40s}  FVS={size:4d}  Δ={impv:+3d}  {elapsed_ms:.0f}ms")

        if not fvs_sizes:
            continue

        summary_row = {
            "algo": algo,
            "threshold": thr,
            "n_instances": valid_count,
            "mean_fvs": round(statistics.mean(fvs_sizes), 3),
            "std_fvs":  round(statistics.stdev(fvs_sizes) if len(fvs_sizes) > 1 else 0.0, 3),
            "median_fvs": round(statistics.median(fvs_sizes), 3),
            "mean_runtime_ms": round(statistics.mean(runtimes), 2),
            "mean_improvement_vs_kma": round(statistics.mean(improvements), 3),
            "pct_beats_kma": round(100.0 * sum(beats_kma) / max(len(beats_kma), 1), 1),
            "validity_rate": round(100.0 * valid_count / max(len(test_files), 1), 1),
        }
        print(f"\n  → [{label}] Summary: "
              f"mean_FVS={summary_row['mean_fvs']:.2f}  "
              f"beats_KMA={summary_row['pct_beats_kma']:.1f}%  "
              f"Δmean={summary_row['mean_improvement_vs_kma']:+.2f}")

    # Write instance CSV
    if all_instance_rows:
        with open(instance_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_instance_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_instance_rows)
        print(f"\n  ✓ Instance results: {instance_path}")

    print(f"\n  ✓ Summary:          {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="GNN-KMA Ablation Study Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--test", required=True,
                        help="Path to directory of .gr/.txt graph files")
    parser.add_argument("--algos", default="KMA,GNN-KMA,GNN-KMA-2",
                        help="Comma-separated list of algorithms to compare")
    parser.add_argument("--thresholds", default="0.65",
                        help="Comma-separated GNN probability thresholds to sweep")
    parser.add_argument("--pop", type=int, default=50)
    parser.add_argument("--gens", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--earlystop", type=int, default=30)
    parser.add_argument("--output", default="results/ablation/")
    args = parser.parse_args()

    if not HAS_CPP_ENGINE:
        print("ERROR: cpp_engine not available")
        sys.exit(1)

    test_path = Path(args.test)
    if not test_path.exists():
        print(f"ERROR: --test path not found: {args.test}")
        sys.exit(1)

    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    thresholds = [float(t.strip()) for t in args.thresholds.split(",") if t.strip()]
    output_dir = Path(args.output)

    if test_path.is_dir():
        test_files = collect_graph_files(test_path)
    else:
        test_files = [str(test_path)]

    if not test_files:
        print(f"ERROR: No graph files found in {args.test}")
        sys.exit(1)

    ablation_study(
        test_files=test_files,
        algos=algos,
        thresholds=thresholds,
        pop_size=args.pop,
        max_gens=args.gens,
        timeout=args.timeout,
        early_stop=args.earlystop,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
