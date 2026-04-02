#!/usr/bin/env python3
"""
test_pace_all.py
================
Helper script to test all algorithms on PACE 2022 instances.

Since benchmark_directed.py and benchmark_undirected.py already support
batch mode with folder inputs, you can use them directly:

DIRECTED GRAPHS - EXAMPLES:
  # Run all algorithms on all directed PACE instances
  python experiments/benchmark_directed.py --algo ALL --test data/pace2022/

  # Run memetic algorithm on all directed PACE instances
  python experiments/benchmark_directed.py --algo MA --test data/pace2022/ --output pace_directed_results.csv

  # Run iterative compression on directed PACE instances
  python experiments/benchmark_directed.py --algo IC --test data/pace2022/ --output pace_ic_results.csv

  # Run bounded search tree on directed PACE instances
  python experiments/benchmark_directed.py --algo BST --test data/pace2022/ --output pace_bst_results.csv

GNN-KMA MODE (GNN-Guided Memetic):
  python experiments/run_hybrid.py --graph data/pace2022/ --type directed --pop 150 --gens 500

UNDIRECTED GRAPHS:
  python experiments/benchmark_undirected.py --algo ALL --test data/pace2022/un --output pace_undirected_results.csv

═══════════════════════════════════════════════════════════════════════════════

This script provides a unified runner for all algorithms if you prefer a single command.
It automatically runs IC, BST, MA, and GNN-KMA on all PACE instances.
"""

import sys
import csv
import time
import subprocess
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def run_algorithm_on_instance(algo: str, graph_file: str, graph_type: str = "directed",
                               pop: int = 100, gens: int = 300) -> Dict:
    """
    Run a single algorithm on a single graph file.
    Returns: dict with runtime, fvs_size, status, error info
    """
    result = {
        "algo": algo,
        "runtime": None,
        "fvs_size": None,
        "status": "UNKNOWN",
        "error": None,
    }

    try:
        start = time.time()

        if algo in {"GNN-KMA", "GNN-KMA-2"}:
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "experiments" / "run_hybrid.py"),
                "--graph", graph_file,
                "--type", graph_type,
                "--pop", str(pop),
                "--gens", str(gens),
            ]
            if algo == "GNN-KMA-2":
                cmd.extend(["--mode", "GNN-KMA-2"])
        else:
            script_name = f"benchmark_{graph_type}.py"
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "experiments" / script_name),
                "--algo", algo,
                "--test", graph_file,
            ]
            if algo == "MA":
                cmd.extend(["--pop", str(pop), "--gens", str(gens)])

        # Run with timeout (300 seconds = 5 minutes per instance)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        elapsed = time.time() - start
        result["runtime"] = elapsed

        # Parse output to extract FVS size and status
        output_text = proc.stdout + "\n" + proc.stderr

        for line in output_text.split("\n"):
            if "size" in line.lower() and any(c.isdigit() for c in line):
                parts = line.split()
                for part in parts:
                    if part.isdigit():
                        result["fvs_size"] = int(part)
                        break
            if "valid" in line.lower():
                result["status"] = "VALID"
            elif "invalid" in line.lower() or "failed" in line.lower():
                result["status"] = "INVALID"

        if result["status"] == "UNKNOWN":
            result["status"] = "SUCCESS" if proc.returncode == 0 else "FAILED"

    except subprocess.TimeoutExpired:
        result["status"] = "TIMEOUT"
        result["error"] = "Timeout after 5 minutes"
    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Test all algorithms on PACE instances",
        epilog="Example: python test_pace_all.py --type directed --pop 150 --gens 500"
    )
    parser.add_argument("--type", choices=["directed", "undirected"], default="directed",
                       help="Graph type")
    parser.add_argument("--filter", default="", help="Filter instances by name pattern")
    parser.add_argument("--pop", type=int, default=100, help="Population size for MA/GNN-KMA variants")
    parser.add_argument("--gens", type=int, default=300, help="Generations for MA/GNN-KMA variants")
    parser.add_argument("--algorithms", default="IC,BST,MA,GNN-KMA,GNN-KMA-2",
                       help="Comma-separated algorithms to test")
    args = parser.parse_args()

    # Paths
    data_dir = PROJECT_ROOT / "data" / "pace2022"
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_csv = results_dir / f"pace_{args.type}_results.csv"

    # Validate data directory
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        print(f"Download PACE data first:")
        print(f"  python download_pace_data.py")
        sys.exit(1)

    # Collect graph files
    graph_files = sorted(data_dir.glob("*.gr"))
    if args.filter:
        graph_files = [f for f in graph_files if args.filter.lower() in f.name.lower()]

    if not graph_files:
        print(f"No .gr files found in {data_dir}")
        sys.exit(1)

    algos = [a.strip().upper() for a in args.algorithms.split(",")]

    print(f"\n{'='*80}")
    print(f"PACE 2022 Algorithm Benchmark")
    print(f"{'='*80}")
    print(f"Type:         {args.type}")
    print(f"Instances:    {len(graph_files)}")
    print(f"Algorithms:   {', '.join(algos)}")
    print(f"Output:       {output_csv}")
    print(f"{'='*80}\n")

    # Prepare CSV output
    with open(output_csv, 'w', newline='') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=[
            'instance', 'graph_file', 'algorithm', 'fvs_size', 'runtime_sec', 'status', 'error'
        ])
        csv_writer.writeheader()

        total = len(graph_files) * len(algos)
        count = 0
        results_by_instance = defaultdict(dict)

        for graph_file in graph_files:
            print(f"\n{graph_file.name}")
            print("-" * 80)

            instance_name = graph_file.stem

            for algo in algos:
                count += 1
                print(f"  [{count:3d}/{total:3d}] {algo:8s} ... ", end="", flush=True)

                result = run_algorithm_on_instance(algo, str(graph_file), args.type, args.pop, args.gens)

                status_str = f"{result['status']:10s}"
                fvs_str = f"{result['fvs_size'] or 'N/A':>6s}"
                time_str = f"{result['runtime']:7.2f}s" if result['runtime'] else "N/A    "

                print(f"{status_str} | FVS: {fvs_str} | Time: {time_str}")

                csv_writer.writerow({
                    'instance': instance_name,
                    'graph_file': graph_file.name,
                    'algorithm': algo,
                    'fvs_size': result['fvs_size'],
                    'runtime_sec': result['runtime'],
                    'status': result['status'],
                    'error': result['error'],
                })
                csv_file.flush()

                results_by_instance[instance_name][algo] = result

    # Print summary
    print(f"\n\n{'='*80}")
    print("SUMMARY - Best Algorithm per Instance")
    print(f"{'='*80}\n")
    print(f"{'Instance':<40} | {'Best Algo':<8} | {'FVS Size':<10}")
    print("-" * 70)

    for instance in sorted(results_by_instance.keys()):
        algos_results = results_by_instance[instance]
        valid_results = {a: r for a, r in algos_results.items()
                        if r['status'] == 'VALID' and r['fvs_size'] is not None}

        if valid_results:
            best_algo = min(valid_results.keys(),
                           key=lambda a: valid_results[a]['fvs_size'])
            best_size = valid_results[best_algo]['fvs_size']
            print(f"{instance:<40} | {best_algo:<8} | {best_size:<10}")

    print(f"\n✓ Results saved to: {output_csv}\n")


if __name__ == "__main__":
    main()
