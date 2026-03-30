#!/usr/bin/env python3
"""
benchmark_undirected.py
=======================
Command-Line Interface for running and benchmarking undirected FVS algorithms.

Usage examples
──────────────
# 1. Single algorithm on a single file
python experiments/benchmark_undirected.py --algo BST --test data/raw_undirected/graph01.txt

# 2. Iterative Compression on one file
python experiments/benchmark_undirected.py --algo IC --test data/raw_undirected/graph01.txt

# 3. Memetic Algorithm with custom parameters
python experiments/benchmark_undirected.py --algo MA --test data/raw_undirected/graph01.txt --pop 100 --gens 500

# 4. Run ALL algorithms and compare
python experiments/benchmark_undirected.py --algo ALL --test data/raw_undirected/graph01.txt

# 5. Batch: run MA on every file in a folder, save CSV
python experiments/benchmark_undirected.py --algo MA --test data/raw_undirected/ --output results.csv

# 6. Batch ALL algorithms on every file in a folder
python experiments/benchmark_undirected.py --algo ALL --test data/synthetic/ --output comparison.csv

Supported --algo values
───────────────────────
  BST    — Bounded Search Tree (exact, slow for large graphs)
  IC     — Iterative Compression (exact, faster in practice)
  MA     — Memetic Algorithm (heuristic, fast, scales to 10k+ vertices)
  ALL    — Run BST, IC, and MA on the same graph; print comparison table

File format (EdgeList)
──────────────────────
Lines starting with '#' or '%' are comments.
Each data line: "u v"  (space/tab separated, 0- or 1-indexed)
The script auto-detects 0-vs-1 indexing and normalizes to 0-indexed.
"""

import argparse
import os
import sys
import time
import csv
from pathlib import Path
from typing import List, Tuple, Optional

# ── Add cpp_engine to path ────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "cpp_engine" / "build"))

try:
    import cpp_engine
except ImportError as e:
    print("ERROR: Cannot import cpp_engine.")
    print("       Did you compile it?  Run inside cpp_engine/:")
    print("         mkdir -p build && cd build && cmake .. && make")
    print(f"       (Original error: {e})")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
#  File Parsing
# ═══════════════════════════════════════════════════════════════════════════════

def parse_graph_file(filepath: str) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Parse an edge-list graph file.

    Supports:
      - Lines starting with '#' or '%': comments (skipped)
      - Lines with 'p edge N M': DIMACS header (sets n = N)
      - Lines with two integers 'u v': an edge
      - Lines with three integers 'u v w': edge with weight (weight ignored)

    Auto-detects 1-indexed vs 0-indexed and normalizes to 0-indexed.

    Returns:
        (n, edges) where n is the vertex count and edges is a list of (u, v) pairs.
    """
    edges: List[Tuple[int, int]] = []
    n_hint: Optional[int] = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('%'):
                continue

            parts = line.split()

            # DIMACS "p edge N M" header
            if parts[0].lower() == 'p' and len(parts) >= 3:
                try:
                    n_hint = int(parts[2])
                except ValueError:
                    pass
                continue

            # Skip other non-numeric header lines
            if not parts[0].lstrip('-').isdigit():
                continue

            try:
                u = int(parts[0])
                v = int(parts[1])
                edges.append((u, v))
            except (ValueError, IndexError):
                continue

    if not edges:
        raise ValueError(f"No edges found in {filepath}")

    # Determine vertex count and indexing
    all_verts = set()
    for u, v in edges:
        all_verts.add(u)
        all_verts.add(v)

    min_v = min(all_verts)
    max_v = max(all_verts)

    # Normalize to 0-indexed
    if min_v == 1:  # 1-indexed input
        edges = [(u - 1, v - 1) for u, v in edges]
        max_v -= 1

    n = n_hint if n_hint is not None else max_v + 1
    n = max(n, max_v + 1)  # ensure n is large enough

    return n, edges


def verify_fvs(n: int, edges: List[Tuple[int, int]], fvs: List[int]) -> bool:
    """
    Verify that `fvs` is a valid Feedback Vertex Set for the undirected graph.
    Removes FVS vertices, then checks if the remaining graph is a forest.
    """
    fvs_set = set(fvs)
    adj: dict = {v: set() for v in range(n)}
    for u, v in edges:
        if u not in fvs_set and v not in fvs_set:
            adj[u].add(v)
            adj[v].add(u)

    # DFS cycle check
    color = [0] * n
    def has_cycle(start: int) -> bool:
        stack = [(start, -1)]
        while stack:
            node, parent = stack.pop()
            if color[node] == 2:
                continue
            if color[node] == 1:
                return True
            color[node] = 1
            for nb in adj[node]:
                if nb == parent:
                    continue
                if color[nb] == 1:
                    return True
                if color[nb] == 0:
                    stack.append((nb, node))
            color[node] = 2
        return False

    for v in range(n):
        if v not in fvs_set and color[v] == 0:
            if has_cycle(v):
                return False
    return True


# ═══════════════════════════════════════════════════════════════════════════════
#  Algorithm Runners
# ═══════════════════════════════════════════════════════════════════════════════

ALGO_MAP = {
    "BST": cpp_engine.solve_undirected_BST,
    "IC":  cpp_engine.solve_undirected_IC,
    "MA":  cpp_engine.solve_undirected_MA,
}

def run_algorithm(algo: str, n: int, edges: List[Tuple[int, int]],
                  pop_size: int = 50, max_gens: int = 200) -> Tuple[List[int], float]:
    """
    Run a single algorithm and return (fvs, elapsed_ms).
    """
    start = time.perf_counter()

    if algo == "MA":
        fvs = cpp_engine.solve_undirected_MA(n, edges, pop_size, max_gens)
    else:
        fn = ALGO_MAP[algo]
        fvs = fn(n, edges)

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return fvs, elapsed_ms


def run_on_file(filepath: str, algo: str, pop_size: int, max_gens: int,
                verbose: bool = True) -> dict:
    """
    Parse a graph file, run the specified algorithm(s), and return results dict.
    """
    try:
        n, edges = parse_graph_file(filepath)
    except Exception as ex:
        print(f"  [SKIP] Could not parse {filepath}: {ex}")
        return {}

    filename = Path(filepath).name
    results = {"file": filename, "n": n, "m": len(edges)}

    if verbose:
        print(f"\n{'─' * 60}")
        print(f"  File : {filename}")
        print(f"  Graph: {n} vertices, {len(edges)} edges")
        print(f"{'─' * 60}")

    algos_to_run = list(ALGO_MAP.keys()) if algo == "ALL" else [algo]

    for alg in algos_to_run:
        if verbose:
            print(f"  Running {alg:4s} ... ", end="", flush=True)

        fvs, elapsed_ms = run_algorithm(alg, n, edges, pop_size, max_gens)
        valid = verify_fvs(n, edges, fvs)

        if verbose:
            status = "✓ VALID" if valid else "✗ INVALID"
            print(f"FVS size = {len(fvs):4d}  |  Time = {elapsed_ms:8.2f} ms  |  {status}")

        results[f"{alg}_size"]  = len(fvs)
        results[f"{alg}_ms"]    = round(elapsed_ms, 2)
        results[f"{alg}_valid"] = valid

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="FVS Undirected Benchmark CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--algo", required=True,
        choices=["BST", "IC", "MA", "ALL"],
        help="Algorithm to run: BST (exact), IC (exact), MA (heuristic), ALL (compare)"
    )
    parser.add_argument(
        "--test", required=True,
        help="Path to a single graph file OR a directory of graph files"
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional: save results to this CSV file"
    )
    parser.add_argument(
        "--pop", type=int, default=50,
        help="[MA only] Population size (default: 50)"
    )
    parser.add_argument(
        "--gens", type=int, default=200,
        help="[MA only] Maximum generations (default: 200)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-run output (only print summary / CSV)"
    )

    args = parser.parse_args()

    # ── Collect input files ──────────────────────────────────────────────────
    test_path = Path(args.test)
    if test_path.is_file():
        files = [str(test_path)]
    elif test_path.is_dir():
        extensions = (".txt", ".gr", ".edges", ".graph", ".dimacs", ".mtx")
        files = sorted(
            str(f) for f in test_path.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        )
        if not files:
            print(f"No graph files found in {test_path}")
            print(f"Expected extensions: {extensions}")
            sys.exit(1)
        print(f"Found {len(files)} graph file(s) in {test_path}")
    else:
        print(f"ERROR: --test path does not exist: {args.test}")
        sys.exit(1)

    # ── Run benchmarks ───────────────────────────────────────────────────────
    all_results = []
    for filepath in files:
        result = run_on_file(filepath, args.algo,
                             args.pop, args.gens,
                             verbose=not args.quiet)
        if result:
            all_results.append(result)

    # ── Print summary table ──────────────────────────────────────────────────
    if len(all_results) > 1 or args.quiet:
        print(f"\n{'═' * 80}")
        print(f"  SUMMARY  ({args.algo} on {len(all_results)} file(s))")
        print(f"{'═' * 80}")

        algos_ran = list(ALGO_MAP.keys()) if args.algo == "ALL" else [args.algo]
        header = f"  {'File':<30} {'n':>6} {'m':>8}"
        for alg in algos_ran:
            header += f"  {alg+' size':>10} {alg+' ms':>10}"
        print(header)
        print("  " + "─" * (len(header) - 2))

        for r in all_results:
            row = f"  {r['file']:<30} {r['n']:>6} {r['m']:>8}"
            for alg in algos_ran:
                sz = r.get(f"{alg}_size", "N/A")
                ms = r.get(f"{alg}_ms", "N/A")
                row += f"  {str(sz):>10} {str(ms):>10}"
            print(row)

    # ── Save CSV ─────────────────────────────────────────────────────────────
    if args.output:
        outpath = Path(args.output)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        with open(outpath, "w", newline="") as csvfile:
            if all_results:
                writer = csv.DictWriter(csvfile, fieldnames=all_results[0].keys())
                writer.writeheader()
                writer.writerows(all_results)
        print(f"\n  Results saved to: {outpath}")


if __name__ == "__main__":
    main()