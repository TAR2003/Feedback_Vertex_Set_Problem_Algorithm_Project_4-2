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

# 7. GNN-KMA (GNN-guided KMA) on one file with custom parameters
python experiments/benchmark_undirected.py --algo GNN-KMA --test data/raw_undirected/graph01.txt --pop 100 --gens 400

# 8. GNN-KMA on a folder of graphs
python experiments/benchmark_undirected.py --algo GNN-KMA --test data/raw_undirected/ --output undirected_GNN-KMA_results.csv

Supported --algo values
───────────────────────
  BST    — Bounded Search Tree (exact, slow for large graphs)
  IC     — Iterative Compression (exact, faster in practice)
  MA     — Memetic Algorithm (heuristic, fast, scales to 10k+ vertices)
    KMA    — Kernelized Memetic Algorithm (kernelization + MA)
    GNN-KMA — GNN-guided KMA (combines GNN inference + kernelized MA refinement)
    ALL    — Run BST, IC, MA, KMA, and GNN-KMA on the same graph; print comparison table

File format (Universal TXT edge-list)
──────────────────────────────────────
Supported examples:
    # format: edge_list_v1
    p edge N M
    u v
or plain two-column edge lines.
Lines starting with '#', '%' are comments.
The script auto-detects 0-vs-1 indexing and normalizes to 0-indexed.
"""

import argparse
import os
import sys
import time
import csv
import multiprocessing as mp
import queue
from pathlib import Path
from typing import List, Tuple, Optional

# ── Add cpp_engine to path ────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
# On Windows with MinGW-built extension modules, ensure runtime DLLs are discoverable.
if os.name == "nt":
    mingw_bin = Path("C:/msys64/mingw64/bin")
    if mingw_bin.exists():
        os.environ["PATH"] = str(mingw_bin) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(mingw_bin))
# Insert in reverse-priority order because each entry is inserted at sys.path[0].
for candidate in ("build", "build-linux", "build-macos", "build-win"):
    sys.path.insert(0, str(PROJECT_ROOT / "cpp_engine" / candidate))
# Try experiments first (where the .so file is compiled) - insert last so it's first in path
sys.path.insert(0, str(SCRIPT_DIR))

try:
    import cpp_engine
except ImportError as e:
    print("ERROR: Cannot import cpp_engine.")
    print("       Did you compile it?  Run inside cpp_engine/:")
    print("         mkdir -p build && cd build && cmake .. && make")
    print(f"       (Original error: {e})")
    sys.exit(1)

# Try importing GNN-KMA solver from run_hybrid.py (graceful fallback)
# Note: Import is deferred until GNN-KMA algorithm is actually requested to avoid slow startup
HAS_GNN_KME = True  # Assume available; will fail gracefully at runtime if not


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


def collect_graph_files(root: Path) -> List[str]:
    """Recursively collect graph files from nested dataset folders."""
    extensions = {".txt", ".gr", ".edges", ".graph", ".dimacs", ".mtx"}
    files: List[str] = []
    for f in sorted(root.rglob("*")):
        if not f.is_file():
            continue
        if f.suffix.lower() in extensions:
            files.append(str(f))
        elif f.suffix == "" and not f.name.startswith('.'):
            files.append(str(f))
    return files


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
#  CSV Result Tracking
# ═══════════════════════════════════════════════════════════════════════════════

def get_results_csv_path(results_dir: str, algo: str, is_directed: bool = False) -> Path:
    """
    Get the CSV file path for a specific algorithm.
    E.g., results/undirected_BST.csv or results/directed_BST.csv
    """
    prefix = "directed" if is_directed else "undirected"
    csv_name = f"{prefix}_{algo}.csv"
    return Path(results_dir) / csv_name


def load_existing_results(csv_path: Path) -> dict:
    """
    Load existing CSV results into a dict keyed by filename.
    Returns empty dict if file doesn't exist.
    """
    results_by_file = {}
    if not csv_path.exists():
        return results_by_file
    
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "file" in row:
                    results_by_file[row["file"]] = row
    except Exception:
        pass
    
    return results_by_file


def is_result_already_recorded(csv_path: Path, filename: str) -> bool:
    """
    Check if a filename is already in the CSV file.
    """
    if not csv_path.exists():
        return False
    
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("file") == filename:
                    return True
    except Exception:
        pass
    
    return False


def append_result_to_csv(csv_path: Path, result_dict: dict) -> None:
    """
    Append a single result row to a CSV file.
    Creates file and header if it doesn't exist.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_exists = csv_path.exists()
    
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result_dict)


def _to_float_or_none(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Algorithm Runners
# ═══════════════════════════════════════════════════════════════════════════════

ALGO_MAP = {
    "BST": cpp_engine.solve_undirected_BST,
    "IC":  cpp_engine.solve_undirected_IC,
    "MA":  cpp_engine.solve_undirected_MA,
    "KMA": getattr(cpp_engine, "solve_undirected_KMA", getattr(cpp_engine, "solve_undirected_KME", cpp_engine.solve_undirected_MA)),
}


def get_dynamic_timeout_seconds(n: int) -> int:
    """
    Dynamic timeout policy based on graph size.
      n <= 50         -> 100s
      50 < n <= 200   -> 500s
      n > 200         -> 600s
    """
    if n <= 50:
        return 100
    if n <= 200:
        return 500
    return 600


def _undirected_worker_run(
    algo: str,
    n: int,
    edges: List[Tuple[int, int]],
    pop_size: int,
    max_gens: int,
    gnn_threshold: float,
    gnn_hidden: int | None,
    out_q: mp.Queue,
) -> None:
    """Child-process worker that runs one algorithm and returns via queue."""
    try:
        start = time.perf_counter()
        if algo == "MA":
            fvs = cpp_engine.solve_undirected_MA(n, edges, pop_size, max_gens)
        elif algo == "KMA":
            if hasattr(cpp_engine, "solve_undirected_KMA"):
                fvs = cpp_engine.solve_undirected_KMA(n, edges, pop_size, max_gens)
            elif hasattr(cpp_engine, "solve_undirected_KME"):
                fvs = cpp_engine.solve_undirected_KME(n, edges, pop_size, max_gens)
            else:
                fvs = cpp_engine.solve_undirected_MA(n, edges, pop_size, max_gens)
        elif algo == "GNN-KMA":
            from run_hybrid import gnn_KMA_solve_undirected
            fvs = gnn_KMA_solve_undirected(n, edges, pop_size, max_gens)
        else:
            fvs = ALGO_MAP[algo](n, edges)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        out_q.put(("OK", fvs, elapsed_ms))
    except Exception as ex:
        out_q.put(("ERROR", str(ex), None))


def run_algorithm_with_timeout(
    algo: str,
    n: int,
    edges: List[Tuple[int, int]],
    pop_size: int,
    max_gens: int,
    gnn_threshold: float,
    gnn_hidden: int | None,
    timeout_s: int,
) -> Tuple[Optional[List[int]], Optional[float], Optional[str]]:
    """
    Run one undirected algorithm in a child process with timeout.

    Returns:
      (fvs, elapsed_ms, error)
      - on success: error is None
      - on timeout: error == "TIMEOUT"
      - on failure: error starts with "ERROR:"
    """
    out_q: mp.Queue = mp.Queue()
    proc = mp.Process(
        target=_undirected_worker_run,
        args=(algo, n, edges, pop_size, max_gens, gnn_threshold, gnn_hidden, out_q),
    )
    proc.start()
    proc.join(timeout=timeout_s)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return None, None, "TIMEOUT"

    try:
        status, payload, elapsed = out_q.get_nowait()
    except queue.Empty:
        return None, None, "ERROR: worker returned no result"

    if status == "OK":
        return payload, elapsed, None
    return None, None, f"ERROR: {payload}"

def run_algorithm(
    algo: str,
    n: int,
    edges: List[Tuple[int, int]],
    pop_size: int = 50,
    max_gens: int = 200,
    gnn_threshold: float = 0.2,
    gnn_hidden: int | None = None,
) -> Tuple[List[int], float]:
    """
    Run a single algorithm and return (fvs, elapsed_ms).
    """
    start = time.perf_counter()

    if algo == "MA":
        fvs = cpp_engine.solve_undirected_MA(n, edges, pop_size, max_gens)
    elif algo == "KMA":
        if hasattr(cpp_engine, "solve_undirected_KMA"):
            fvs = cpp_engine.solve_undirected_KMA(n, edges, pop_size, max_gens)
        elif hasattr(cpp_engine, "solve_undirected_KME"):
            fvs = cpp_engine.solve_undirected_KME(n, edges, pop_size, max_gens)
        else:
            fvs = cpp_engine.solve_undirected_MA(n, edges, pop_size, max_gens)
    elif algo == "GNN-KMA":
        from run_hybrid import gnn_KMA_solve_undirected
        fvs = gnn_KMA_solve_undirected(
            n,
            edges,
            pop_size,
            max_gens,
            gnn_threshold=gnn_threshold,
            gnn_hidden_dim=gnn_hidden,
        )
    else:
        fn = ALGO_MAP[algo]
        fvs = fn(n, edges)

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return fvs, elapsed_ms


def run_on_file(
    filepath: str,
    algo: str,
    pop_size: int,
    max_gens: int,
    gnn_threshold: float = 0.2,
    gnn_hidden: int | None = None,
    results_dir: str = "results",
    verbose: bool = True,
) -> dict:
    """
    Parse a graph file, run the specified algorithm(s), and return results dict.
    If results_dir is provided, saves individual algorithm results to CSV files.
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

    algos_to_run = ["BST", "IC", "MA", "KMA", "GNN-KMA"] if algo == "ALL" else [algo]

    for alg in algos_to_run:
        # Check if this algorithm already processed this file
        csv_path = get_results_csv_path(results_dir, alg, is_directed=False)
        if is_result_already_recorded(csv_path, filename):
            if verbose:
                print(f"  {alg:4s} — [SKIPPED] Already stored result")
            # Load the existing result from CSV and map to summary keys.
            existing = load_existing_results(csv_path).get(filename, {})
            if existing:
                results[f"{alg}_size"] = existing.get("FVS_size", existing.get(f"{alg}_size", "N/A"))
                runtime_s = _to_float_or_none(existing.get("runtime"))
                if runtime_s is not None:
                    results[f"{alg}_ms"] = round(runtime_s * 1000.0, 2)
                else:
                    results[f"{alg}_ms"] = existing.get(f"{alg}_ms", existing.get("runtime", "N/A"))
                results[f"{alg}_valid"] = existing.get("validity", existing.get(f"{alg}_valid", "N/A"))
            continue
        
        timeout_s = get_dynamic_timeout_seconds(n)
        if verbose:
            print(f"  Running {alg:4s} (timeout={timeout_s}s) ... ", end="", flush=True)

        fvs, elapsed_ms, error = run_algorithm_with_timeout(
            alg, n, edges, pop_size, max_gens, gnn_threshold, gnn_hidden, timeout_s
        )

        # Build single-algorithm result row with unified schema.
        algo_result = {"file": filename, "n": n, "m": len(edges)}
        
        if error == "TIMEOUT":
            if verbose:
                print("TIMEOUT")
            algo_result["FVS_size"] = "TIMEOUT"
            algo_result["runtime"] = timeout_s
            algo_result["validity"] = "TIMEOUT"
            results[f"{alg}_size"] = "TIMEOUT"
            results[f"{alg}_ms"] = "TIMEOUT"
            results[f"{alg}_valid"] = "TIMEOUT"
        elif error is not None:
            if verbose:
                print(error)
            algo_result["FVS_size"] = "ERROR"
            algo_result["runtime"] = "ERROR"
            algo_result["validity"] = False
            results[f"{alg}_size"] = "ERROR"
            results[f"{alg}_ms"] = "ERROR"
            results[f"{alg}_valid"] = False
        else:
            valid = verify_fvs(n, edges, fvs)
            if verbose:
                status = "✓ VALID" if valid else "✗ INVALID"
                print(f"FVS size = {len(fvs):4d}  |  Time = {elapsed_ms:8.2f} ms  |  {status}")
            
            algo_result["FVS_size"] = len(fvs)
            algo_result["runtime"] = round(elapsed_ms / 1000.0, 6)
            algo_result["validity"] = valid
            
            results[f"{alg}_size"]  = len(fvs)
            results[f"{alg}_ms"]    = round(elapsed_ms, 2)
            results[f"{alg}_valid"] = valid
        
        # Immediately save this algorithm's result to its CSV
        append_result_to_csv(csv_path, algo_result)

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
        choices=["BST", "IC", "MA", "KMA", "GNN-KMA", "ALL"],
        help="Algorithm to run: BST (exact), IC (exact), MA (heuristic), KMA (kernelized MA), GNN-KMA (GNN+KMA), ALL (compare)"
    )
    parser.add_argument(
        "--test", required=True,
        help="Path to a single graph file OR a directory of graph files"
    )
    parser.add_argument(
        "--results-dir", default="results",
        help="Directory where per-algorithm CSV result files will be stored (default: results)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional: also save combined summary to this CSV file"
    )
    parser.add_argument(
        "--pop", type=int, default=50,
        help="[MA only] Population size (default: 50)"
    )
    parser.add_argument(
        "--gens", "--gen", type=int, default=200,
        help="[MA only] Maximum generations (default: 200)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-run output (only print summary / CSV)"
    )
    parser.add_argument(
        "--gnn-threshold", type=float, default=0.2,
        help="[GNN-KMA only] Probability threshold for GNN candidate selection (default: 0.2)"
    )
    parser.add_argument(
        "--gnn-hidden", type=int, default=None,
        help="[GNN-KMA only] Optional hidden dimension override for loading GNN weights"
    )

    args = parser.parse_args()

    if not (0.0 <= args.gnn_threshold <= 1.0):
        print("ERROR: --gnn-threshold must be between 0.0 and 1.0")
        sys.exit(1)
    if args.gnn_hidden is not None and args.gnn_hidden <= 0:
        print("ERROR: --gnn-hidden must be a positive integer")
        sys.exit(1)

    # ── Collect input files ──────────────────────────────────────────────────
    test_path = Path(args.test)
    if test_path.is_file():
        files = [str(test_path)]
    elif test_path.is_dir():
        files = collect_graph_files(test_path)

        if not files:
            print(f"No graph files found in {test_path}")
            print("Expected extensions: .txt, .gr, .edges, .graph, .dimacs, .mtx")
            sys.exit(1)
        print(f"Found {len(files)} graph file(s) in {test_path} (recursive)")
    else:
        print(f"ERROR: --test path does not exist: {args.test}")
        sys.exit(1)

    # ── Run benchmarks ───────────────────────────────────────────────────────
    all_results = []
    for filepath in files:
        result = run_on_file(
            filepath,
            args.algo,
            args.pop,
            args.gens,
            gnn_threshold=args.gnn_threshold,
            gnn_hidden=args.gnn_hidden,
            results_dir=args.results_dir,
            verbose=not args.quiet,
        )
        if result:
            all_results.append(result)

    # ── Print summary table ──────────────────────────────────────────────────
    if len(all_results) > 1 or args.quiet:
        print(f"\n{'═' * 80}")
        print(f"  SUMMARY  ({args.algo} on {len(all_results)} file(s))")
        print(f"{'═' * 80}")

        if args.algo == "ALL":
            algos_ran = ["BST", "IC", "MA", "KMA", "GNN-KMA"]
        else:
            algos_ran = [args.algo]
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