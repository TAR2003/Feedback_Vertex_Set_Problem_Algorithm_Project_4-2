#!/usr/bin/env python3
"""
benchmark_directed.py
=====================
Command-Line Interface for running and benchmarking DIRECTED FVS algorithms.

Usage examples
──────────────
# 1. Single algorithm on a single file
python experiments/benchmark_directed.py --algo BST --test data/raw_directed/pace001.gr

# 2. Directed Iterative Compression
python experiments/benchmark_directed.py --algo IC --test data/raw_directed/pace001.gr

# 3. Memetic Algorithm (heuristic, for large graphs)
python experiments/benchmark_directed.py --algo MA --test data/raw_directed/pace001.gr --pop 80 --gens 300

# 4. Compare ALL algorithms on one file
python experiments/benchmark_directed.py --algo ALL --test data/raw_directed/pace001.gr

# 5. Batch run on a folder, save CSV
python experiments/benchmark_directed.py --algo MA --test data/raw_directed/ --output directed_results.csv

# 6. Batch ALL algorithms on every file in a folder
python experiments/benchmark_directed.py --algo ALL --test data/raw_directed/ --output directed_comparison.csv

# 7. HYBRID (GNN-guided KME) on one file with custom parameters
python experiments/benchmark_directed.py --algo HYBRID --test data/raw_directed/pace001.gr --pop 100 --gens 400

# 8. HYBRID on a folder of graphs
python experiments/benchmark_directed.py --algo HYBRID --test data/raw_directed/ --output directed_hybrid_results.csv

Supported --algo values
───────────────────────
  BST    — Directed Bounded Search Tree (exact, uses SCC decomposition)
  IC     — Directed Iterative Compression (exact, greedy + compression)
  MA     — Directed Memetic Algorithm (heuristic, scales to large graphs)
    KME    — Directed Kernelized Memetic Algorithm (kernelization + MA)
    HYBRID — GNN-guided KME (combines GNN inference + kernelized MA refinement)
    ALL    — Run BST, IC, MA, KME, and HYBRID; print comparison table

Directed Graph File Format (PACE 2022 .gr)
──────────────────────────────────────────
Lines starting with 'c' or '#': comments
'p dfvs N M' or 'p fvs N M': header (N vertices, M edges)
Each edge line: "u v" (directed edge u → v, 1-indexed in PACE format)
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
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
# Try platform-specific build directories first, then legacy build/
for candidate in ("build-linux", "build-macos", "build-win", "build"):
    sys.path.insert(0, str(PROJECT_ROOT / "cpp_engine" / candidate))
# Try experiments first (where the .so file is compiled) - insert last so it's first in path
sys.path.insert(0, str(SCRIPT_DIR))

try:
    import cpp_engine
except ImportError as e:
    print("ERROR: Cannot import cpp_engine.")
    print(f"  Script dir: {SCRIPT_DIR}")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  sys.path[0]: {sys.path[0]}")
    print(f"  sys.path[1]: {sys.path[1]}")
    print("  Did you compile it? Use one of these from project root:")
    print("    Linux/macOS:  python3 build_engine.py")
    print("    Linux helper: ./build_cpp.sh")
    print("    Windows:      python build_engine.py  or  .\\build_cpp.ps1")
    print(f"       (Original error: {e})")
    sys.exit(1)

# Try importing hybrid solver from run_hybrid.py (graceful fallback)
# Note: Import is deferred until hybrid algorithm is actually requested to avoid slow startup
HAS_HYBRID = True  # Assume available; will fail gracefully at runtime if not


# ═══════════════════════════════════════════════════════════════════════════════
#  File Parsing  (supports PACE .gr and generic edge-list formats)
# ═══════════════════════════════════════════════════════════════════════════════

def parse_metis_directed(filepath: str) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Parse METIS format directed graph.
    
    Format:
      Line 1: n m t  (n vertices, m edges, t weight type)
      Lines 2..n+1: adjacency list for each vertex (1-indexed vertices)
    
    Returns:
        (n, edges)  where edges are 0-indexed directed pairs (u, v).
    """
    edges: List[Tuple[int, int]] = []
    n = None
    
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            
            parts = line.split()
            if not parts:
                continue
            
            # First non-comment line: n m t
            if n is None:
                try:
                    n = int(parts[0])
                    # m = int(parts[1])
                    # t = int(parts[2])
                    continue
                except (ValueError, IndexError):
                    raise ValueError(f"Invalid METIS header at line {line_num}: {line}")
            
            # Adjacency list for vertex (line_num - 1) in METIS (1-indexed)
            # We need to map: line 2 -> vertex 1, line 3 -> vertex 2, etc.
            # But skip non-numeric/comment lines, so we track actual vertex index
            try:
                neighbors = [int(x) for x in parts]
                # This line corresponds to adjacency for vertex (current line in data)
                # In METIS, vertices are 1-indexed, so we convert to 0-indexed
                u = line_num - 2  # line_num=2 -> u=0 (first vertex), line_num=3 -> u=1, etc.
                
                for v in neighbors:
                    # v is 1-indexed from file, convert to 0-indexed
                    edges.append((u, v - 1))
            except ValueError:
                raise ValueError(f"Invalid vertex list at line {line_num}: {line}")
    
    if n is None:
        raise ValueError(f"No valid METIS header found in {filepath}")
    
    return n, edges


def parse_directed_graph_file(filepath: str) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Parse a DIRECTED graph file.

    Supported formats:
      1. METIS format (PACE 2022):
        1024 2103 0           <- 1024 vertices, 2103 edges
        346 649               <- adjacency list for vertex 1 (1-indexed)
        371                   <- adjacency list for vertex 2
        ...
      
      2. PACE .gr / Edge-list format:
        c comment
        p dfvs 10 20          <- 10 vertices, 20 directed edges
        1 2                   <- directed edge 1 → 2 (1-indexed)

      3. Generic edge-list:
        # comment
        u v                   <- directed edge u → v
        u v w                 <- edge with weight (weight ignored)

    Returns:
        (n, edges)  where edges are 0-indexed directed pairs (u, v).
    """
    # First, try to detect format by reading first non-comment line
    first_line = None
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith(('c', '#', '%')):
                first_line = line
                break
    
    if not first_line:
        raise ValueError(f"No data found in {filepath}")
    
    parts = first_line.split()
    
    # Try to detect METIS format: first line is "n m t" (3 integers)
    is_metis = False
    if len(parts) == 3:
        try:
            n_candidate = int(parts[0])
            m_candidate = int(parts[1])
            t_candidate = int(parts[2])
            # METIS has n > 0, m > 0, t is usually 0
            # If first line looks like METIS header, try METIS parsing
            if n_candidate > 0 and m_candidate > 0 and 0 <= t_candidate <= 1:
                is_metis = True
        except ValueError:
            pass
    
    if is_metis:
        return parse_metis_directed(filepath)
    
    # Otherwise parse as edge-list
    edges: List[Tuple[int, int]] = []
    n_hint: Optional[int] = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # PACE / DIMACS comment
            if line.startswith('c ') or line.startswith('#') or line.startswith('%'):
                continue

            parts = line.split()

            # PACE header: "p dfvs N M" or "p fvs N M" or "p edge N M"
            if parts[0].lower() == 'p' and len(parts) >= 3:
                try:
                    n_hint = int(parts[2])
                except ValueError:
                    pass
                continue

            # Skip other alphabetic lines
            if not parts[0].lstrip('-').isdigit():
                continue

            try:
                u = int(parts[0])
                v = int(parts[1])
                edges.append((u, v))
            except (ValueError, IndexError):
                continue

    if not edges:
        raise ValueError(f"No directed edges found in {filepath}")

    all_verts = set()
    for u, v in edges:
        all_verts.add(u)
        all_verts.add(v)

    min_v = min(all_verts)
    max_v = max(all_verts)

    # Normalize: 1-indexed → 0-indexed
    if min_v == 1:
        edges = [(u - 1, v - 1) for u, v in edges]
        max_v -= 1

    n = n_hint if n_hint is not None else max_v + 1
    n = max(n, max_v + 1)
    return n, edges


def verify_dfvs(n: int, edges: List[Tuple[int, int]], fvs: List[int]) -> bool:
    """
    Verify that `fvs` is a valid Directed Feedback Vertex Set.
    Removes FVS vertices from the graph, then checks if the result is a DAG.
    Uses DFS with 3-coloring (WHITE=0, GRAY=1, BLACK=2).
    """
    fvs_set = set(fvs)
    # Build adjacency list excluding FVS vertices
    out_adj: dict = {v: [] for v in range(n) if v not in fvs_set}
    for u, v in edges:
        if u not in fvs_set and v not in fvs_set:
            out_adj[u].append(v)

    color = [0] * n

    def dfs(u: int) -> bool:
        color[u] = 1  # GRAY
        for nb in out_adj.get(u, []):
            if color[nb] == 1:
                return True  # back edge → cycle
            if color[nb] == 0 and dfs(nb):
                return True
        color[u] = 2  # BLACK
        return False

    for v in range(n):
        if v not in fvs_set and color[v] == 0:
            if dfs(v):
                return False
    return True


# ═══════════════════════════════════════════════════════════════════════════════
#  Algorithm Runners
# ═══════════════════════════════════════════════════════════════════════════════

ALGO_MAP_D = {
    "BST": cpp_engine.solve_directed_BST,
    "IC":  cpp_engine.solve_directed_IC,
    "MA":  cpp_engine.solve_directed_MA,
    "KME": getattr(cpp_engine, "solve_directed_KME", cpp_engine.solve_directed_MA),
}


def get_dynamic_timeout_seconds(n: int) -> int:
    """
    Dynamic timeout policy based on graph size.
      n <= 50         -> 60s
      50 < n <= 200   -> 300s
      n > 200         -> 600s
    """
    if n <= 50:
        return 60
    if n <= 200:
        return 300
    return 600


def _directed_worker_run(algo: str, n: int, edges: List[Tuple[int, int]],
                         pop_size: int, max_gens: int, out_q: mp.Queue) -> None:
    """Child-process worker that runs one algorithm and returns via queue."""
    try:
        start = time.perf_counter()
        if algo == "MA":
            fvs = cpp_engine.solve_directed_MA(n, edges, pop_size, max_gens)
        elif algo == "KME":
            if hasattr(cpp_engine, "solve_directed_KME"):
                fvs = cpp_engine.solve_directed_KME(n, edges, pop_size, max_gens)
            else:
                fvs = cpp_engine.solve_directed_MA(n, edges, pop_size, max_gens)
        elif algo == "HYBRID":
            from run_hybrid import hybrid_solve_directed
            fvs = hybrid_solve_directed(n, edges, pop_size, max_gens)
        else:
            fvs = ALGO_MAP_D[algo](n, edges)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        out_q.put(("OK", fvs, elapsed_ms))
    except Exception as ex:
        out_q.put(("ERROR", str(ex), None))


def run_directed_algorithm_with_timeout(
    algo: str,
    n: int,
    edges: List[Tuple[int, int]],
    pop_size: int,
    max_gens: int,
    timeout_s: int,
) -> Tuple[Optional[List[int]], Optional[float], Optional[str]]:
    """
    Run one directed algorithm in a child process with timeout.

    Returns:
      (fvs, elapsed_ms, error)
      - on success: error is None
      - on timeout: error == "TIMEOUT"
      - on failure: error starts with "ERROR:"
    """
    out_q: mp.Queue = mp.Queue()
    proc = mp.Process(
        target=_directed_worker_run,
        args=(algo, n, edges, pop_size, max_gens, out_q),
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

def run_directed_algorithm(algo: str, n: int, edges: List[Tuple[int, int]],
                            pop_size: int = 50, max_gens: int = 200
                            ) -> Tuple[List[int], float]:
    """Run one directed algorithm. Returns (fvs, elapsed_ms)."""
    start = time.perf_counter()

    if algo == "MA":
        fvs = cpp_engine.solve_directed_MA(n, edges, pop_size, max_gens)
    elif algo == "KME":
        if hasattr(cpp_engine, "solve_directed_KME"):
            fvs = cpp_engine.solve_directed_KME(n, edges, pop_size, max_gens)
        else:
            fvs = cpp_engine.solve_directed_MA(n, edges, pop_size, max_gens)
    elif algo == "HYBRID":
        from run_hybrid import hybrid_solve_directed
        fvs = hybrid_solve_directed(n, edges, pop_size, max_gens)
    else:
        fvs = ALGO_MAP_D[algo](n, edges)

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return fvs, elapsed_ms


def run_on_file(filepath: str, algo: str, pop_size: int, max_gens: int,
                verbose: bool = True) -> dict:
    """
    Parse a directed graph file, run the specified algorithm(s), print and
    return a results dictionary.
    """
    try:
        n, edges = parse_directed_graph_file(filepath)
    except Exception as ex:
        print(f"  [SKIP] Could not parse {filepath}: {ex}")
        return {}

    filename = Path(filepath).name
    results = {"file": filename, "n": n, "m": len(edges)}

    if verbose:
        print(f"\n{'─' * 60}")
        print(f"  File : {filename}")
        print(f"  Graph: {n} vertices, {len(edges)} directed edges")
        print(f"{'─' * 60}")

    algos_to_run = ["BST", "IC", "MA", "KME", "HYBRID"] if algo == "ALL" else [algo]

    for alg in algos_to_run:
        timeout_s = get_dynamic_timeout_seconds(n)
        if verbose:
            print(f"  Running {alg:4s} (timeout={timeout_s}s) ... ", end="", flush=True)

        fvs, elapsed_ms, error = run_directed_algorithm_with_timeout(
            alg, n, edges, pop_size, max_gens, timeout_s
        )

        if error == "TIMEOUT":
            if verbose:
                print("TIMEOUT")
            results[f"{alg}_size"] = "TIMEOUT"
            results[f"{alg}_ms"] = "TIMEOUT"
            results[f"{alg}_valid"] = "TIMEOUT"
            continue

        if error is not None:
            if verbose:
                print(error)
            results[f"{alg}_size"] = "ERROR"
            results[f"{alg}_ms"] = "ERROR"
            results[f"{alg}_valid"] = False
            continue

        valid = verify_dfvs(n, edges, fvs)

        if verbose:
            status = "✓ VALID" if valid else "✗ INVALID"
            print(f"DFVS size = {len(fvs):4d}  |  Time = {elapsed_ms:8.2f} ms  |  {status}")

        results[f"{alg}_size"]  = len(fvs)
        results[f"{alg}_ms"]    = round(elapsed_ms, 2)
        results[f"{alg}_valid"] = valid

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="FVS Directed Benchmark CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--algo", required=True,
        choices=["BST", "IC", "MA", "KME", "HYBRID", "ALL"],
        help="Algorithm: BST (exact), IC (exact), MA (heuristic), KME (kernelized MA), HYBRID (GNN+KME), ALL (compare)"
    )
    parser.add_argument(
        "--test", required=True,
        help="Path to a single .gr/.txt file OR a directory of graph files"
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional: save results to this CSV file path"
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
        help="Suppress per-run verbose output"
    )

    args = parser.parse_args()

    # ── Collect input files ──────────────────────────────────────────────────
    test_path = Path(args.test)
    if test_path.is_file():
        files = [str(test_path)]
    elif test_path.is_dir():
        # Include files with these extensions
        extensions = (".gr", ".txt", ".edges", ".graph", ".dimacs", ".mtx")
        files = []
        
        for f in sorted(test_path.iterdir()):
            if not f.is_file():
                continue
            # Include files with known extensions
            if f.suffix.lower() in extensions:
                files.append(str(f))
            # Also include files without extension (e.g., PACE h_001, h_002, ...)
            # but only if they look like graph files (skip obvious non-graph files)
            elif f.suffix == "" and not f.name.startswith('.'):
                files.append(str(f))
        
        if not files:
            print(f"No graph files found in {test_path}")
            sys.exit(1)
        print(f"Found {len(files)} graph file(s) in {test_path}")
    else:
        print(f"ERROR: --test path does not exist: {args.test}")
        sys.exit(1)

    # ── Run benchmarks ───────────────────────────────────────────────────────
    all_results = []
    for filepath in files:
        result = run_on_file(filepath, args.algo, args.pop, args.gens,
                             verbose=not args.quiet)
        if result:
            all_results.append(result)

    # ── Print summary ────────────────────────────────────────────────────────
    if len(all_results) > 1 or args.quiet:
        print(f"\n{'═' * 80}")
        print(f"  SUMMARY  (DIRECTED {args.algo} on {len(all_results)} file(s))")
        print(f"{'═' * 80}")

        if args.algo == "ALL":
            algos_ran = ["BST", "IC", "MA", "KME", "HYBRID"]
        else:
            algos_ran = [args.algo]
        header = f"  {'File':<28} {'n':>6} {'m':>8}"
        for alg in algos_ran:
            header += f"  {alg+' size':>10} {alg+' ms':>10}"
        print(header)
        print("  " + "─" * (len(header) - 2))

        for r in all_results:
            row = f"  {r['file']:<28} {r['n']:>6} {r['m']:>8}"
            for alg in algos_ran:
                sz = r.get(f"{alg}_size", "N/A")
                ms = r.get(f"{alg}_ms",   "N/A")
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