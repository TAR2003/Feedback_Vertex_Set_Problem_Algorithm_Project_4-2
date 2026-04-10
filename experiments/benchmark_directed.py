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

# 7. GNN-KMA (GNN-guided KMA) on one file with custom parameters
python experiments/benchmark_directed.py --algo GNN-KMA --test data/raw_directed/pace001.gr --pop 100 --gens 400 --gnn-threshold 0.2 --gnn-hidden 64

# 8. GNN-KMA on a folder of graphs
python experiments/benchmark_directed.py --algo GNN-KMA --test data/raw_directed/ --output directed_GNN-KMA_results.csv

Supported --algo values
───────────────────────
  BST    — Directed Bounded Search Tree (exact, uses SCC decomposition)
  IC     — Directed Iterative Compression (exact, greedy + compression)
  MA     — Directed Memetic Algorithm (heuristic, scales to large graphs)
    KMA    — Directed Kernelized Memetic Algorithm (kernelization + MA)
    GNN-KMA — GNN-guided KMA (combines GNN inference + kernelized MA refinement)
    ALL    — Run BST, IC, MA, KMA, and GNN-KMA; print comparison table

Directed Graph File Format
──────────────────────────
Supports both PACE .gr and universal TXT edge-list:
    # format: edge_list_v1
    p edge N M
    u v
PACE data remains supported as-is:
    p dfvs N M + 1-indexed edge lines.
"""

import argparse
import os
import sys
sys.setrecursionlimit(20000)
import time
import csv
import multiprocessing as mp
import queue
from pathlib import Path
from typing import List, Tuple, Optional

# ── Add cpp_engine to path ────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
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

# Try importing GNN-KMA solver from run_hybrid.py (graceful fallback)
# Note: Import is deferred until GNN-KMA algorithm is actually requested to avoid slow startup
HAS_GNN_KMA = True  # Assume available; will fail gracefully at runtime if not


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
#  CSV Result Tracking
# ═══════════════════════════════════════════════════════════════════════════════

def get_results_csv_path(
    results_dir: str,
    algo: str,
    is_directed: bool = False,
    result_tag: Optional[str] = None,
) -> Path:
    """
    Get the CSV file path for a specific algorithm.
    Adds algorithm type suffix: _exact for BST/IC, _heuristic for MA/KMA/GNN-KMA/GNN-KMA-2
    E.g., results/directed_BST_exact.csv or results/directed_KMA_heuristic.csv
    """
    prefix = "directed" if is_directed else "undirected"
    
    # Prefer explicit result tag (dataset track) when provided by pipeline.
    if result_tag in {"exact", "heuristic"}:
        algo_type = result_tag
    else:
        # Fallback to algorithm family for backwards compatibility.
        if algo in ["BST", "IC"]:
            algo_type = "exact"
        elif algo in ["MA", "KMA", "DKMA", "GNN-KMA", "GNN-KMA-2", "GNN-KMA-3", "GNN-DKMA"]:
            algo_type = "heuristic"
        else:
            algo_type = "unknown"
    
    csv_name = f"{prefix}_{algo}_{algo_type}.csv"
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


def _describe_runtime_backend(gnn_device: str) -> str:
    """Human-readable runtime backend for per-test logging."""
    try:
        import torch  # type: ignore
    except ImportError:
        return "CPU (PyTorch unavailable)"

    pref = (gnn_device or "auto").lower()
    if pref == "cpu":
        return "CPU (forced by --gnn-device=cpu)"
    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "unknown-gpu"
        return f"CUDA ({name})"
    if pref == "cuda":
        return "CPU (CUDA requested but unavailable)"
    return "CPU (no CUDA detected)"


def _validate_cuda_requested_or_exit(gnn_device: str) -> None:
    """Fail fast when user explicitly requests CUDA but runtime cannot provide it."""
    if (gnn_device or "auto").lower() != "cuda":
        return
    try:
        import torch  # type: ignore
    except ImportError:
        print("ERROR: --gnn-device=cuda requested, but PyTorch is not installed.")
        print("       Install CUDA-enabled PyTorch or use --gnn-device cpu.")
        sys.exit(1)

    if not torch.cuda.is_available():
        build = getattr(torch.version, "cuda", None)
        build_msg = f"torch.version.cuda={build}" if build is not None else "CPU-only PyTorch build"
        print("ERROR: --gnn-device=cuda requested, but CUDA is unavailable in current Python runtime.")
        print(f"       Details: {build_msg}, device_count={torch.cuda.device_count()}")
        print("       This indicates wrong interpreter or CPU-only torch wheel.")
        print("       Use a CUDA-enabled Python environment, then rerun the benchmark.")
        sys.exit(1)


def _torch_runtime_diagnostics() -> str:
    """Short diagnostics for interpreter + torch runtime backend."""
    py_exec = sys.executable
    try:
        import torch  # type: ignore
    except ImportError:
        return f"python={py_exec} | torch=NOT_INSTALLED"

    tver = getattr(torch, "__version__", "unknown")
    cver = getattr(torch.version, "cuda", None)
    avail = torch.cuda.is_available()
    dcnt = torch.cuda.device_count()
    gpu = "none"
    if avail and dcnt > 0:
        try:
            gpu = torch.cuda.get_device_name(0)
        except Exception:
            gpu = "unknown-gpu"
    return (
        f"python={py_exec} | torch={tver} | torch_cuda={cver} "
        f"| cuda_available={avail} | device_count={dcnt} | gpu={gpu}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Algorithm Runners
# ═══════════════════════════════════════════════════════════════════════════════

ALGO_MAP_D = {
    "BST": cpp_engine.solve_directed_BST,
    "IC":  cpp_engine.solve_directed_IC,
    "MA":  cpp_engine.solve_directed_MA,
    "KMA": getattr(cpp_engine, "solve_directed_KMA", getattr(cpp_engine, "solve_directed_KMA", cpp_engine.solve_directed_MA)),
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


def _directed_worker_run(algo: str, n: int, edges: List[Tuple[int, int]],
                         pop_size: int, max_gens: int,
                         gnn_threshold: float, gnn_hidden: Optional[int],
                         timeout_seconds: int,
                         gnn_timeout: int,
                         early_stop: int,
                         gnn_device: str,
                         feature_device: str,
                         gnn_batch_size: int,
                         commit_threshold: float,
                         dynkern_every: int,
                         gain_search: bool,
                         diversify: bool,
                         out_q: mp.Queue) -> None:
    """Child-process worker that runs one algorithm and returns via queue."""
    try:
        start = time.perf_counter()
        if algo == "MA":
            fvs = cpp_engine.solve_directed_MA(n, edges, pop_size, max_gens, early_stop, timeout_seconds)
        elif algo == "KMA":
            from run_hybrid import kma_solve_directed, set_runtime_gnn_options
            set_runtime_gnn_options(gnn_device=gnn_device, feature_device=feature_device, gnn_batch_size=gnn_batch_size)
            fvs = kma_solve_directed(
                n,
                edges,
                pop_size,
                max_gens,
                max_time_seconds=timeout_seconds,
                early_stop=early_stop,
            )
        elif algo == "GNN-KMA":
            from run_hybrid import gnn_KMA_solve_directed, set_runtime_gnn_options
            set_runtime_gnn_options(gnn_device=gnn_device, feature_device=feature_device, gnn_batch_size=gnn_batch_size)
            fvs = gnn_KMA_solve_directed(
                n,
                edges,
                pop_size,
                max_gens,
                gnn_threshold=gnn_threshold,
                gnn_hidden_dim=gnn_hidden,
                gnn_timeout=gnn_timeout,
                max_time_seconds=timeout_seconds,
                early_stop=early_stop,
            )
        elif algo == "GNN-KMA-2":
            from run_hybrid import gnn_KMA2_solve_directed, set_runtime_gnn_options
            set_runtime_gnn_options(gnn_device=gnn_device, feature_device=feature_device, gnn_batch_size=gnn_batch_size)
            fvs = gnn_KMA2_solve_directed(
                n,
                edges,
                pop_size,
                max_gens,
                gnn_threshold=gnn_threshold,
                gnn_hidden_dim=gnn_hidden,
                gnn_timeout=gnn_timeout,
                max_time_seconds=timeout_seconds,
                early_stop=early_stop,
            )
        elif algo == "GNN-KMA-3":
            from run_hybrid import gnn_KMA3_solve_directed, set_runtime_gnn_options
            set_runtime_gnn_options(gnn_device=gnn_device, feature_device=feature_device, gnn_batch_size=gnn_batch_size)
            fvs = gnn_KMA3_solve_directed(
                n,
                edges,
                pop_size,
                max_gens,
                gnn_threshold=gnn_threshold,
                gnn_hidden_dim=gnn_hidden,
                gnn_timeout=gnn_timeout,
                max_time_seconds=timeout_seconds,
                early_stop=early_stop,
            )
        elif algo == "DKMA":
            from run_hybrid import dkma_solve_directed, set_runtime_gnn_options
            set_runtime_gnn_options(gnn_device=gnn_device, feature_device=feature_device, gnn_batch_size=gnn_batch_size)
            fvs = dkma_solve_directed(
                n,
                edges,
                pop_size,
                max_gens,
                early_stop=early_stop,
                max_time_seconds=timeout_seconds,
                commit_threshold=commit_threshold,
                dynkern_every=dynkern_every,
                gain_search=gain_search,
                diversify=diversify,
            )
        elif algo == "GNN-DKMA":
            from run_hybrid import gnn_dkma_solve_directed, set_runtime_gnn_options
            set_runtime_gnn_options(gnn_device=gnn_device, feature_device=feature_device, gnn_batch_size=gnn_batch_size)
            fvs = gnn_dkma_solve_directed(
                n,
                edges,
                gnn_version="v1",
                gnn_threshold=gnn_threshold,
                gnn_hidden=gnn_hidden,
                gnn_timeout=gnn_timeout,
                pop_size=pop_size,
                max_gens=max_gens,
                early_stop=early_stop,
                max_time_seconds=timeout_seconds,
                commit_threshold=commit_threshold,
                dynkern_every=dynkern_every,
                gain_search=gain_search,
                diversify=diversify,
            )
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
    gnn_threshold: float,
    gnn_hidden: Optional[int],
    timeout_seconds: int,
    timeout_s: int,
    gnn_timeout: int,
    early_stop: int,
    gnn_device: str,
    feature_device: str,
    gnn_batch_size: int,
    commit_threshold: float,
    dynkern_every: int,
    gain_search: bool,
    diversify: bool,
) -> Tuple[Optional[List[int]], Optional[float], Optional[dict], Optional[str]]:
    """
    Run one directed algorithm in a child process with timeout.

    Returns:
    (fvs, elapsed_ms, stage_metrics, error)
      - on success: error is None
      - on timeout: error == "TIMEOUT"
      - on failure: error starts with "ERROR:"
    """
    # Heuristic solvers return best-so-far at their own timeout boundary.
    # Keep them in-process so we can always receive that solution instead of
    # parent-level TIMEOUT without a candidate set.
    if algo in {"MA", "KMA", "DKMA", "GNN-KMA", "GNN-KMA-2", "GNN-KMA-3", "GNN-DKMA"}:
        try:
            fvs, elapsed_ms, stage_metrics = run_directed_algorithm(
                algo,
                n,
                edges,
                pop_size,
                max_gens,
                gnn_threshold,
                gnn_hidden,
                timeout_seconds,
                gnn_timeout,
                early_stop,
                gnn_device,
                feature_device,
                gnn_batch_size,
                commit_threshold,
                dynkern_every,
                gain_search,
                diversify,
            )
            return fvs, elapsed_ms, stage_metrics, None
        except Exception as ex:
            return None, None, None, f"ERROR: {ex}"

    out_q: mp.Queue = mp.Queue()
    proc = mp.Process(
        target=_directed_worker_run,
        args=(
            algo,
            n,
            edges,
            pop_size,
            max_gens,
            gnn_threshold,
            gnn_hidden,
            timeout_seconds,
            gnn_timeout,
            early_stop,
            gnn_device,
            feature_device,
            gnn_batch_size,
            commit_threshold,
            dynkern_every,
            gain_search,
            diversify,
            out_q,
        ),
    )
    proc.start()
    proc.join(timeout=timeout_s)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=1.0)
        if proc.is_alive() and hasattr(proc, "kill"):
            proc.kill()
            proc.join(timeout=1.0)
        return None, None, None, "TIMEOUT"

    try:
        status, payload, elapsed = out_q.get_nowait()
    except queue.Empty:
        return None, None, None, "ERROR: worker returned no result"

    if status == "OK":
        return payload, elapsed, None, None
    return None, None, None, f"ERROR: {payload}"

def run_directed_algorithm(algo: str, n: int, edges: List[Tuple[int, int]],
                            pop_size: int = 20, max_gens: int = 100,
                            gnn_threshold: float = 0.2,
                            gnn_hidden: Optional[int] = None,
                            timeout_seconds: int = 600,
                            gnn_timeout: int = 60,
                            early_stop: int = 20,
                            gnn_device: str = "auto",
                            feature_device: str = "auto",
                            gnn_batch_size: int = 2048,
                            commit_threshold: float = 0.6,
                            dynkern_every: int = 5,
                            gain_search: bool = True,
                            diversify: bool = True,
                            ) -> Tuple[List[int], float, dict]:
    """Run one directed algorithm. Returns (fvs, elapsed_ms)."""
    start = time.perf_counter()

    stage_metrics = {
        "kernelization_ms": 0.0,
        "gnn_candidate_ms": 0.0,
        "ma_ms": 0.0,
        "initial_kernel_size": n,
        "final_kernel_size": n,
        "n_dynamic_reductions": 0,
    }

    if algo in {"KMA", "DKMA", "GNN-KMA", "GNN-KMA-2", "GNN-KMA-3", "GNN-DKMA"}:
        try:
            from run_hybrid import set_runtime_gnn_options
            set_runtime_gnn_options(
                gnn_device=gnn_device,
                feature_device=feature_device,
                gnn_batch_size=gnn_batch_size,
            )
        except Exception:
            pass

    if algo == "MA":
        fvs = cpp_engine.solve_directed_MA(n, edges, pop_size, max_gens, early_stop, timeout_seconds)
    elif algo == "KMA":
        from run_hybrid import kma_solve_directed
        fvs, stage_metrics = kma_solve_directed(
            n,
            edges,
            pop_size,
            max_gens,
            max_time_seconds=timeout_seconds,
            early_stop=early_stop,
            return_diagnostics=True,
        )
    elif algo == "GNN-KMA":
        from run_hybrid import gnn_KMA_solve_directed
        fvs, stage_metrics = gnn_KMA_solve_directed(
            n,
            edges,
            pop_size,
            max_gens,
            gnn_threshold=gnn_threshold,
            gnn_hidden_dim=gnn_hidden,
            gnn_timeout=gnn_timeout,
            max_time_seconds=timeout_seconds,
            early_stop=early_stop,
            return_diagnostics=True,
        )
    elif algo == "GNN-KMA-2":
        from run_hybrid import gnn_KMA2_solve_directed
        fvs, stage_metrics = gnn_KMA2_solve_directed(
            n,
            edges,
            pop_size,
            max_gens,
            gnn_threshold=gnn_threshold,
            gnn_hidden_dim=gnn_hidden,
            gnn_timeout=gnn_timeout,
            max_time_seconds=timeout_seconds,
            early_stop=early_stop,
            return_diagnostics=True,
        )
    elif algo == "GNN-KMA-3":
        from run_hybrid import gnn_KMA3_solve_directed
        fvs, stage_metrics = gnn_KMA3_solve_directed(
            n,
            edges,
            pop_size,
            max_gens,
            gnn_threshold=gnn_threshold,
            gnn_hidden_dim=gnn_hidden,
            gnn_timeout=gnn_timeout,
            max_time_seconds=timeout_seconds,
            early_stop=early_stop,
            return_diagnostics=True,
        )
    elif algo == "DKMA":
        from run_hybrid import dkma_solve_directed
        fvs, stage_metrics = dkma_solve_directed(
            n,
            edges,
            pop_size=pop_size,
            max_gens=max_gens,
            early_stop=early_stop,
            max_time_seconds=timeout_seconds,
            commit_threshold=commit_threshold,
            dynkern_every=dynkern_every,
            gain_search=gain_search,
            diversify=diversify,
            return_diagnostics=True,
        )
    elif algo == "GNN-DKMA":
        from run_hybrid import gnn_dkma_solve_directed
        fvs = gnn_dkma_solve_directed(
            n,
            edges,
            gnn_version="v1",
            gnn_threshold=gnn_threshold,
            gnn_hidden=gnn_hidden,
            gnn_timeout=gnn_timeout,
            pop_size=pop_size,
            max_gens=max_gens,
            early_stop=early_stop,
            max_time_seconds=timeout_seconds,
            commit_threshold=commit_threshold,
            dynkern_every=dynkern_every,
            gain_search=gain_search,
            diversify=diversify,
        )
    else:
        fvs = ALGO_MAP_D[algo](n, edges)

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if algo == "MA":
        stage_metrics["ma_ms"] = elapsed_ms
    return fvs, elapsed_ms, stage_metrics


def run_on_file(filepath: str, algo: str, pop_size: int, max_gens: int,
                gnn_threshold: float = 0.2, gnn_hidden: Optional[int] = None,
                timeout_seconds: int = 600,
                gnn_timeout: int = 60,
                early_stop: int = 20,
                gnn_device: str = "auto",
                feature_device: str = "auto",
                gnn_batch_size: int = 2048,
                commit_threshold: float = 0.6,
                dynkern_every: int = 5,
                gain_search: bool = True,
                diversify: bool = True,
                results_dir: str = "results", result_tag: Optional[str] = None,
                verbose: bool = True) -> dict:
    """
    Parse a directed graph file, run the specified algorithm(s), print and
    return a results dictionary.
    If results_dir is provided, saves individual algorithm results to CSV files.
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
        backend = _describe_runtime_backend(gnn_device)
        print(
            f"  Runtime backend: {backend} | gnn_device={gnn_device} "
            f"| feature_device={feature_device} | gnn_batch_size={gnn_batch_size}"
        )
        print(f"{'─' * 60}")

    if algo == "ALL":
        algos_to_run = ["BST", "IC", "MA", "KMA", "DKMA", "GNN-KMA", "GNN-KMA-2", "GNN-KMA-3", "GNN-DKMA"]
    elif algo == "PUREALGO":
        algos_to_run = ["BST", "IC", "MA", "KMA", "DKMA"]
    else:
        algos_to_run = [algo]

    for alg in algos_to_run:
        # Check if this algorithm already processed this file
        csv_path = get_results_csv_path(
            results_dir,
            alg,
            is_directed=True,
            result_tag=result_tag,
        )
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
        
        timeout_s = timeout_seconds
        if verbose:
            print(f"  Running {alg:4s} (timeout={timeout_s}s) ... ", end="", flush=True)

        fvs, elapsed_ms, stage_metrics, error = run_directed_algorithm_with_timeout(
            alg,
            n,
            edges,
            pop_size,
            max_gens,
            gnn_threshold,
            gnn_hidden,
            timeout_seconds,
            timeout_s,
            gnn_timeout,
            early_stop,
            gnn_device,
            feature_device,
            gnn_batch_size,
            commit_threshold,
            dynkern_every,
            gain_search,
            diversify,
        )

        # Build single-algorithm result row with unified schema.
        algo_result = {"file": filename, "n": n, "m": len(edges)}
        
        if error == "TIMEOUT":
            if verbose:
                print("TIMEOUT")
            algo_result["FVS_size"] = "TIMEOUT"
            algo_result["runtime"] = timeout_s
            algo_result["validity"] = "TIMEOUT"
            algo_result["kernelization_time"] = "TIMEOUT"
            algo_result["gnn_candidate_time"] = "TIMEOUT"
            algo_result["initial_kernel_size"] = "TIMEOUT"
            algo_result["final_kernel_size"] = "TIMEOUT"
            algo_result["n_dynamic_reductions"] = "TIMEOUT"
            algo_result["solution_size"] = "TIMEOUT"
            algo_result["time_seconds"] = timeout_s
            results[f"{alg}_size"] = "TIMEOUT"
            results[f"{alg}_ms"] = "TIMEOUT"
            results[f"{alg}_valid"] = "TIMEOUT"
        elif error is not None:
            if verbose:
                print(error)
            algo_result["FVS_size"] = "ERROR"
            algo_result["runtime"] = "ERROR"
            algo_result["validity"] = False
            algo_result["kernelization_time"] = "ERROR"
            algo_result["gnn_candidate_time"] = "ERROR"
            algo_result["initial_kernel_size"] = "ERROR"
            algo_result["final_kernel_size"] = "ERROR"
            algo_result["n_dynamic_reductions"] = "ERROR"
            algo_result["solution_size"] = "ERROR"
            algo_result["time_seconds"] = "ERROR"
            results[f"{alg}_size"] = "ERROR"
            results[f"{alg}_ms"] = "ERROR"
            results[f"{alg}_valid"] = False
        else:
            valid = verify_dfvs(n, edges, fvs)
            if verbose:
                status = "✓ VALID" if valid else "✗ INVALID"
                print(f"DFVS size = {len(fvs):4d}  |  Time = {elapsed_ms:8.2f} ms  |  {status}")
            
            algo_result["FVS_size"] = len(fvs)
            algo_result["runtime"] = round(elapsed_ms / 1000.0, 6)
            algo_result["validity"] = valid
            kernel_s = round((stage_metrics or {}).get("kernelization_ms", 0.0) / 1000.0, 6)
            gnn_s = round((stage_metrics or {}).get("gnn_candidate_ms", 0.0) / 1000.0, 6)
            algo_result["kernelization_time"] = kernel_s
            algo_result["gnn_candidate_time"] = gnn_s
            algo_result["initial_kernel_size"] = int((stage_metrics or {}).get("initial_kernel_size", n))
            algo_result["final_kernel_size"] = int((stage_metrics or {}).get("final_kernel_size", n))
            algo_result["n_dynamic_reductions"] = int((stage_metrics or {}).get("n_dynamic_reductions", 0))
            algo_result["solution_size"] = len(fvs)
            algo_result["time_seconds"] = round(elapsed_ms / 1000.0, 6)
            
            results[f"{alg}_size"]  = len(fvs)
            results[f"{alg}_ms"]    = round(elapsed_ms, 2)
            results[f"{alg}_valid"] = valid
            results[f"{alg}_kernel_ms"] = round((stage_metrics or {}).get("kernelization_ms", 0.0), 2)
            results[f"{alg}_gnn_ms"] = round((stage_metrics or {}).get("gnn_candidate_ms", 0.0), 2)
        
        # Immediately save this algorithm's result to its CSV
        append_result_to_csv(csv_path, algo_result)

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
        choices=["BST", "IC", "MA", "KMA", "DKMA", "GNN-KMA", "GNN-KMA-2", "GNN-KMA-3", "GNN-DKMA", "ALL", "PUREALGO"],
        help="Algorithm: BST/IC exact, MA/KMA/DKMA heuristic, GNN-KMA/GNN-KMA-2/GNN-KMA-3/GNN-DKMA hybrid, ALL compare, PUREALGO baseline set"
    )
    parser.add_argument(
        "--test", required=True,
        help="Path to a single .gr/.txt file OR a directory of graph files"
    )
    parser.add_argument(
        "--results-dir", default="results",
        help="Directory where per-algorithm CSV result files will be stored (default: results)"
    )
    parser.add_argument(
        "--result-tag", choices=["exact", "heuristic"], default=None,
        help="Optional output suffix override for CSV naming based on dataset track"
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional: also save combined summary to this CSV file"
    )
    parser.add_argument(
        "--pop", type=int, default=20,
        help="[MA/KMA/GNN-KMA] Population size (default: 20)"
    )
    parser.add_argument(
        "--gens", "--gen", type=int, default=100,
        help="[MA/KMA/GNN-KMA] Maximum generations (default: 100)"
    )
    parser.add_argument(
        "--timeout", type=int, default=600,
        help="Hard wall-clock timeout in seconds for MA/KMA/GNN-KMA/GNN-KMA-2/GNN-KMA-3 solvers (default: 600)"
    )
    parser.add_argument(
        "--earlystop", type=int, default=20,
        help="Patience / early-stopping generations without improvement (default: 20)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-run verbose output"
    )
    parser.add_argument(
        "--gnn-threshold", type=float, default=0.2,
        help="[GNN-KMA/GNN-KMA-2/GNN-KMA-3 only] Probability threshold for GNN candidate selection (default: 0.2)"
    )
    parser.add_argument(
        "--gnn-hidden", type=int, default=None,
        help="[GNN-KMA/GNN-KMA-2/GNN-KMA-3 only] Optional hidden dimension override for loading GNN weights"
    )
    parser.add_argument(
        "--gnn-timeout", type=int, default=60,
        help="Hard wall-clock timeout in seconds for the GNN candidate inference phase (default: 60)"
    )
    parser.add_argument(
        "--gnn-device", choices=["auto", "cuda", "cpu"], default="cpu",
        help="Preferred GNN inference backend (default: cpu)"
    )
    parser.add_argument(
        "--feature-device", choices=["auto", "cuda", "cpu"], default="cpu",
        help="Preferred feature-engineering backend hint (default: cpu)"
    )
    parser.add_argument(
        "--gnn-batch-size", type=int, default=2048,
        help="NeighborLoader batch size for GNN inference (default: 2048)"
    )
    parser.add_argument(
        "--commit-threshold", type=float, default=0.6,
        help="[DKMA/GNN-DKMA] Population consensus threshold for dynamic commitments (default: 0.6)"
    )
    parser.add_argument(
        "--dynkern-every", type=int, default=5,
        help="[DKMA/GNN-DKMA] Re-kernelize every N generations (default: 5)"
    )
    parser.add_argument(
        "--no-gain-search", action="store_true",
        help="[DKMA/GNN-DKMA] Disable post-loop gain-based local search"
    )
    parser.add_argument(
        "--no-diversify", action="store_true",
        help="[DKMA/GNN-DKMA] Disable topological-order diversification"
    )

    args = parser.parse_args()

    if not (0.0 <= args.gnn_threshold <= 1.0):
        print("ERROR: --gnn-threshold must be between 0.0 and 1.0")
        sys.exit(1)
    if args.timeout <= 0:
        print("ERROR: --timeout must be a positive integer")
        sys.exit(1)
    if args.gnn_timeout <= 0:
        print("ERROR: --gnn-timeout must be a positive integer")
        sys.exit(1)
    if args.gnn_batch_size <= 0:
        print("ERROR: --gnn-batch-size must be a positive integer")
        sys.exit(1)
    if args.earlystop <= 0:
        print("ERROR: --earlystop must be a positive integer")
        sys.exit(1)
    if args.gnn_hidden is not None and args.gnn_hidden <= 0:
        print("ERROR: --gnn-hidden must be a positive integer")
        sys.exit(1)
    if not (0.0 < args.commit_threshold < 1.0):
        print("ERROR: --commit-threshold must be in (0, 1)")
        sys.exit(1)
    if args.dynkern_every <= 0:
        print("ERROR: --dynkern-every must be a positive integer")
        sys.exit(1)

    _validate_cuda_requested_or_exit(args.gnn_device)

    # ── Collect input files ──────────────────────────────────────────────────
    test_path = Path(args.test)
    if test_path.is_file():
        files = [str(test_path)]
    elif test_path.is_dir():
        files = collect_graph_files(test_path)

        if not files:
            print(f"No graph files found in {test_path}")
            sys.exit(1)
        print(f"Found {len(files)} graph file(s) in {test_path} (recursive)")
    else:
        print(f"ERROR: --test path does not exist: {args.test}")
        sys.exit(1)

    # ── Run benchmarks ───────────────────────────────────────────────────────
    if not args.quiet:
        print(f"Runtime diagnostics: {_torch_runtime_diagnostics()}")

    all_results = []
    for filepath in files:
        result = run_on_file(filepath, args.algo, args.pop, args.gens,
                             gnn_threshold=args.gnn_threshold,
                             gnn_hidden=args.gnn_hidden,
                             timeout_seconds=args.timeout,
                             gnn_timeout=args.gnn_timeout,
                             early_stop=args.earlystop,
                             gnn_device=args.gnn_device,
                             feature_device=args.feature_device,
                             gnn_batch_size=args.gnn_batch_size,
                             commit_threshold=args.commit_threshold,
                             dynkern_every=args.dynkern_every,
                             gain_search=(not args.no_gain_search),
                             diversify=(not args.no_diversify),
                             results_dir=args.results_dir,
                            result_tag=args.result_tag,
                             verbose=not args.quiet)
        if result:
            all_results.append(result)

    # ── Print summary ────────────────────────────────────────────────────────
    if len(all_results) > 1 or args.quiet:
        print(f"\n{'═' * 80}")
        print(f"  SUMMARY  (DIRECTED {args.algo} on {len(all_results)} file(s))")
        print(f"{'═' * 80}")

        if args.algo == "ALL":
            algos_ran = ["BST", "IC", "MA", "KMA", "DKMA", "GNN-KMA", "GNN-KMA-2", "GNN-KMA-3", "GNN-DKMA"]
        elif args.algo == "PUREALGO":
            algos_ran = ["BST", "IC", "MA", "KMA", "DKMA"]
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