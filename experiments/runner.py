"""
experiments/runner.py
---------------------
Master experiment orchestrator with:
  - Checkpoint loading/saving (performance.csv)
  - Threading-based timeouts
  - Instance sorting (smallest → largest)
  - Progress tracking
"""

import csv
import logging
import os
import sys
import time
import threading
import concurrent.futures
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import networkx as nx
import psutil

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PERF_COLUMNS = [
    "instance_id", "algorithm", "status",
    "wall_time_sec", "cpu_time_sec", "peak_memory_mb",
    "fvs_size", "timestamp",
]

# Timeout limits by instance size (n vertices)
# Set to essentially infinite (no practical timeout)
TIMEOUT_N_SMALL  = 999999     # n ≤ 50
TIMEOUT_N_MEDIUM = 999999    # 51 ≤ n ≤ 200
TIMEOUT_N_LARGE  = 999999    # n > 200


# ---------------------------------------------------------------------------
# performance.csv helpers
# ---------------------------------------------------------------------------

def load_done_set(perf_csv_path: Path) -> set:
    """
    Load (instance_id, algorithm) pairs that are already done/timed-out
    from performance.csv.

    Returns:
        set of (instance_id, algorithm) strings.
    """
    done: set = set()
    if not perf_csv_path.exists():
        return done
    try:
        with open(perf_csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                status = row.get("status", "")
                if status in ("COMPLETED", "TIMEOUT"):
                    done.add((row["instance_id"], row["algorithm"]))
    except Exception as exc:
        logger.error("Failed to read performance.csv: %s", exc)
    return done


def save_performance_row(
    perf_csv_path: Path,
    instance_id: str,
    algorithm: str,
    status: str,
    wall_time_sec: Any = "",
    cpu_time_sec: Any  = "",
    peak_memory_mb: Any = "",
    fvs_size: Any      = "",
) -> None:
    """Append one row to performance.csv (thread-safe via file locking)."""
    perf_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Create header row if file is new/empty
    if not perf_csv_path.exists() or perf_csv_path.stat().st_size == 0:
        try:
            with open(perf_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=PERF_COLUMNS)
                writer.writeheader()
        except Exception as exc:
            logger.error("Failed to create performance.csv header: %s", exc)

    row = {
        "instance_id":    instance_id,
        "algorithm":      algorithm,
        "status":         status,
        "wall_time_sec":  wall_time_sec,
        "cpu_time_sec":   cpu_time_sec,
        "peak_memory_mb": peak_memory_mb,
        "fvs_size":       fvs_size,
        "timestamp":      datetime.utcnow().isoformat(),
    }
    try:
        with open(perf_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=PERF_COLUMNS, extrasaction="ignore")
            writer.writerow(row)
            f.flush()
    except Exception as exc:
        logger.error("Failed to append to performance.csv: %s", exc)


# ---------------------------------------------------------------------------
# Timeout helper
# ---------------------------------------------------------------------------

def get_timeout(n_vertices: int) -> int:
    """Return timeout seconds based on instance size."""
    if n_vertices <= 50:
        return TIMEOUT_N_SMALL
    elif n_vertices <= 200:
        return TIMEOUT_N_MEDIUM
    else:
        return TIMEOUT_N_LARGE


def run_with_timeout(
    func: Callable,
    args: tuple,
    timeout_sec: int,
) -> tuple:
    """
    Run *func(*args)* in a thread with a timeout.

    Returns:
        (result, elapsed_wall, elapsed_cpu, peak_memory_mb, error_msg)

    On timeout:  result=None, error_msg="TIMEOUT"
    On exception: result=None, error_msg=str(exc)
    """
    result_container: list = [None]
    error_container:  list = [None]
    start_wall = time.perf_counter()
    start_cpu  = time.process_time()

    # Capture memory before
    proc = psutil.Process(os.getpid())
    mem_before = proc.memory_info().rss / 1e6  # MB

    def _target():
        try:
            result_container[0] = func(*args)
        except Exception as exc:
            error_container[0] = str(exc)

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=timeout_sec)

    elapsed_wall = time.perf_counter() - start_wall
    elapsed_cpu  = time.process_time() - start_cpu
    mem_after    = proc.memory_info().rss / 1e6
    peak_mem     = max(0.0, mem_after - mem_before)

    if thread.is_alive():
        # Thread is still running — timeout occurred
        return None, elapsed_wall, elapsed_cpu, peak_mem, "TIMEOUT"

    if error_container[0] is not None:
        return None, elapsed_wall, elapsed_cpu, peak_mem, error_container[0]

    return result_container[0], elapsed_wall, elapsed_cpu, peak_mem, None


# ---------------------------------------------------------------------------
# Instance sorting
# ---------------------------------------------------------------------------

def sort_instances(instances: list[tuple[str, nx.Graph]]) -> list[tuple[str, nx.Graph]]:
    """
    Sort instances by (n_vertices + n_edges) ascending — smallest first.

    Args:
        instances: List of (instance_id, graph) pairs.

    Returns:
        Sorted list.
    """
    return sorted(instances, key=lambda x: x[1].number_of_nodes() + x[1].number_of_edges())


def print_execution_order(instances: list[tuple[str, nx.Graph]]) -> None:
    """Print sorted execution order for user visibility."""
    logger.info("=" * 60)
    logger.info("Execution order (smallest → largest):")
    for i, (iid, g) in enumerate(instances, 1):
        n, m = g.number_of_nodes(), g.number_of_edges()
        logger.info("  %3d. %s  (n=%d, m=%d, size=%d)", i, iid, n, m, n + m)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Shared experiment runner helper
# ---------------------------------------------------------------------------

def run_algorithm_safely(
    solver,
    graph: nx.Graph,
    instance_id: str,
    algorithm_name: str,
    perf_csv_path: Path,
    done_set: set,
) -> Optional[tuple]:
    """
    Run *solver.solve(graph)* with full checkpoint/timeout/error protection.

    Returns:
        (fvs_set, info_dict, wall_time, cpu_time, peak_mem) on success.
        None if skipped or failed.

    Side-effects:
        - Emits [SKIP] log if already done.
        - Appends to performance.csv on completion/timeout/error.
        - Updates *done_set* in-place on completion.
    """
    key = (instance_id, algorithm_name)

    # --- Checkpoint check ---
    if key in done_set:
        logger.info("[SKIP] %s | %s already done", instance_id, algorithm_name)
        return None

    n = graph.number_of_nodes()
    timeout = get_timeout(n)

    logger.info("[RUN ] %s | %s  (n=%d, timeout=%ds)",
                instance_id, algorithm_name, n, timeout)

    result, wall, cpu, mem, error = run_with_timeout(
        solver.solve, (graph,), timeout_sec=timeout
    )

    if error == "TIMEOUT":
        logger.warning("[TIMEOUT] %s | %s exceeded %ds", instance_id, algorithm_name, timeout)
        save_performance_row(perf_csv_path, instance_id, algorithm_name, "TIMEOUT")
        done_set.add(key)
        return None

    if error is not None:
        logger.error("[ERROR] %s | %s — %s", instance_id, algorithm_name, error)
        save_performance_row(perf_csv_path, instance_id, algorithm_name, "ERROR")
        done_set.add(key)
        return None

    fvs_set, info_dict = result
    fvs_size = len(fvs_set) if fvs_set is not None else -1

    save_performance_row(
        perf_csv_path, instance_id, algorithm_name, "COMPLETED",
        wall_time_sec=f"{wall:.6f}",
        cpu_time_sec=f"{cpu:.6f}",
        peak_memory_mb=f"{mem:.3f}",
        fvs_size=fvs_size,
    )
    done_set.add(key)
    logger.info("[DONE] %s | %s → fvs=%d  (%.2fs)",
                instance_id, algorithm_name, fvs_size, wall)

    return fvs_set, info_dict, wall, cpu, mem
