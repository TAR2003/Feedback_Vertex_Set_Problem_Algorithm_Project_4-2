"""
experiments/runner.py
---------------------
Master experiment orchestrator with:
    - Checkpoint loading from report.csv
  - Threading-based timeouts
  - Instance sorting (smallest → largest)
  - Progress tracking
"""

import csv
import logging
import os
import time
import threading
from pathlib import Path
from typing import Any, Callable, Optional

import networkx as nx
import psutil

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Timeout limits by instance size (n vertices)
# NO TIMEOUTS - set to None to run indefinitely
TIMEOUT_N_SMALL  = None     # n ≤ 50
TIMEOUT_N_MEDIUM = None     # 51 ≤ n ≤ 200
TIMEOUT_N_LARGE  = None     # n > 200


# ---------------------------------------------------------------------------
# report.csv checkpoint helpers
# ---------------------------------------------------------------------------

def _normalize_run_number(value: Any) -> int:
    """Normalize run number values from CSV to an int key component."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def load_done_set(report_csv_path: Path) -> set:
    """
    Load completed rows from report.csv.

    A completed key is:
      (experiment_id, instance_id, algorithm, run_number)

    Returns:
        Set of completion keys.
    """
    done: set = set()
    if not report_csv_path.exists():
        return done

    try:
        with open(report_csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                exp = str(row.get("experiment_id", "")).strip()
                iid = str(row.get("instance_id", "")).strip()
                algo = str(row.get("algorithm", "")).strip()
                if not (exp and iid and algo):
                    continue

                # Only treat successful/completed runs as done.
                # Legacy rows with skipped/failed states often use fvs_size < 0.
                try:
                    fvs_size = float(row.get("fvs_size", "nan"))
                except (TypeError, ValueError):
                    fvs_size = float("nan")
                if not (fvs_size >= 0):
                    continue

                run_number = _normalize_run_number(row.get("run_number", 1))
                done.add((exp, iid, algo, run_number))
    except Exception as exc:
        logger.error("Failed to read report.csv for checkpoint loading: %s", exc)

    return done


def is_run_done(
    done_set: set,
    experiment_id: str,
    instance_id: str,
    algorithm_name: str,
    run_number: int = 1,
) -> bool:
    """Return True if this exact run key already exists in report.csv state."""
    key = (experiment_id, instance_id, algorithm_name, int(run_number))
    return key in done_set


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
    timeout_sec: int | None = None,
) -> tuple:
    """
    Run *func(*args)* in a thread with optional timeout.

    Returns:
        (result, elapsed_wall, elapsed_cpu, peak_memory_mb, error_msg)

    On timeout:  result=None, error_msg="TIMEOUT"
    On exception: result=None, error_msg=str(exc)
    On success:   error_msg=None
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
    
    # If timeout_sec is None, wait indefinitely
    thread.join(timeout=timeout_sec)

    elapsed_wall = time.perf_counter() - start_wall
    elapsed_cpu  = time.process_time() - start_cpu
    mem_after    = proc.memory_info().rss / 1e6
    peak_mem     = max(0.0, mem_after - mem_before)

    if thread.is_alive():
        # Thread is still running — timeout occurred (but should not with timeout_sec=None)
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
    experiment_id: str,
    instance_id: str,
    algorithm_name: str,
    done_set: set,
    run_number: int = 1,
) -> Optional[tuple]:
    """
    Run *solver.solve(graph)* with checkpoint/timeout/error protection.

    Returns:
        (fvs_set, info_dict, wall_time, cpu_time, peak_mem) on success.
        None if skipped or failed.

    Side-effects:
        - Emits [SKIP] log if already done.
        - Updates *done_set* in-place after timeout/error/success so repeated
          calls in the same process do not duplicate work.
    """
    run_number = int(run_number)
    key = (experiment_id, instance_id, algorithm_name, run_number)

    # --- Checkpoint check ---
    if key in done_set:
        logger.info("[SKIP] %s | %s | %s run=%d already done",
                    experiment_id, instance_id, algorithm_name, run_number)
        return None

    n = graph.number_of_nodes()
    timeout = get_timeout(n)

    logger.info("[RUN ] %s | %s  (n=%d)", instance_id, algorithm_name, n)

    result, wall, cpu, mem, error = run_with_timeout(
        solver.solve, (graph,), timeout_sec=timeout
    )

    if error == "TIMEOUT":
        logger.warning("[TIMEOUT] %s | %s exceeded time limit", instance_id, algorithm_name)
        done_set.add(key)
        return None

    if error is not None:
        logger.error("[ERROR] %s | %s — %s", instance_id, algorithm_name, error)
        done_set.add(key)
        return None

    fvs_set, info_dict = result
    fvs_size = len(fvs_set) if fvs_set is not None else -1

    done_set.add(key)
    logger.info("[DONE] %s | %s → fvs=%d  (%.2fs)",
                instance_id, algorithm_name, fvs_size, wall)

    return fvs_set, info_dict, wall, cpu, mem
