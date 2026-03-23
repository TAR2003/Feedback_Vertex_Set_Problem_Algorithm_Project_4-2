"""
experiments/exp1_correctness.py
--------------------------------
EXP1: Correctness Validation
Verify every algorithm produces a valid FVS on small instances (n ≤ 50).
"""

import logging
from pathlib import Path

import networkx as nx

from algorithms.iterative_compression import IterativeCompression
from algorithms.kernelization_bst import KernelizationBST
from algorithms.memetic_ga import MemeticGA
from analysis.report_writer import ReportWriter
from data.validator import is_valid_fvs, graph_stats
from experiments.runner import run_algorithm_safely, sort_instances

logger = logging.getLogger(__name__)

EXPERIMENT_ID = "EXP1"
MAX_N = 50  # Only run on small instances

SOLVERS = [
    IterativeCompression(),
    KernelizationBST(),
    MemeticGA(max_generations=50, population_size=30, random_seed=42),
]


def run(
    config: dict,
    report_writer: ReportWriter,
    done_set: set,
) -> None:
    """
    Run EXP1: correctness validation on all instances with n ≤ MAX_N.

    Args:
        config:        Configuration dict from main.py.
        report_writer: ReportWriter instance for results/report.csv.
        done_set:      Mutable set of (instance_id, algorithm) already done.
    """
    perf_csv = config["perf_csv_path"]
    all_instances: list = config.get("all_instances", [])

    # Filter to small instances only
    small = [(iid, g) for iid, g in all_instances
             if g.number_of_nodes() <= MAX_N]
    small = sort_instances(small)

    total = len(small) * len(SOLVERS)
    valid_count = 0
    run_count   = 0

    logger.info("[EXP1] Correctness check: %d instances × %d algorithms = %d runs",
                len(small), len(SOLVERS), total)

    for instance_id, graph in small:
        stats = graph_stats(graph)
        gtype = _infer_graph_type(instance_id)

        for solver in SOLVERS:
            algo = solver.short_name()
            outcome = run_algorithm_safely(
                solver, graph, instance_id, algo, perf_csv, done_set
            )

            if outcome is None:
                # Skipped (done/timeout/error) — still count as attempted
                fvs_size = -1
                valid    = False
                notes    = "SKIPPED"
            else:
                fvs_set, _, wall, cpu, mem = outcome
                valid    = is_valid_fvs(graph, fvs_set)
                fvs_size = len(fvs_set) if fvs_set else 0
                valid_count += int(valid)
                run_count   += 1
                notes = "" if valid else "INVALID_FVS"
                if not valid:
                    logger.error("[EXP1] INVALID FVS! %s | %s", instance_id, algo)

            report_writer.write_row(
                experiment_id=EXPERIMENT_ID,
                instance_id=instance_id,
                graph_type=gtype,
                n_vertices=stats["n_vertices"],
                n_edges=stats["n_edges"],
                graph_density=stats["density"],
                algorithm=algo,
                run_number=1,
                fvs_size=fvs_size,
                wall_time_sec=outcome[2] if outcome else 0.0,
                cpu_time_sec=outcome[3]  if outcome else 0.0,
                peak_memory_mb=outcome[4] if outcome else 0.0,
                is_valid_solution=valid,
                notes=notes,
            )

    validity_rate = (valid_count / run_count * 100) if run_count > 0 else 0
    logger.info("[EXP1] Validity rate: %.1f%% (%d/%d)", validity_rate, valid_count, run_count)


def _infer_graph_type(instance_id: str) -> str:
    """Derive graph type label from instance_id prefix."""
    for prefix in ("ER_", "BA_", "Grid_", "WS_", "CycleHeavy_", "Tree_", "realworld_"):
        if instance_id.startswith(prefix):
            return prefix.rstrip("_")
    return "Unknown"
