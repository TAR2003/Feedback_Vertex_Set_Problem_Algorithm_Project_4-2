"""
experiments/exp8_optimality_gap.py
------------------------------------
EXP8: Optimality Gap Assessment
Compares IC, KBST, MEMETIC against BRUTE_FORCE ground truth on n ≤ 20.
"""

import logging
from pathlib import Path

from algorithms.brute_force import BruteForce
from algorithms.iterative_compression import IterativeCompression
from algorithms.kernelization_bst import KernelizationBST
from algorithms.memetic_ga import MemeticGA
from analysis.report_writer import ReportWriter
from data.validator import is_valid_fvs, graph_stats
from experiments.runner import is_run_done, run_algorithm_safely, sort_instances
from experiments.exp1_correctness import _infer_graph_type

logger = logging.getLogger(__name__)
EXPERIMENT_ID = "EXP8"
MAX_N = 20

APPROX_SOLVERS = [
    IterativeCompression(),
    KernelizationBST(),
    MemeticGA(max_generations=50, population_size=30, random_seed=42),
]
BF_SOLVER = BruteForce()


def run(config: dict, report_writer: ReportWriter, done_set: set) -> None:
    """Run EXP8: optimality gap on all n ≤ 20 instances."""
    all_instances = config.get("all_instances", [])

    tiny = [(iid, g) for iid, g in all_instances
            if g.number_of_nodes() <= MAX_N]
    tiny = sort_instances(tiny)

    logger.info("[EXP8] Optimality gap: %d instances", len(tiny))

    for instance_id, graph in tiny:
        stats = graph_stats(graph)
        gtype = _infer_graph_type(instance_id)

        # --- Ground truth: Brute Force ---
        if is_run_done(done_set, EXPERIMENT_ID, instance_id, "BRUTE_FORCE", run_number=1):
            logger.info("[SKIP] %s | BRUTE_FORCE already recorded for %s", instance_id, EXPERIMENT_ID)
            bf_outcome = None
        else:
            bf_outcome = run_algorithm_safely(
                BF_SOLVER,
                graph,
                EXPERIMENT_ID,
                instance_id,
                "BRUTE_FORCE",
                done_set,
                run_number=1,
            )
        if bf_outcome is None:
            optimal_size = None
        else:
            bf_fvs, _, bf_wall, bf_cpu, bf_mem = bf_outcome
            optimal_size = len(bf_fvs)
            report_writer.write_row(
                experiment_id=EXPERIMENT_ID,
                instance_id=instance_id,
                graph_type=gtype,
                n_vertices=stats["n_vertices"],
                n_edges=stats["n_edges"],
                graph_density=stats["density"],
                algorithm="BRUTE_FORCE",
                run_number=1,
                fvs_size=optimal_size,
                optimal_fvs_size=optimal_size,
                approximation_ratio=1.0,
                optimality_gap_pct=0.0,
                wall_time_sec=bf_wall,
                cpu_time_sec=bf_cpu,
                peak_memory_mb=bf_mem,
                is_valid_solution=True,
                notes="Ground truth",
            )

        # --- Approximate algorithms ---
        for solver in APPROX_SOLVERS:
            algo = solver.short_name()
            if is_run_done(done_set, EXPERIMENT_ID, instance_id, algo, run_number=1):
                logger.info("[SKIP] %s | %s already recorded for %s", instance_id, algo, EXPERIMENT_ID)
                continue

            outcome = run_algorithm_safely(
                solver,
                graph,
                EXPERIMENT_ID,
                instance_id,
                algo,
                done_set,
                run_number=1,
            )

            if outcome is None:
                fvs_size, wall, cpu, mem = -1, 0.0, 0.0, 0.0
                valid = False
                approx_ratio = ""
                gap_pct      = ""
            else:
                fvs_set, _, wall, cpu, mem = outcome
                valid    = is_valid_fvs(graph, fvs_set)
                fvs_size = len(fvs_set)

                if optimal_size is not None and optimal_size > 0 and valid:
                    approx_ratio = fvs_size / optimal_size
                    gap_pct      = (fvs_size - optimal_size) / optimal_size * 100
                elif optimal_size == 0:
                    approx_ratio = 1.0 if fvs_size == 0 else ""
                    gap_pct      = 0.0 if fvs_size == 0 else ""
                else:
                    approx_ratio = ""
                    gap_pct      = ""

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
                optimal_fvs_size=optimal_size if optimal_size is not None else "",
                approximation_ratio=approx_ratio,
                optimality_gap_pct=gap_pct,
                wall_time_sec=wall,
                cpu_time_sec=cpu,
                peak_memory_mb=mem,
                is_valid_solution=valid,
            )
