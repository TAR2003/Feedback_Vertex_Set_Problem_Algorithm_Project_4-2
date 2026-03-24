"""
experiments/exp9_realworld.py
-------------------------------
EXP9: Real-World Network Validation
Performance on realistic social/infrastructure networks.
"""

import logging
from pathlib import Path

from algorithms.iterative_compression import IterativeCompression
from algorithms.kernelization_bst import KernelizationBST
from algorithms.memetic_ga import MemeticGA
from analysis.report_writer import ReportWriter
from data.validator import is_valid_fvs, graph_stats
from experiments.runner import is_run_done, run_algorithm_safely, sort_instances
from experiments.exp1_correctness import _infer_graph_type

logger = logging.getLogger(__name__)
EXPERIMENT_ID = "EXP9"

SOLVERS = [
    IterativeCompression(),
    KernelizationBST(),
    MemeticGA(max_generations=100, population_size=50, random_seed=42),
]


def run(config: dict, report_writer: ReportWriter, done_set: set) -> None:
    """Run EXP9: all algorithms on all real-world graphs."""
    all_instances = config.get("all_instances", [])

    rw = [(iid, g) for iid, g in all_instances if "realworld" in iid]
    rw = sort_instances(rw)

    logger.info("[EXP9] Real-world: %d datasets × %d algorithms", len(rw), len(SOLVERS))

    for instance_id, graph in rw:
        stats = graph_stats(graph)
        gtype = "realworld"

        for solver in SOLVERS:
            algo    = solver.short_name()
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
                wall, cpu, mem = 0.0, 0.0, 0.0
                fvs_size = -1
                valid    = False
            else:
                fvs_set, _, wall, cpu, mem = outcome
                valid    = is_valid_fvs(graph, fvs_set)
                fvs_size = len(fvs_set)

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
                wall_time_sec=wall,
                cpu_time_sec=cpu,
                peak_memory_mb=mem,
                is_valid_solution=valid,
                notes=f"density={stats['density']:.4f}, avg_deg={stats['avg_degree']:.2f}",
            )
