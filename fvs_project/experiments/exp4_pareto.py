"""
experiments/exp4_pareto.py
---------------------------
EXP4: Quality-Runtime Pareto Analysis
Given a time budget, which algorithm gives the best FVS size?
"""

import json
import logging
from pathlib import Path

import networkx as nx

from algorithms.iterative_compression import IterativeCompression
from algorithms.kernelization_bst import KernelizationBST
from algorithms.memetic_ga import MemeticGA
from analysis.report_writer import ReportWriter
from data.validator import is_valid_fvs, graph_stats
from experiments.runner import run_algorithm_safely, sort_instances
from experiments.exp1_correctness import _infer_graph_type

logger = logging.getLogger(__name__)

EXPERIMENT_ID = "EXP4"
PARETO_N_VALUES = {20, 50, 100, 200}  # Instance sizes to include

SOLVERS = [
    IterativeCompression(),
    KernelizationBST(),
    MemeticGA(max_generations=100, population_size=50, random_seed=42),
]


def run(config: dict, report_writer: ReportWriter, done_set: set) -> None:
    """Run EXP4: Pareto frontier analysis on mixed-size instances."""
    perf_csv    = config["perf_csv_path"]
    results_dir = config["results_dir"]
    all_instances = config.get("all_instances", [])

    selected = [(iid, g) for iid, g in all_instances
                if g.number_of_nodes() in PARETO_N_VALUES]
    selected = sort_instances(selected)

    pareto_data: list = []

    logger.info("[EXP4] Pareto: %d instances × %d algorithms", len(selected), len(SOLVERS))

    for instance_id, graph in selected:
        stats = graph_stats(graph)
        gtype = _infer_graph_type(instance_id)

        for solver in SOLVERS:
            algo    = solver.short_name()
            outcome = run_algorithm_safely(solver, graph, instance_id, algo, perf_csv, done_set)

            if outcome is None:
                wall, cpu, mem = 0.0, 0.0, 0.0
                fvs_size = -1
                valid    = False
            else:
                fvs_set, _, wall, cpu, mem = outcome
                valid    = is_valid_fvs(graph, fvs_set)
                fvs_size = len(fvs_set)
                if valid:
                    pareto_data.append({
                        "instance_id": instance_id,
                        "algorithm":   algo,
                        "fvs_size":    fvs_size,
                        "wall_time":   wall,
                    })

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
            )

    _save_json({"pareto_points": pareto_data}, results_dir / "exp4_pareto.json")


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as exc:
        logger.error("Failed to save %s: %s", path, exc)
