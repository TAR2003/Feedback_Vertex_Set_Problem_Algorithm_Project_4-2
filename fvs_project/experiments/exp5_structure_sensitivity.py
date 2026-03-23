"""
experiments/exp5_structure_sensitivity.py
-------------------------------------------
EXP5: Graph Structure Sensitivity
Tests whether algorithm performance differs by graph type.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

from algorithms.iterative_compression import IterativeCompression
from algorithms.kernelization_bst import KernelizationBST
from algorithms.memetic_ga import MemeticGA
from analysis.report_writer import ReportWriter
from data.validator import is_valid_fvs, graph_stats
from experiments.runner import run_algorithm_safely, sort_instances
from experiments.exp1_correctness import _infer_graph_type

logger = logging.getLogger(__name__)
EXPERIMENT_ID = "EXP5"

SOLVERS = [
    IterativeCompression(),
    KernelizationBST(),
    MemeticGA(max_generations=100, population_size=50, random_seed=42),
]


def run(config: dict, report_writer: ReportWriter, done_set: set) -> None:
    """Run EXP5: per-graph-type performance heatmap data collection."""
    perf_csv    = config["perf_csv_path"]
    results_dir = config["results_dir"]
    all_instances = sort_instances(config.get("all_instances", []))

    # Accumulate fvs sizes and runtimes for heatmap
    heatmap: dict = defaultdict(lambda: defaultdict(lambda: {"sizes": [], "times": []}))

    logger.info("[EXP5] Structure sensitivity: %d instances", len(all_instances))

    for instance_id, graph in all_instances:
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
                if valid and fvs_size >= 0:
                    heatmap[algo][gtype]["sizes"].append(fvs_size)
                    heatmap[algo][gtype]["times"].append(wall)

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

    # Compute medians per cell
    summary: dict = {}
    for algo, gtypes in heatmap.items():
        summary[algo] = {}
        for gtype, data in gtypes.items():
            sizes = data["sizes"]
            times = data["times"]
            summary[algo][gtype] = {
                "median_fvs":  _median(sizes),
                "mean_fvs":    (_sum(sizes) / len(sizes)) if sizes else None,
                "median_time": _median(times),
            }

    _save_json(summary, results_dir / "exp5_heatmap_data.json")


def _median(lst: list):
    if not lst:
        return None
    s = sorted(lst)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _sum(lst: list):
    return sum(lst)


def _save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as exc:
        logger.error("Failed to save %s: %s", path, exc)
