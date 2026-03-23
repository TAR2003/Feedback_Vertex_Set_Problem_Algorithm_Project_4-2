"""
experiments/exp2_solution_quality.py
-------------------------------------
EXP2: Solution Quality Comparison
Compare FVS sizes across IC, KBST, MEMETIC on instances n ≤ 200.
"""

import json
import logging
from pathlib import Path

import networkx as nx

from algorithms.iterative_compression import IterativeCompression
from algorithms.kernelization_bst import KernelizationBST
from algorithms.memetic_ga import MemeticGA
from analysis.report_writer import ReportWriter
from analysis.statistics import friedman_test
from data.validator import is_valid_fvs, graph_stats
from experiments.runner import run_algorithm_safely, sort_instances
from experiments.exp1_correctness import _infer_graph_type

logger = logging.getLogger(__name__)

EXPERIMENT_ID = "EXP2"
MAX_N = 200

SOLVERS = [
    IterativeCompression(),
    KernelizationBST(),
    MemeticGA(max_generations=100, population_size=50, random_seed=42),
]


def run(config: dict, report_writer: ReportWriter, done_set: set) -> None:
    """Run EXP2: solution quality comparison on instances with n ≤ MAX_N."""
    perf_csv = config["perf_csv_path"]
    results_dir: Path = config["results_dir"]
    all_instances = config.get("all_instances", [])

    medium = [(iid, g) for iid, g in all_instances
              if g.number_of_nodes() <= MAX_N]
    medium = sort_instances(medium)

    # Store per-instance fvs sizes for statistical analysis
    fvs_by_algo: dict = {s.short_name(): [] for s in SOLVERS}

    logger.info("[EXP2] Quality comparison: %d instances × %d algorithms",
                len(medium), len(SOLVERS))

    for instance_id, graph in medium:
        stats = graph_stats(graph)
        gtype = _infer_graph_type(instance_id)
        sizes: dict = {}

        for solver in SOLVERS:
            algo    = solver.short_name()
            outcome = run_algorithm_safely(solver, graph, instance_id, algo, perf_csv, done_set)

            if outcome is None:
                fvs_size = -1
                valid    = False
                wall, cpu, mem = 0.0, 0.0, 0.0
                notes    = "SKIPPED"
            else:
                fvs_set, _, wall, cpu, mem = outcome
                valid    = is_valid_fvs(graph, fvs_set)
                fvs_size = len(fvs_set)
                notes    = "" if valid else "INVALID_FVS"
                if valid:
                    fvs_by_algo[algo].append(fvs_size)
                sizes[algo] = fvs_size

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
                notes=notes,
            )

        # Compute win/tie rankings for this instance
        _log_instance_ranking(instance_id, sizes)

    # Friedman test across all algorithms
    groups = [fvs_by_algo[s.short_name()] for s in SOLVERS]
    friedman = friedman_test(*groups)
    logger.info("[EXP2] Friedman test p-value=%.4f, significant=%s",
                friedman.get("p_value", float("nan")), friedman.get("significant"))

    # Save summary JSON
    summary = {
        "friedman_test": friedman,
        "fvs_counts": {k: len(v) for k, v in fvs_by_algo.items()},
        "median_fvs": {k: (sorted(v)[len(v)//2] if v else None)
                       for k, v in fvs_by_algo.items()},
    }
    _save_json(summary, results_dir / "exp2_stats.json")


def _log_instance_ranking(instance_id: str, sizes: dict) -> None:
    """Log which algorithm won on this instance."""
    valid_sizes = {k: v for k, v in sizes.items() if v >= 0}
    if not valid_sizes:
        return
    best_size = min(valid_sizes.values())
    winners   = [k for k, v in valid_sizes.items() if v == best_size]
    logger.debug("[EXP2] %s: winners=%s (size=%d)", instance_id, winners, best_size)


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as exc:
        logger.error("Failed to save %s: %s", path, exc)
