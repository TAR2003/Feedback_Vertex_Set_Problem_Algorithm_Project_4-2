"""
experiments/exp10_robustness.py
---------------------------------
EXP10: Robustness and Stability of GA
10 independent MEMETIC runs per instance for n ∈ {50, 100, 200}.
"""

import json
import logging
from pathlib import Path
from collections import defaultdict

from algorithms.memetic_ga import MemeticGA
from analysis.report_writer import ReportWriter
from data.validator import is_valid_fvs, graph_stats
from experiments.runner import is_run_done, run_algorithm_safely, sort_instances
from experiments.exp1_correctness import _infer_graph_type

logger = logging.getLogger(__name__)
EXPERIMENT_ID = "EXP10"
N_RUNS = 10
TARGET_N_VALUES = {50, 100, 200}


def run(config: dict, report_writer: ReportWriter, done_set: set) -> None:
    """Run EXP10: 10 independent MEMETIC runs per instance."""
    results_dir = config["results_dir"]
    all_instances = config.get("all_instances", [])

    selected = [(iid, g) for iid, g in all_instances
                if g.number_of_nodes() in TARGET_N_VALUES]
    selected = sort_instances(selected)

    robustness_stats: dict = {}

    logger.info("[EXP10] Robustness: %d instances × %d runs", len(selected), N_RUNS)

    for instance_id, graph in selected:
        stats   = graph_stats(graph)
        gtype   = _infer_graph_type(instance_id)
        sizes_for_run: list = []

        for seed in range(1, N_RUNS + 1):
            run_id = f"EXP10_{instance_id}_run{seed}"
            solver = MemeticGA(
                max_generations=100,
                population_size=50,
                random_seed=seed,
            )
            if is_run_done(done_set, EXPERIMENT_ID, run_id, "MEMETIC", run_number=seed):
                logger.info("[SKIP] %s | MEMETIC run=%d already recorded for %s",
                            run_id, seed, EXPERIMENT_ID)
                continue

            outcome = run_algorithm_safely(
                solver,
                graph,
                EXPERIMENT_ID,
                run_id,
                "MEMETIC",
                done_set,
                run_number=seed,
            )

            if outcome is None:
                fvs_size, wall, cpu, mem = -1, 0.0, 0.0, 0.0
                valid = False
            else:
                fvs_set, _, wall, cpu, mem = outcome
                valid    = is_valid_fvs(graph, fvs_set)
                fvs_size = len(fvs_set)
                if valid:
                    sizes_for_run.append(fvs_size)

            report_writer.write_row(
                experiment_id=EXPERIMENT_ID,
                instance_id=run_id,
                graph_type=gtype,
                n_vertices=stats["n_vertices"],
                n_edges=stats["n_edges"],
                graph_density=stats["density"],
                algorithm="MEMETIC",
                run_number=seed,
                fvs_size=fvs_size,
                wall_time_sec=wall,
                cpu_time_sec=cpu,
                peak_memory_mb=mem,
                is_valid_solution=valid,
                notes=f"seed={seed}",
            )

        # Compute stability statistics after all runs
        if sizes_for_run:
            mean_s = sum(sizes_for_run) / len(sizes_for_run)
            std_s  = _std(sizes_for_run)
            cv     = (std_s / mean_s * 100) if mean_s > 0 else 0.0
            robustness_stats[instance_id] = {
                "mean_fvs_size":           mean_s,
                "std_fvs_size":            std_s,
                "min_fvs_size":            min(sizes_for_run),
                "max_fvs_size":            max(sizes_for_run),
                "coefficient_of_variation": cv,
            }
            logger.info("[EXP10] %s: mean=%.1f, std=%.2f, CV=%.1f%%",
                        instance_id, mean_s, std_s, cv)

    _save_json(robustness_stats, results_dir / "exp10_stats.json")


def _std(data: list) -> float:
    """Sample standard deviation."""
    n = len(data)
    if n < 2:
        return 0.0
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / (n - 1)
    return variance ** 0.5


def _save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as exc:
        logger.error("Failed to save %s: %s", path, exc)
