"""
experiments/exp7_convergence.py
---------------------------------
EXP7: GA Convergence Analysis
Records best FVS size per generation across 5 runs per instance.
"""

import json
import logging
from pathlib import Path

from algorithms.memetic_ga import MemeticGA
from analysis.report_writer import ReportWriter
from data.validator import is_valid_fvs, graph_stats
from experiments.runner import run_algorithm_safely
from experiments.exp1_correctness import _infer_graph_type

logger = logging.getLogger(__name__)
EXPERIMENT_ID = "EXP7"
N_RUNS = 5


def run(config: dict, report_writer: ReportWriter, done_set: set) -> None:
    """Run EXP7: convergence analysis — 5 independent runs on 5 instances."""
    perf_csv    = config["perf_csv_path"]
    results_dir = config["results_dir"]
    all_instances = config.get("all_instances", [])

    # Select 5 representative instances: small, medium, large, real-world, cycle-heavy
    selected = _pick_representative_instances(all_instances)

    convergence_data: dict = {}
    logger.info("[EXP7] Convergence: %d instances × %d runs", len(selected), N_RUNS)

    for instance_id, graph in selected:
        stats = graph_stats(graph)
        gtype = _infer_graph_type(instance_id)
        run_curves: list = []

        for seed in range(1, N_RUNS + 1):
            solver = MemeticGA(
                max_generations=200,
                population_size=50,
                random_seed=seed,
            )
            run_id = f"EXP7_{instance_id}_run{seed}"
            outcome = run_algorithm_safely(solver, graph, run_id, "MEMETIC", perf_csv, done_set)

            if outcome is None:
                continue

            fvs_set, info_dict, wall, cpu, mem = outcome
            valid    = is_valid_fvs(graph, fvs_set)
            fvs_size = len(fvs_set)
            curve    = info_dict.get("convergence", [])
            run_curves.append(curve)

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

        convergence_data[instance_id] = run_curves

    _save_json(convergence_data, results_dir / "exp7_convergence.json")


def _pick_representative_instances(all_instances: list) -> list:
    """Pick one small, one medium, one large, one real-world, one cycle-heavy."""
    small = _pick_by_size(all_instances, 20, 30, exclude="realworld")
    medium = _pick_by_size(all_instances, 90, 120, exclude="realworld")
    large  = _pick_by_size(all_instances, 400, 600, exclude="realworld")
    rw     = next(((i, g) for i, g in all_instances if "realworld" in i), None)
    cyc    = next(((i, g) for i, g in all_instances if "CycleHeavy" in i), None)

    return [x for x in [small, medium, large, rw, cyc] if x is not None]


def _pick_by_size(instances: list, lo: int, hi: int, exclude: str = "") -> tuple:
    """Pick first instance whose n is in [lo, hi] and id doesn't contain exclude."""
    for iid, g in instances:
        if lo <= g.number_of_nodes() <= hi and exclude not in iid:
            return (iid, g)
    return None


def _save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as exc:
        logger.error("Failed to save %s: %s", path, exc)
