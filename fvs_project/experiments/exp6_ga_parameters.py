"""
experiments/exp6_ga_parameters.py
-----------------------------------
EXP6: GA Hyperparameter Sensitivity Grid Search.
4 * 4 * 3 * 5 = 240 runs, all checkpointed.
"""

import hashlib
import json
import logging
from pathlib import Path

import networkx as nx

from algorithms.memetic_ga import MemeticGA
from analysis.report_writer import ReportWriter
from data.validator import is_valid_fvs, graph_stats
from experiments.runner import run_algorithm_safely, sort_instances
from experiments.exp1_correctness import _infer_graph_type

logger = logging.getLogger(__name__)
EXPERIMENT_ID = "EXP6"

# 5 representative instance ids (must match generated names)
REPRESENTATIVE_IDS = [
    "ER_n100_p0.3_seed1",
    "BA_n100_m3_seed1",
    "WS_n100_k6_b03_seed1",
    "Grid_10x10",
    "CycleHeavy_n50_density50_random_overlay",
]

POP_SIZES      = [20, 50, 100, 200]
MUTATION_RATES = [0.01, 0.05, 0.1, 0.2]
CROSSOVER_RATES = [0.5, 0.7, 0.9]


def _params_hash(pop: int, mut: float, cross: float) -> str:
    """Create a short hash string for the parameter combination."""
    key = f"pop{pop}_mut{mut}_cross{cross}"
    return hashlib.md5(key.encode()).hexdigest()[:8]


def run(config: dict, report_writer: ReportWriter, done_set: set) -> None:
    """Run EXP6: GA parameter grid search on 5 representative instances."""
    perf_csv    = config["perf_csv_path"]
    results_dir = config["results_dir"]
    all_instances = config.get("all_instances", [])

    # Build lookup: stem → graph
    instance_map = {iid: g for iid, g in all_instances}

    grid_results: list = []
    total = len(REPRESENTATIVE_IDS) * len(POP_SIZES) * len(MUTATION_RATES) * len(CROSSOVER_RATES)
    logger.info("[EXP6] GA grid search: %d runs total", total)

    for base_id in REPRESENTATIVE_IDS:
        # Try to find the instance (partial match)
        graph = _find_instance(instance_map, base_id)
        if graph is None:
            logger.warning("[EXP6] Instance not found: %s — skipping", base_id)
            continue

        stats = graph_stats(graph)
        gtype = _infer_graph_type(base_id)

        for pop in POP_SIZES:
            for mut in MUTATION_RATES:
                for cross in CROSSOVER_RATES:
                    phash = _params_hash(pop, mut, cross)
                    inst_id = f"EXP6_GA_{base_id}_{phash}"

                    solver = MemeticGA(
                        population_size=pop,
                        max_generations=50,   # Fewer gens for speed in grid search
                        mutation_rate=mut,
                        crossover_rate=cross,
                        random_seed=42,
                    )
                    outcome = run_algorithm_safely(
                        solver, graph, inst_id, "MEMETIC", perf_csv, done_set
                    )

                    if outcome is None:
                        wall, cpu, mem = 0.0, 0.0, 0.0
                        fvs_size = -1
                        valid    = False
                    else:
                        fvs_set, _, wall, cpu, mem = outcome
                        valid    = is_valid_fvs(graph, fvs_set)
                        fvs_size = len(fvs_set)
                        grid_results.append({
                            "base_id":    base_id,
                            "pop":        pop,
                            "mut":        mut,
                            "cross":      cross,
                            "fvs_size":   fvs_size,
                            "wall_time":  wall,
                        })

                    report_writer.write_row(
                        experiment_id=EXPERIMENT_ID,
                        instance_id=inst_id,
                        graph_type=gtype,
                        n_vertices=stats["n_vertices"],
                        n_edges=stats["n_edges"],
                        graph_density=stats["density"],
                        algorithm="MEMETIC",
                        run_number=1,
                        fvs_size=fvs_size,
                        wall_time_sec=wall,
                        cpu_time_sec=cpu,
                        peak_memory_mb=mem,
                        is_valid_solution=valid,
                        notes=f"pop={pop},mut={mut},cross={cross}",
                    )

    _save_json({"grid_results": grid_results}, results_dir / "exp6_grid.json")


def _find_instance(instance_map: dict, base_id: str):
    """Exact match first, then prefix match."""
    if base_id in instance_map:
        return instance_map[base_id]
    for key, g in instance_map.items():
        if key.startswith(base_id[:15]):
            return g
    return None


def _save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as exc:
        logger.error("Failed to save %s: %s", path, exc)
