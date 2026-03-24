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

import pandas as pd

from algorithms.iterative_compression import IterativeCompression
from algorithms.kernelization_bst import KernelizationBST
from algorithms.memetic_ga import MemeticGA
from analysis.report_writer import ReportWriter
from data.validator import is_valid_fvs, graph_stats
from experiments.runner import is_run_done, run_algorithm_safely, sort_instances
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
    results_dir = config["results_dir"]
    all_instances = sort_instances(config.get("all_instances", []))

    # Accumulate fvs sizes and runtimes for heatmap
    heatmap: dict = defaultdict(lambda: defaultdict(lambda: {"sizes": [], "times": []}))
    
    # Try to load from report.csv first
    report_csv = results_dir / "report.csv"
    if report_csv.exists():
        try:
            df = pd.read_csv(report_csv)
            # Filter for valid solutions from EXP1-5 (structure-sensitive experiments)
            exp_data = df[(df['experiment_id'].isin(['EXP1', 'EXP2', 'EXP3', 'EXP5'])) & 
                         (df['is_valid_solution'] == True)]
            
            for _, row in exp_data.iterrows():
                if pd.notna(row['fvs_size']) and pd.notna(row['wall_time_sec']):
                    algo = row['algorithm']
                    gtype = _infer_graph_type(row['instance_id'])
                    heatmap[algo][gtype]["sizes"].append(int(row['fvs_size']))
                    heatmap[algo][gtype]["times"].append(float(row['wall_time_sec']))
            
            logger.info("[EXP5] Loaded %d data points from report.csv", len(df))
        except Exception as exc:
            logger.warning("[EXP5] Failed to load from report.csv: %s", exc)

    # If no data loaded, run new instances
    if not any(heatmap.values()):
        logger.info("[EXP5] Structure sensitivity: %d instances", len(all_instances))
        
        for instance_id, graph in all_instances:
            stats = graph_stats(graph)
            gtype = _infer_graph_type(instance_id)

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

    logger.info("[EXP5] Saving heatmap with %d cells", len(summary))
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
