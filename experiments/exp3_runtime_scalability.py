"""
experiments/exp3_runtime_scalability.py
-----------------------------------------
EXP3: Runtime and Scalability Analysis
Measures wall-clock runtime vs graph size for all algorithms.
IC limited to n ≤ 200, KBST limited to n ≤ 500, MEMETIC runs on all.
"""

import json
import logging
from pathlib import Path
from collections import defaultdict

import networkx as nx
import pandas as pd

from algorithms.iterative_compression import IterativeCompression
from algorithms.kernelization_bst import KernelizationBST
from algorithms.memetic_ga import MemeticGA
from analysis.report_writer import ReportWriter
from data.validator import is_valid_fvs, graph_stats
from experiments.runner import is_run_done, run_algorithm_safely, sort_instances
from experiments.exp1_correctness import _infer_graph_type

logger = logging.getLogger(__name__)

EXPERIMENT_ID = "EXP3"

IC_MAX_N    = 200
KBST_MAX_N  = 500


def run(config: dict, report_writer: ReportWriter, done_set: set) -> None:
    """Run EXP3: runtime scalability on ALL instances."""
    results_dir = config["results_dir"]
    all_instances = sort_instances(config.get("all_instances", []))

    solvers_by_limit = [
        (IterativeCompression(), IC_MAX_N,   "IC"),
        (KernelizationBST(),     KBST_MAX_N, "KBST"),
        (MemeticGA(max_generations=100, population_size=50, random_seed=42),
         10**9, "MEMETIC"),
    ]

    logger.info("[EXP3] Scalability: %d instances", len(all_instances))

    for instance_id, graph in all_instances:
        stats = graph_stats(graph)
        n     = stats["n_vertices"]
        gtype = _infer_graph_type(instance_id)

        for solver, max_n, algo in solvers_by_limit:
            if n > max_n:
                logger.info("[SKIP] %s not run on n=%d (too large, expected exponential)",
                            algo, n)
                continue

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
                notes    = "SKIPPED"
            else:
                fvs_set, _, wall, cpu, mem = outcome
                valid    = is_valid_fvs(graph, fvs_set)
                fvs_size = len(fvs_set)
                notes    = "TIMEOUT" if fvs_size < 0 else ""

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

    # Rebuild cumulative scaling data from report.csv so repeated per-instance
    # calls still produce a complete scalability artifact.
    scaling_data: dict = defaultdict(lambda: defaultdict(list))
    report_csv = results_dir / "report.csv"
    if report_csv.exists():
        try:
            df = pd.read_csv(report_csv)
            exp3_rows = df[(df["experiment_id"] == EXPERIMENT_ID) & (df["is_valid_solution"] == True)]
            for _, row in exp3_rows.iterrows():
                algo = row.get("algorithm")
                gtype = row.get("graph_type")
                n_val = pd.to_numeric(row.get("n_vertices"), errors="coerce")
                t_val = pd.to_numeric(row.get("wall_time_sec"), errors="coerce")
                if pd.isna(n_val) or pd.isna(t_val):
                    continue
                scaling_data[str(algo)][str(gtype)].append({
                    "n": int(n_val),
                    "wall_time_sec": float(t_val),
                })
        except Exception as exc:
            logger.warning("[EXP3] Failed to aggregate cumulative scaling data: %s", exc)

    # Save scaling data as JSON for plot.py
    _save_json(dict(scaling_data), results_dir / "exp3_scaling.json")


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as exc:
        logger.error("Failed to save %s: %s", path, exc)
