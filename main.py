"""
main.py
-------
Single entry point for the FVS research project.

Usage:
  python main.py                  # Full pipeline
  python main.py --quick          # QUICK_MODE: only n ≤ 200 instances
  python main.py --tiny           # TINY_MODE: only 30 smallest instances
  python main.py --plots-only     # Skip experiments, generate plots only
  python main.py --exp EXP3       # Run only experiment 3
  python main.py --download-only  # Only generate/download datasets
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import psutil

# ---------------------------------------------------------------------------
# Project root setup — add fvs_project/ to sys.path so all imports work
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


# ---------------------------------------------------------------------------
# Logging setup — file + stdout
# ---------------------------------------------------------------------------
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

log_format = "[%(asctime)s] [%(levelname)s] %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(RESULTS_DIR / "run.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------------------------

# TINY_MODE: Number of smallest instances to run (easily configurable)
TINY_MODE_COUNT = 30  # Change this to run fewer/more instances in --tiny mode


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║        FEEDBACK VERTEX SET — CSE 462 Research Project        ║
║   Algorithms: IterativeCompression | KernelBST | MemeticGA   ║
║   Experiments: EXP1–EXP10 | Plots: 16 figures                ║
╚══════════════════════════════════════════════════════════════╝
"""


def print_banner(quick_mode: bool, tiny_mode: bool = False) -> None:
    """Print startup banner with system info."""
    print(BANNER)
    cores = os.cpu_count()
    ram   = psutil.virtual_memory().total / 1e9
    print(f"  CPU cores : {cores}")
    print(f"  RAM       : {ram:.1f} GB")
    print(f"  QUICK_MODE: {quick_mode}")
    print(f"  TINY_MODE : {tiny_mode}")
    print(f"  Project   : {PROJECT_DIR}")
    print()


# ---------------------------------------------------------------------------
# Directory bootstrap
# ---------------------------------------------------------------------------

def create_directories() -> None:
    """Create all required project subdirectories."""
    dirs = [
        PROJECT_DIR / "data" / "synthetic",
        PROJECT_DIR / "data" / "real_world",
        PROJECT_DIR / "results",
        PROJECT_DIR / "figures",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    logger.info("All directories ready.")


# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------

def _load_experiments(selector: str | None) -> list:
    """
    Import and return experiment modules to run.

    Args:
        selector: E.g. "EXP3" to run only experiment 3, or None for all.
    """
    from experiments import (
        exp1_correctness, exp2_solution_quality, exp3_runtime_scalability,
        exp4_pareto, exp5_structure_sensitivity, exp6_ga_parameters,
        exp7_convergence, exp8_optimality_gap, exp9_realworld, exp10_robustness,
    )
    all_exps = [
        ("EXP1",  exp1_correctness),
        ("EXP2",  exp2_solution_quality),
        ("EXP3",  exp3_runtime_scalability),
        ("EXP4",  exp4_pareto),
        ("EXP5",  exp5_structure_sensitivity),
        ("EXP6",  exp6_ga_parameters),
        ("EXP7",  exp7_convergence),
        ("EXP8",  exp8_optimality_gap),
        ("EXP9",  exp9_realworld),
        ("EXP10", exp10_robustness),
    ]
    if selector:
        return [(k, m) for k, m in all_exps if k == selector.upper()]
    return all_exps


PER_INSTANCE_EXPERIMENTS = {"EXP1", "EXP2", "EXP3", "EXP8", "EXP9"}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Full pipeline: generate data → run experiments → generate plots."""
    parser = argparse.ArgumentParser(description="FVS Research Project")
    parser.add_argument("--quick",         action="store_true",
                        help="Quick mode: only n ≤ 200 instances")
    parser.add_argument("--tiny",          action="store_true",
                        help="Tiny mode: only 30 smallest instances")
    parser.add_argument("--plots-only",    action="store_true",
                        help="Skip experiments; generate plots only")
    parser.add_argument("--download-only", action="store_true",
                        help="Only generate/download datasets")
    parser.add_argument("--exp",           type=str, default=None,
                        help="Run only a specific experiment (e.g. EXP3)")
    args = parser.parse_args()

    # Apply QUICK_MODE and TINY_MODE
    quick = args.quick or (os.environ.get("FVS_QUICK_MODE", "1") != "0")
    tiny  = args.tiny
    
    # TINY_MODE implies QUICK_MODE
    if tiny:
        quick = True
    
    if quick:
        os.environ["FVS_QUICK_MODE"] = "1"
    else:
        os.environ["FVS_QUICK_MODE"] = "0"

    print_banner(quick_mode=quick, tiny_mode=tiny)
    create_directories()

    # --- Dataset generation ---
    if not args.plots_only:
        logger.info("=== Step 1: Generating synthetic datasets ===")
        from data.generator import generate_all
        generate_all(quick_mode=quick)

        logger.info("=== Step 2: Downloading real-world datasets ===")
        from data.downloader import download_all
        download_all()

    if args.download_only:
        logger.info("Download-only mode: done.")
        return

    # --- Load all instances ---
    from data.generator import load_all_graphs as load_synthetic
    from data.downloader import load_all_graphs as load_realworld
    from experiments.runner import (
        load_done_set, sort_instances, print_execution_order,
    )
    from analysis.report_writer import ReportWriter

    if not args.plots_only:
        synthetic   = load_synthetic(quick_mode=quick)
        real_world  = load_realworld()
        all_instances = sort_instances(synthetic + real_world)

        # Apply TINY_MODE: keep only smallest instances by n
        if tiny:
            all_instances_sorted_by_n = sorted(
                all_instances,
                key=lambda x: x[1].number_of_nodes()
            )
            all_instances = all_instances_sorted_by_n[:TINY_MODE_COUNT]
            logger.info("TINY_MODE: Filtered to %d smallest instances", TINY_MODE_COUNT)

        print_execution_order(all_instances)

        report_csv = RESULTS_DIR / "report.csv"
        done_set = load_done_set(report_csv)
        total    = len(all_instances)
        logger.info("Total instances to consider: %d", total)
        logger.info("Loaded %d completed run keys from report.csv", len(done_set))

        report_writer = ReportWriter(RESULTS_DIR)

        config = {
            "results_dir":    RESULTS_DIR,
            "figures_dir":    PROJECT_DIR / "figures",
            "all_instances":  all_instances,
            "quick_mode":     quick,
            "tiny_mode":      tiny,
        }

        # --- Run experiments ---
        logger.info("=== Step 3: Running experiments ===")
        experiments = _load_experiments(args.exp)
        per_instance = [(eid, mod) for eid, mod in experiments if eid in PER_INSTANCE_EXPERIMENTS]
        batch_only = [(eid, mod) for eid, mod in experiments if eid not in PER_INSTANCE_EXPERIMENTS]

        completed = 0

        if per_instance:
            logger.info("--- Running per-instance sequence for %d experiments ---", len(per_instance))
            for idx, (instance_id, graph) in enumerate(all_instances, 1):
                n = graph.number_of_nodes()
                m = graph.number_of_edges()
                logger.info("=== Instance %d/%d: %s (n=%d, m=%d) ===",
                            idx, len(all_instances), instance_id, n, m)
                per_instance_config = dict(config)
                per_instance_config["all_instances"] = [(instance_id, graph)]

                for exp_id, exp_module in per_instance:
                    t0 = time.perf_counter()
                    logger.info("--- Starting %s on %s ---", exp_id, instance_id)
                    try:
                        exp_module.run(per_instance_config, report_writer, done_set)
                        completed += 1
                        logger.info("--- %s on %s completed in %.1fs ---",
                                    exp_id, instance_id, time.perf_counter() - t0)
                    except Exception as exc:
                        logger.error("--- %s on %s FAILED: %s ---",
                                     exp_id, instance_id, exc, exc_info=True)

        for exp_id, exp_module in batch_only:
            t0 = time.perf_counter()
            logger.info("--- Starting %s ---", exp_id)
            try:
                exp_module.run(config, report_writer, done_set)
                completed += 1
                elapsed = time.perf_counter() - t0
                logger.info("--- %s completed in %.1fs ---", exp_id, elapsed)
            except Exception as exc:
                logger.error("--- %s FAILED: %s ---", exp_id, exc, exc_info=True)

        logger.info("Experiment executions completed: %d", completed)

    # --- Generate plots ---
    logger.info("=== Step 4: Generating figures ===")
    _run_plot_py()

    # --- Summary ---
    fig_count = len(list((PROJECT_DIR / "figures").glob("*.pdf")))
    print()
    print("=" * 50)
    print(f"  Results saved to : {RESULTS_DIR / 'report.csv'}")
    print(f"  Figures saved to : {PROJECT_DIR / 'figures'}/")
    print(f"  Figures generated: {fig_count}")
    print(f"  Log              : {RESULTS_DIR / 'run.log'}")
    print("=" * 50)


def _run_plot_py() -> None:
    """Import and run plot.py in the same Python process."""
    plot_script = PROJECT_DIR / "plot.py"
    if not plot_script.exists():
        logger.warning("plot.py not found; skipping figure generation")
        return
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("plot", str(plot_script))
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, "main"):
            mod.main()
        logger.info("plot.py completed successfully")
    except Exception as exc:
        logger.error("plot.py failed: %s", exc, exc_info=True)


if __name__ == "__main__":
    main()
