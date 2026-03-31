#!/usr/bin/env python3
"""
Unified data + benchmark runner for the FVS project.

What it does:
1) Ensures data folders are populated (real-world + synthetic) without re-downloading
   or regenerating files that already exist.
2) Runs directed and/or undirected benchmarks using existing benchmark CLIs.

Examples:
  python experiments/run_pipeline.py --mode all --algo ALL
  python experiments/run_pipeline.py --mode directed --algo MA --include-pace
  python experiments/run_pipeline.py --mode undirected --algo IC --prepare-only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

DOWNLOAD_REAL_SCRIPT = PROJECT_ROOT / "data" / "download_real_world.py"
GENERATE_SYNTH_SCRIPT = PROJECT_ROOT / "data" / "generate_synthetic.py"
BENCHMARK_UND_SCRIPT = PROJECT_ROOT / "experiments" / "benchmark_undirected.py"
BENCHMARK_DIR_SCRIPT = PROJECT_ROOT / "experiments" / "benchmark_directed.py"

GRAPH_EXTENSIONS = {".txt", ".gr", ".edges", ".graph", ".dimacs", ".mtx"}


def has_graph_files(folder: Path) -> bool:
    if not folder.exists() or not folder.is_dir():
        return False
    for item in folder.rglob("*"):
        if not item.is_file():
            continue
        if item.suffix.lower() in GRAPH_EXTENSIONS:
            return True
        if item.suffix == "" and not item.name.startswith("."):
            return True
    return False


def run_command(cmd: Sequence[str]) -> int:
    print("[RUN] " + " ".join(str(x) for x in cmd))
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return proc.returncode


def prepare_datasets(
    mode: str,
    skip_real: bool,
    skip_synth: bool,
    force: bool,
    total_undirected: int,
    total_directed: int,
    exact_ratio: float,
) -> None:
    if not skip_real:
        cmd = [sys.executable, str(DOWNLOAD_REAL_SCRIPT)]
        if force:
            cmd.append("--force")
        rc = run_command(cmd)
        if rc != 0:
            raise RuntimeError("Real-world dataset step failed")

    if not skip_synth:
        cmd = [
            sys.executable,
            str(GENERATE_SYNTH_SCRIPT),
            "--total-undirected",
            str(total_undirected),
            "--total-directed",
            str(total_directed),
            "--exact-ratio",
            str(exact_ratio),
        ]
        if mode == "undirected":
            cmd.extend(["--family", "undirected"])
        elif mode == "directed":
            cmd.extend(["--family", "directed"])
        if force:
            cmd.append("--force")
        rc = run_command(cmd)
        if rc != 0:
            raise RuntimeError("Synthetic dataset step failed")


def run_undirected(algo: str, pop: int, gens: int, quiet: bool, stamp: str) -> List[Path]:
    outputs: List[Path] = []
    test_dirs = [
        DATA_DIR / "raw_undirected",
        DATA_DIR / "synthetic" / "undirected",
    ]

    for test_dir in test_dirs:
        if not has_graph_files(test_dir):
            print(f"[SKIP] no benchmark graph files in {test_dir}")
            continue

        out_csv = RESULTS_DIR / f"undirected_{test_dir.name}_{algo}_{stamp}.csv"
        cmd = [
            sys.executable,
            str(BENCHMARK_UND_SCRIPT),
            "--algo",
            algo,
            "--test",
            str(test_dir),
            "--output",
            str(out_csv),
            "--pop",
            str(pop),
            "--gens",
            str(gens),
        ]
        if quiet:
            cmd.append("--quiet")

        rc = run_command(cmd)
        if rc != 0:
            print(f"[WARN] benchmark failed for {test_dir}")
            continue

        outputs.append(out_csv)

    return outputs


def run_directed(algo: str, pop: int, gens: int, quiet: bool, include_pace: bool, stamp: str) -> List[Path]:
    outputs: List[Path] = []
    test_dirs = [
        DATA_DIR / "raw_directed",
        DATA_DIR / "synthetic" / "directed",
    ]
    if include_pace:
        test_dirs.append(DATA_DIR / "pace2022")

    for test_dir in test_dirs:
        if not has_graph_files(test_dir):
            print(f"[SKIP] no benchmark graph files in {test_dir}")
            continue

        out_csv = RESULTS_DIR / f"directed_{test_dir.name}_{algo}_{stamp}.csv"
        cmd = [
            sys.executable,
            str(BENCHMARK_DIR_SCRIPT),
            "--algo",
            algo,
            "--test",
            str(test_dir),
            "--output",
            str(out_csv),
            "--pop",
            str(pop),
            "--gens",
            str(gens),
        ]
        if quiet:
            cmd.append("--quiet")

        rc = run_command(cmd)
        if rc != 0:
            print(f"[WARN] benchmark failed for {test_dir}")
            continue

        outputs.append(out_csv)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap datasets and run FVS benchmarks")
    parser.add_argument(
        "--mode",
        choices=["all", "undirected", "directed"],
        default="all",
        help="Which benchmark family to run",
    )
    parser.add_argument(
        "--algo",
        choices=["BST", "IC", "MA", "KME", "HYBRID", "ALL"],
        default="ALL",
        help="Algorithm selection forwarded to benchmark scripts",
    )
    parser.add_argument("--pop", type=int, default=50, help="Population size for MA/HYBRID")
    parser.add_argument("--gens", type=int, default=200, help="Max generations for MA/HYBRID")
    parser.add_argument(
        "--include-pace",
        action="store_true",
        help="Include data/pace2022 in directed benchmark runs",
    )
    parser.add_argument(
        "--skip-real",
        action="store_true",
        help="Skip real-world data download step",
    )
    parser.add_argument(
        "--skip-synthetic",
        action="store_true",
        help="Skip synthetic data generation step",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite in data preparation scripts",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only prepare datasets, do not run benchmarks",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Pass quiet mode to benchmark scripts",
    )
    parser.add_argument(
        "--total-undirected",
        type=int,
        default=100_000,
        help="Synthetic undirected total count (percentage split stays fixed)",
    )
    parser.add_argument(
        "--total-directed",
        type=int,
        default=100_000,
        help="Synthetic directed total count (percentage split stays fixed)",
    )
    parser.add_argument(
        "--exact-ratio",
        type=float,
        default=0.5,
        help="Fraction of each synthetic category allocated to exact_track",
    )

    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    prepare_datasets(
        mode=args.mode,
        skip_real=args.skip_real,
        skip_synth=args.skip_synthetic,
        force=args.force,
        total_undirected=args.total_undirected,
        total_directed=args.total_directed,
        exact_ratio=args.exact_ratio,
    )

    if args.prepare_only:
        print("[DONE] dataset preparation finished (prepare-only mode)")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_outputs: List[Path] = []

    if args.mode in {"all", "undirected"}:
        all_outputs.extend(run_undirected(args.algo, args.pop, args.gens, args.quiet, stamp))

    if args.mode in {"all", "directed"}:
        all_outputs.extend(
            run_directed(args.algo, args.pop, args.gens, args.quiet, args.include_pace, stamp)
        )

    print("\nPipeline summary")
    print("----------------")
    if not all_outputs:
        print("No benchmark output files were produced.")
        return

    for out_file in all_outputs:
        print(f"[OK] {out_file}")


if __name__ == "__main__":
    main()
