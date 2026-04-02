#!/usr/bin/env python3
"""
Run heuristics-only benchmark track (N > 35 buckets) across directed/undirected folders.

This command only targets:
- data/synthetic/undirected/heuristic_track/**
- data/synthetic/directed/heuristic_track/**

Default algorithms: MA, KMA, GNN-KMA.

Examples:
  python experiments/run_heuristics_track.py
  python experiments/run_heuristics_track.py --mode undirected --algos MA KMA --pop 80 --gens 300
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "synthetic"
RESULTS_DIR = PROJECT_ROOT / "results"

UND_BENCH = PROJECT_ROOT / "experiments" / "benchmark_undirected.py"
DIR_BENCH = PROJECT_ROOT / "experiments" / "benchmark_directed.py"


def _run(cmd: Sequence[str]) -> int:
    print("[RUN] " + " ".join(str(x) for x in cmd))
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return proc.returncode


def _run_undirected(algos: Sequence[str], pop: int, gens: int, quiet: bool, stamp: str) -> List[Path]:
    out_files: List[Path] = []
    target = DATA_DIR / "undirected" / "heuristic_track"
    if not target.exists():
        print(f"[SKIP] Missing folder: {target}")
        return out_files

    for algo in algos:
        out_csv = RESULTS_DIR / f"heuristic_undirected_{algo}_{stamp}.csv"
        cmd = [
            sys.executable,
            str(UND_BENCH),
            "--algo",
            algo,
            "--test",
            str(target),
            "--output",
            str(out_csv),
            "--pop",
            str(pop),
            "--gens",
            str(gens),
        ]
        if quiet:
            cmd.append("--quiet")
        rc = _run(cmd)
        if rc == 0:
            out_files.append(out_csv)
        else:
            print(f"[WARN] undirected {algo} failed with exit code {rc}")

    return out_files


def _run_directed(algos: Sequence[str], pop: int, gens: int, quiet: bool, stamp: str) -> List[Path]:
    out_files: List[Path] = []
    target = DATA_DIR / "directed" / "heuristic_track"
    if not target.exists():
        print(f"[SKIP] Missing folder: {target}")
        return out_files

    for algo in algos:
        out_csv = RESULTS_DIR / f"heuristic_directed_{algo}_{stamp}.csv"
        cmd = [
            sys.executable,
            str(DIR_BENCH),
            "--algo",
            algo,
            "--test",
            str(target),
            "--output",
            str(out_csv),
            "--pop",
            str(pop),
            "--gens",
            str(gens),
        ]
        if quiet:
            cmd.append("--quiet")
        rc = _run(cmd)
        if rc == 0:
            out_files.append(out_csv)
        else:
            print(f"[WARN] directed {algo} failed with exit code {rc}")

    return out_files


def _validate_algos(algos: Sequence[str]) -> None:
    allowed = {"MA", "KMA", "GNN-KMA"}
    invalid = [a for a in algos if a not in allowed]
    if invalid:
        raise ValueError(
            f"Invalid heuristic algorithm(s): {invalid}. Allowed: {sorted(allowed)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run heuristics-only benchmark track")
    parser.add_argument(
        "--mode",
        choices=["all", "undirected", "directed"],
        default="all",
        help="Which family to benchmark",
    )
    parser.add_argument(
        "--algos",
        nargs="+",
        default=["MA", "KMA", "GNN-KMA"],
        help="Heuristic algorithms to run (subset of: MA KMA GNN-KMA)",
    )
    parser.add_argument("--pop", type=int, default=50)
    parser.add_argument("--gens", type=int, default=200)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    _validate_algos(args.algos)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputs: List[Path] = []

    if args.mode in {"all", "undirected"}:
        outputs.extend(_run_undirected(args.algos, args.pop, args.gens, args.quiet, stamp))

    if args.mode in {"all", "directed"}:
        outputs.extend(_run_directed(args.algos, args.pop, args.gens, args.quiet, stamp))

    print("\nHeuristic-track summary")
    print("-----------------------")
    if not outputs:
        print("No output CSV files produced.")
        return

    for path in outputs:
        print(f"[OK] {path}")


if __name__ == "__main__":
    main()
