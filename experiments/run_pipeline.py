#!/usr/bin/env python3
"""
Unified data + benchmark runner for the FVS project.

Smart behavior:
1) Checks how many synthetic datasets already exist per category
2) Only generates missing files to reach your target count
3) Keeps category distribution percentages constant (undirected: 20/20/20/20/20, directed: 30/20/20/15/15)
4) Downloads real-world data
5) Runs directed and/or undirected benchmarks

Examples:
  python experiments/run_pipeline.py --mode all --algo ALL --total-undirected 100 --total-directed 100
  python experiments/run_pipeline.py --mode directed --algo MA --include-pace --total-directed 50
  python experiments/run_pipeline.py --mode undirected --algo IC --prepare-only --total-undirected 100
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
synthetic_DIR = DATA_DIR / "synthetic"

DOWNLOAD_REAL_SCRIPT = PROJECT_ROOT / "data" / "download_real_world.py"
GENERATE_SYNTH_SCRIPT = PROJECT_ROOT / "data" / "generate_synthetic.py"
BENCHMARK_UND_SCRIPT = PROJECT_ROOT / "experiments" / "benchmark_undirected.py"
BENCHMARK_DIR_SCRIPT = PROJECT_ROOT / "experiments" / "benchmark_directed.py"

GRAPH_EXTENSIONS = {".txt", ".gr", ".edges", ".graph", ".dimacs", ".mtx"}

# Category weights - MUST match those in generate_synthetic.py
UNDIRECTED_WEIGHTS: Dict[str, float] = {
    "real_world": 0.20,
    "scale_free": 0.20,
    "small_world": 0.20,
    "random_er": 0.20,
    "grids_trees": 0.20,
}

DIRECTED_WEIGHTS: Dict[str, float] = {
    "real_world_ego": 0.30,
    "scale_free": 0.20,
    "random_er": 0.20,
    "directed_grids": 0.15,
    "dags": 0.15,
}

EXACT_TRACK = "exact_track"
HEURISTIC_TRACK = "heuristic_track"


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


def count_existing_txt_files(family: str) -> int:
    """Count all .txt files in data/synthetic/{family}/*/"""
    synth_root = DATA_DIR / "synthetic" / family
    if not synth_root.exists():
        return 0
    return len(list(synth_root.rglob("*.txt")))


def allocate_counts(total: int, weights: Dict[str, float]) -> Dict[str, int]:
    """Allocate total count across categories maintaining weight percentages."""
    if total < 0:
        raise ValueError("total must be >= 0")
    if not weights:
        return {}

    base = {k: int(total * w) for k, w in weights.items()}
    used = sum(base.values())
    rem = total - used

    order = sorted(weights.keys(), key=lambda k: ((total * weights[k]) - base[k]), reverse=True)
    i = 0
    while rem > 0:
        k = order[i % len(order)]
        base[k] += 1
        rem -= 1
        i += 1

    return base


def split_tracks(total: int, exact_ratio: float) -> Tuple[int, int]:
    """Split total into exact_track and heuristic_track."""
    exact = int(total * exact_ratio)
    return exact, total - exact


def plan_per_bucket(total: int, exact_ratio: float, weights: Dict[str, float]) -> Dict[Tuple[str, str], int]:
    per_category = allocate_counts(total, weights)
    plan: Dict[Tuple[str, str], int] = {}
    for category, cat_total in per_category.items():
        exact, heuristic = split_tracks(cat_total, exact_ratio)
        plan[(EXACT_TRACK, category)] = exact
        plan[(HEURISTIC_TRACK, category)] = heuristic
    return plan


def select_synthetic_subset(family: str, total: int, exact_ratio: float) -> Path:
    """Build a synthetic subset folder that contains exactly the requested count."""
    weights = UNDIRECTED_WEIGHTS if family == "undirected" else DIRECTED_WEIGHTS
    src_root = DATA_DIR / "synthetic" / family
    dst_root = synthetic_DIR / family
    plan = plan_per_bucket(total, exact_ratio, weights)

    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    for (track, category), needed in plan.items():
        src_bucket = src_root / track / category
        files = sorted(p for p in src_bucket.glob("*.txt") if p.is_file())
        if len(files) < needed:
            raise RuntimeError(
                f"Not enough files in {src_bucket}: have {len(files)}, need {needed}. "
                "Run preparation again."
            )

        dst_bucket = dst_root / track / category
        dst_bucket.mkdir(parents=True, exist_ok=True)

        # Keep deterministic selection by taking the lexicographically first N files.
        for src in files[:needed]:
            shutil.copy2(src, dst_bucket / src.name)
            copied += 1

    print(f"[SMART] Prepared {copied} synthetic {family} file(s) in {dst_root}")
    return dst_root


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
    """
    Smart dataset preparation:
    - For real-world: prepare using the requested totals (unless skip_real)
    - For synthetic: count existing files, only generate missing ones to reach target
    """
    if not skip_real:
        cmd = [
            sys.executable,
            str(DOWNLOAD_REAL_SCRIPT),
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
            raise RuntimeError("Real-world dataset step failed")

    if not skip_synth:
        # Check how many synthetic files already exist
        existing_undirected = count_existing_txt_files("undirected") if mode in {"all", "undirected"} else 0
        existing_directed = count_existing_txt_files("directed") if mode in {"all", "directed"} else 0

        # Only generate synthetic if needed
        if mode in {"all", "undirected"} and existing_undirected < total_undirected:
            needed = total_undirected - existing_undirected
            print(f"\n[SMART] Undirected: {existing_undirected} exist, need {needed} more to reach {total_undirected}")
            cmd = [
                sys.executable,
                str(GENERATE_SYNTH_SCRIPT),
                "--total-undirected",
                str(total_undirected),
                "--exact-ratio",
                str(exact_ratio),
                "--family",
                "undirected",
            ]
            if force:
                cmd.append("--force")
            rc = run_command(cmd)
            if rc != 0:
                raise RuntimeError("Synthetic undirected dataset step failed")

        if mode in {"all", "directed"} and existing_directed < total_directed:
            needed = total_directed - existing_directed
            print(f"\n[SMART] Directed: {existing_directed} exist, need {needed} more to reach {total_directed}")
            cmd = [
                sys.executable,
                str(GENERATE_SYNTH_SCRIPT),
                "--total-directed",
                str(total_directed),
                "--exact-ratio",
                str(exact_ratio),
                "--family",
                "directed",
            ]
            if force:
                cmd.append("--force")
            rc = run_command(cmd)
            if rc != 0:
                raise RuntimeError("Synthetic directed dataset step failed")

        # Report final state
        if mode in {"all", "undirected"}:
            final_undirected = count_existing_txt_files("undirected")
            status = "[OK]" if final_undirected >= total_undirected else "[WARN]"
            print(f"{status} Undirected: {final_undirected}/{total_undirected} datasets ready")

        if mode in {"all", "directed"}:
            final_directed = count_existing_txt_files("directed")
            status = "[OK]" if final_directed >= total_directed else "[WARN]"
            print(f"{status} Directed: {final_directed}/{total_directed} datasets ready")



def run_undirected(
    algo: str,
    pop: int,
    gens: int,
    quiet: bool,
    synthetic_dir: Path,
    synthetic_only: bool,
) -> List[Path]:
    outputs: List[Path] = []
    test_dirs = [synthetic_dir] if synthetic_only else [DATA_DIR / "raw_undirected", synthetic_dir]

    for test_dir in test_dirs:
        if not has_graph_files(test_dir):
            print(f"[SKIP] no benchmark graph files in {test_dir}")
            continue

        cmd = [
            sys.executable,
            str(BENCHMARK_UND_SCRIPT),
            "--algo",
            algo,
            "--test",
            str(test_dir),
            "--results-dir",
            str(RESULTS_DIR),
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

        if algo == "ALL":
            outputs.extend([
                RESULTS_DIR / "undirected_BST.csv",
                RESULTS_DIR / "undirected_IC.csv",
                RESULTS_DIR / "undirected_MA.csv",
                RESULTS_DIR / "undirected_KMA.csv",
                RESULTS_DIR / "undirected_GNN-KME.csv",
            ])
        else:
            outputs.append(RESULTS_DIR / f"undirected_{algo}.csv")

    return outputs


def run_directed(
    algo: str,
    pop: int,
    gens: int,
    quiet: bool,
    include_pace: bool,
    synthetic_dir: Path,
    synthetic_only: bool,
) -> List[Path]:
    outputs: List[Path] = []
    test_dirs = [synthetic_dir] if synthetic_only else [DATA_DIR / "raw_directed", synthetic_dir]
    if include_pace:
        test_dirs.append(DATA_DIR / "pace2022")

    for test_dir in test_dirs:
        if not has_graph_files(test_dir):
            print(f"[SKIP] no benchmark graph files in {test_dir}")
            continue

        cmd = [
            sys.executable,
            str(BENCHMARK_DIR_SCRIPT),
            "--algo",
            algo,
            "--test",
            str(test_dir),
            "--results-dir",
            str(RESULTS_DIR),
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

        if algo == "ALL":
            outputs.extend([
                RESULTS_DIR / "directed_BST.csv",
                RESULTS_DIR / "directed_IC.csv",
                RESULTS_DIR / "directed_MA.csv",
                RESULTS_DIR / "directed_KMA.csv",
                RESULTS_DIR / "directed_GNN-KME.csv",
            ])
        else:
            outputs.append(RESULTS_DIR / f"directed_{algo}.csv")

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smart dataset preparation + FVS benchmark runner",
        epilog="""
Smart behavior: Only generates synthetic datasets that don't already exist.
Example: --total-undirected 100 will check data/synthetic/undirected/ and only
generate missing files to reach 100 total.
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["all", "undirected", "directed"],
        default="all",
        help="Which benchmark family to run",
    )
    parser.add_argument(
        "--algo",
        choices=["BST", "IC", "MA", "KMA", "GNN-KME", "ALL"],
        default="ALL",
        help="Algorithm selection forwarded to benchmark scripts",
    )
    parser.add_argument("--pop", type=int, default=50, help="Population size for MA/GNN-KME")
    parser.add_argument("--gens", type=int, default=200, help="Max generations for MA/GNN-KME")
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
        help="Force overwrite existing synthetic datasets",
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
        "--synthetic-only",
        action="store_true",
        help="Benchmark only synthetic selected datasets (exclude raw_* folders)",
    )
    parser.add_argument(
        "--total-undirected",
        type=int,
        default=100,
        help="Target synthetic undirected dataset count (only generates missing files)",
    )
    parser.add_argument(
        "--total-directed",
        type=int,
        default=100,
        help="Target synthetic directed dataset count (only generates missing files)",
    )
    parser.add_argument(
        "--exact-ratio",
        type=float,
        default=0.5,
        help="Fraction of each category allocated to exact_track (% split per category preserved)",
    )

    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  FVS Benchmark Pipeline - Smart Dataset Mode")
    print(f"{'='*70}")
    print(f"  Mode: {args.mode} | Algo: {args.algo}")
    print(f"  Target: {args.total_undirected} undirected, {args.total_directed} directed")
    print(f"  Exact ratio: {args.exact_ratio:.2f}")
    print(f"{'='*70}\n")

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
        print("\n[DONE] dataset preparation finished (prepare-only mode)\n")
        return

    selected_undirected_dir = DATA_DIR / "synthetic" / "undirected"
    selected_directed_dir = DATA_DIR / "synthetic" / "directed"
    if args.mode in {"all", "undirected"}:
        selected_undirected_dir = select_synthetic_subset(
            family="undirected",
            total=args.total_undirected,
            exact_ratio=args.exact_ratio,
        )
    if args.mode in {"all", "directed"}:
        selected_directed_dir = select_synthetic_subset(
            family="directed",
            total=args.total_directed,
            exact_ratio=args.exact_ratio,
        )

    all_outputs: List[Path] = []

    if args.mode in {"all", "undirected"}:
        all_outputs.extend(
            run_undirected(
                args.algo,
                args.pop,
                args.gens,
                args.quiet,
                selected_undirected_dir,
                args.synthetic_only,
            )
        )

    if args.mode in {"all", "directed"}:
        all_outputs.extend(
            run_directed(
                args.algo,
                args.pop,
                args.gens,
                args.quiet,
                args.include_pace,
                selected_directed_dir,
                args.synthetic_only,
            )
        )

    print("\n" + "="*70)
    print("  Pipeline Summary")
    print("="*70)
    if not all_outputs:
        print("  No benchmark output files were produced.")
        print("="*70 + "\n")
        return

    for out_file in sorted(set(all_outputs)):
        print(f"  [OK] {out_file}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
