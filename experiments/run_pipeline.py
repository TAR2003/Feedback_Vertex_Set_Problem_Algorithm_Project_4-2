#!/usr/bin/env python3
"""
Unified benchmark runner for the FVS project (existing datasets only).

Behavior:
1) Checks how many synthetic datasets already exist for undirected/directed families
2) Exits early if requested totals are not available
3) Builds deterministic first-N subsets from existing files
4) Runs directed and/or undirected benchmarks

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
SELECTED_SYNTH_ROOT = DATA_DIR / "_selected_synthetic"

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
    """Count all .txt files in data/synthetic/{family}/**/."""
    synth_root = DATA_DIR / "synthetic" / family
    if not synth_root.exists():
        return 0
    return len(list(synth_root.rglob("*.txt")))


def list_existing_txt_files(family: str) -> List[Path]:
    """List all existing .txt files for a synthetic family in deterministic order."""
    synth_root = DATA_DIR / "synthetic" / family
    if not synth_root.exists():
        return []
    return sorted(p for p in synth_root.rglob("*.txt") if p.is_file())


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


def select_synthetic_subset(
    family: str,
    total: int,
    exact_ratio: float,
    track: str = "both",
) -> Path:
    """Build a synthetic subset folder with deterministic per-track sampling.

    If exact/heuristic track folders exist, selects files according to `track`:
    - both: up to `total` files from each track
    - exact: up to `total` files from exact_track only
    - heuristic: up to `total` files from heuristic_track only
    """
    src_root = DATA_DIR / "synthetic" / family
    dst_root = SELECTED_SYNTH_ROOT / family
    del exact_ratio  # kept for call compatibility

    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    exact_root = src_root / EXACT_TRACK
    heuristic_root = src_root / HEURISTIC_TRACK

    if exact_root.exists() and heuristic_root.exists():
        if track == "both":
            track_roots = [
                (EXACT_TRACK, exact_root),
                (HEURISTIC_TRACK, heuristic_root),
            ]
        elif track == "exact":
            track_roots = [(EXACT_TRACK, exact_root)]
        elif track == "heuristic":
            track_roots = [(HEURISTIC_TRACK, heuristic_root)]
        else:
            raise ValueError("track must be one of: both, exact, heuristic")

        for track_name, track_root in track_roots:
            track_files = sorted(p for p in track_root.rglob("*.txt") if p.is_file())
            actual_count = min(len(track_files), total)
            if len(track_files) < total:
                print(f"[WARN] Not enough synthetic {family} files in {track_name}: "
                      f"have {len(track_files)}, requested {total}. Using {actual_count} files.")

            for src in track_files[:actual_count]:
                rel = src.relative_to(src_root)
                dst = dst_root / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                copied += 1
    else:
        files = list_existing_txt_files(family)
        actual_count = min(len(files), total)
        if len(files) < total:
            print(f"[WARN] Not enough existing synthetic {family} files: "
                  f"have {len(files)}, requested {total}. Using {actual_count} files.")

        for src in files[:actual_count]:
            rel = src.relative_to(src_root)
            dst = dst_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1

    print(f"[SMART] Prepared {copied} synthetic {family} file(s) in {dst_root}")
    return dst_root


def run_command(cmd: Sequence[str]) -> int:
    print("[RUN] " + " ".join(str(x) for x in cmd))
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return proc.returncode



def validate_existing_datasets(mode: str, total_undirected: int, total_directed: int) -> Tuple[int, int]:
    """Validate available synthetic files and return adjusted counts.
    
    Clamps requested counts to available data with warnings.
    Returns: (adjusted_total_undirected, adjusted_total_directed)
    """
    actual_undirected = total_undirected
    actual_directed = total_directed
    
    if mode in {"all", "undirected"}:
        existing_undirected = count_existing_txt_files("undirected")
        if existing_undirected < total_undirected:
            print(
                f"[WARN] Not enough undirected synthetic files: "
                f"have {existing_undirected}, requested {total_undirected}. "
                f"Using {existing_undirected} files."
            )
            actual_undirected = existing_undirected
        else:
            print(f"[OK] Undirected synthetic files: {existing_undirected} (requested {total_undirected})")

    if mode in {"all", "directed"}:
        existing_directed = count_existing_txt_files("directed")
        if existing_directed < total_directed:
            print(
                f"[WARN] Not enough directed synthetic files: "
                f"have {existing_directed}, requested {total_directed}. "
                f"Using {existing_directed} files."
            )
            actual_directed = existing_directed
        else:
            print(f"[OK] Directed synthetic files: {existing_directed} (requested {total_directed})")
    
    return actual_undirected, actual_directed



def _results_path(algo: str, family: str) -> Path:
    if algo in ["BST", "IC"]:
        kind = "exact"
    elif algo in ["MA", "KMA", "GNN-KMA", "GNN-KMA-2"]:
        kind = "heuristic"
    else:
        kind = "unknown"
    return RESULTS_DIR / f"{family}_{algo}_{kind}.csv"


def _results_path_for_tag(algo: str, family: str, result_tag: str | None) -> Path:
    if result_tag in {"exact", "heuristic"}:
        kind = result_tag
    else:
        if algo in ["BST", "IC"]:
            kind = "exact"
        elif algo in ["MA", "KMA", "GNN-KMA", "GNN-KMA-2"]:
            kind = "heuristic"
        else:
            kind = "unknown"
    return RESULTS_DIR / f"{family}_{algo}_{kind}.csv"


def _expand_synthetic_track_targets(synthetic_dir: Path, track: str = "both") -> List[Tuple[Path, str | None]]:
    """Split synthetic input into selected track directories when available."""
    exact_dir = synthetic_dir / EXACT_TRACK
    heuristic_dir = synthetic_dir / HEURISTIC_TRACK

    targets: List[Tuple[Path, str | None]] = []
    if track in {"both", "exact"} and exact_dir.exists() and exact_dir.is_dir():
        targets.append((exact_dir, "exact"))
    if track in {"both", "heuristic"} and heuristic_dir.exists() and heuristic_dir.is_dir():
        targets.append((heuristic_dir, "heuristic"))

    if not targets:
        targets.append((synthetic_dir, None))
    return targets


def run_undirected(
    algo: str,
    pop: int,
    gens: int,
    timeout: int,
    earlystop: int,
    gnn_threshold: float,
    gnn_hidden: int | None,
    quiet: bool,
    synthetic_dir: Path,
    synthetic_only: bool,
    track: str,
) -> List[Path]:
    outputs: List[Path] = []
    test_targets: List[Tuple[Path, str | None]] = []
    if not synthetic_only:
        test_targets.append((DATA_DIR / "raw_undirected", None))
    test_targets.extend(_expand_synthetic_track_targets(synthetic_dir, track=track))

    if algo == "ALL":
        algos_to_run = ["BST", "IC", "MA", "KMA", "GNN-KMA", "GNN-KMA-2"]
    elif algo == "PUREALGO":
        algos_to_run = ["BST", "IC", "MA", "KMA"]
    else:
        algos_to_run = [algo]

    for test_dir, result_tag in test_targets:
        if not has_graph_files(test_dir):
            print(f"[SKIP] no benchmark graph files in {test_dir}")
            continue

        for run_algo in algos_to_run:
            cmd = [
                sys.executable,
                str(BENCHMARK_UND_SCRIPT),
                "--algo",
                run_algo,
                "--test",
                str(test_dir),
                "--results-dir",
                str(RESULTS_DIR),
                "--pop",
                str(pop),
                "--gens",
                str(gens),
                "--timeout",
                str(timeout),
                "--earlystop",
                str(earlystop),
                "--gnn-threshold",
                str(gnn_threshold),
            ]
            if result_tag is not None:
                cmd.extend(["--result-tag", result_tag])
            if gnn_hidden is not None:
                cmd.extend(["--gnn-hidden", str(gnn_hidden)])
            if quiet:
                cmd.append("--quiet")

            rc = run_command(cmd)
            if rc != 0:
                print(f"[WARN] benchmark failed for {test_dir} (algo={run_algo})")
                continue

            outputs.append(_results_path_for_tag(run_algo, "undirected", result_tag))

    return outputs


def run_directed(
    algo: str,
    pop: int,
    gens: int,
    timeout: int,
    earlystop: int,
    gnn_threshold: float,
    gnn_hidden: int | None,
    quiet: bool,
    include_pace: bool,
    synthetic_dir: Path,
    synthetic_only: bool,
    track: str,
) -> List[Path]:
    outputs: List[Path] = []
    test_targets: List[Tuple[Path, str | None]] = []
    if not synthetic_only:
        test_targets.append((DATA_DIR / "raw_directed", None))
    test_targets.extend(_expand_synthetic_track_targets(synthetic_dir, track=track))
    if include_pace:
        test_targets.append((DATA_DIR / "pace2022", None))

    if algo == "ALL":
        algos_to_run = ["BST", "IC", "MA", "KMA", "GNN-KMA", "GNN-KMA-2"]
    elif algo == "PUREALGO":
        algos_to_run = ["BST", "IC", "MA", "KMA"]
    else:
        algos_to_run = [algo]

    for test_dir, result_tag in test_targets:
        if not has_graph_files(test_dir):
            print(f"[SKIP] no benchmark graph files in {test_dir}")
            continue

        for run_algo in algos_to_run:
            cmd = [
                sys.executable,
                str(BENCHMARK_DIR_SCRIPT),
                "--algo",
                run_algo,
                "--test",
                str(test_dir),
                "--results-dir",
                str(RESULTS_DIR),
                "--pop",
                str(pop),
                "--gens",
                str(gens),
                "--timeout",
                str(timeout),
                "--earlystop",
                str(earlystop),
                "--gnn-threshold",
                str(gnn_threshold),
            ]
            if result_tag is not None:
                cmd.extend(["--result-tag", result_tag])
            if gnn_hidden is not None:
                cmd.extend(["--gnn-hidden", str(gnn_hidden)])
            if quiet:
                cmd.append("--quiet")

            rc = run_command(cmd)
            if rc != 0:
                print(f"[WARN] benchmark failed for {test_dir} (algo={run_algo})")
                continue

            outputs.append(_results_path_for_tag(run_algo, "directed", result_tag))

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Existing-dataset-only FVS benchmark runner",
        epilog="""
    This pipeline never downloads or generates datasets.
    Example: --total-undirected 100 will check data/synthetic/undirected/ and
    exit if fewer than 100 files exist.
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["all", "undirected", "directed"],
        type=str.lower,
        default="all",
        help="Which benchmark family to run",
    )
    parser.add_argument(
        "--algo",
        choices=["BST", "IC", "MA", "KMA", "GNN-KMA", "GNN-KMA-2", "ALL", "PUREALGO"],
        type=str.upper,
        default="ALL",
        help="Algorithm selection forwarded to benchmark scripts (PUREALGO: BST, IC, MA, KMA)",
    )
    parser.add_argument("--pop", type=int, default=20, help="Population size for MA/KMA/GNN-KMA variants")
    parser.add_argument("--gens", "--gen", type=int, default=100, help="Max generations for MA/KMA/GNN-KMA variants")
    parser.add_argument("--timeout", type=int, default=600, help="Hard wall-clock timeout in seconds for MA/KMA/GNN-KMA variants")
    parser.add_argument("--earlystop", type=int, default=20, help="Patience / early-stopping generations without improvement (default: 20)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="[GNN-KMA/GNN-KMA-2 only] Probability threshold for GNN candidate selection (default: 0.2)",
    )
    parser.add_argument(
        "--gnn-hidden",
        type=int,
        default=None,
        help="[GNN-KMA/GNN-KMA-2 only] Optional hidden dimension override for GNN weights",
    )
    parser.add_argument(
        "--include-pace",
        action="store_true",
        help="Include data/pace2022 in directed benchmark runs",
    )
    parser.add_argument(
        "--skip-real",
        action="store_true",
        help="Deprecated compatibility flag (ignored; pipeline does not download)",
    )
    parser.add_argument(
        "--skip-synthetic",
        action="store_true",
        help="Deprecated compatibility flag (ignored; pipeline does not generate)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Deprecated compatibility flag (ignored by this pipeline)",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only validate dataset counts, do not run benchmarks",
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
        default=None,
        help="Target synthetic undirected dataset count (defaults to all available files)",
    )
    parser.add_argument(
        "--total-directed",
        type=int,
        default=None,
        help="Target synthetic directed dataset count (defaults to all available files)",
    )
    parser.add_argument(
        "--exact-ratio",
        type=float,
        default=0.5,
        help="Fraction of each category allocated to exact_track (% split per category preserved)",
    )
    parser.add_argument(
        "--track",
        choices=["both", "exact", "heuristic"],
        type=str.lower,
        default="both",
        help="Synthetic track selection: both, exact, or heuristic",
    )

    args = parser.parse_args()

    # If totals not specified, use all available files
    if args.total_undirected is None:
        args.total_undirected = count_existing_txt_files("undirected")
    if args.total_directed is None:
        args.total_directed = count_existing_txt_files("directed")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  FVS Benchmark Pipeline - Existing Dataset Mode")
    print(f"{'='*70}")
    print(f"  Mode: {args.mode} | Algo: {args.algo}")
    print(f"  Track: {args.track}")
    print(f"  Target: {args.total_undirected} undirected, {args.total_directed} directed")
    print(f"  Exact ratio: {args.exact_ratio:.2f}")
    print(f"{'='*70}\n")

    if args.skip_real or args.skip_synthetic or args.force:
        print("[INFO] Compatibility flags detected (--skip-real/--skip-synthetic/--force); ignored.")

    if not (0.0 <= args.threshold <= 1.0):
        raise ValueError("--threshold must be in [0.0, 1.0]")
    if args.timeout <= 0:
        raise ValueError("--timeout must be a positive integer")
    if args.earlystop <= 0:
        raise ValueError("--earlystop must be a positive integer")
    if args.gnn_hidden is not None and args.gnn_hidden <= 0:
        raise ValueError("--gnn-hidden must be a positive integer")

    # Validate and adjust counts to available data
    args.total_undirected, args.total_directed = validate_existing_datasets(
        mode=args.mode,
        total_undirected=args.total_undirected,
        total_directed=args.total_directed,
    )

    if args.prepare_only:
        print("\n[DONE] dataset validation finished (prepare-only mode)\n")
        return

    selected_undirected_dir = DATA_DIR / "synthetic" / "undirected"
    selected_directed_dir = DATA_DIR / "synthetic" / "directed"
    if args.mode in {"all", "undirected"}:
        selected_undirected_dir = select_synthetic_subset(
            family="undirected",
            total=args.total_undirected,
            exact_ratio=args.exact_ratio,
            track=args.track,
        )
    if args.mode in {"all", "directed"}:
        selected_directed_dir = select_synthetic_subset(
            family="directed",
            total=args.total_directed,
            exact_ratio=args.exact_ratio,
            track=args.track,
        )

    all_outputs: List[Path] = []

    if args.mode in {"all", "undirected"}:
        all_outputs.extend(
            run_undirected(
                args.algo,
                args.pop,
                args.gens,
                args.timeout,
                args.earlystop,
                args.threshold,
                args.gnn_hidden,
                args.quiet,
                selected_undirected_dir,
                args.synthetic_only,
                args.track,
            )
        )

    if args.mode in {"all", "directed"}:
        all_outputs.extend(
            run_directed(
                args.algo,
                args.pop,
                args.gens,
                args.timeout,
                args.earlystop,
                args.threshold,
                args.gnn_hidden,
                args.quiet,
                args.include_pace,
                selected_directed_dir,
                args.synthetic_only,
                args.track,
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
