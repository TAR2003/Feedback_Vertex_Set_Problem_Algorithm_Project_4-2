#!/usr/bin/env python3
"""
One-command setup for the full 200k two-track benchmark input suite.

What this script does:
1) Optional cleanup of data subfolders (preserves data/pace2022 and all .py files).
2) Generates real-world buckets by track/category.
3) Generates remaining synthetic buckets by track/category.

Default split:
- Undirected total: 100,000
- Directed total:   100,000
- Exact/heuristic track split per category: 50/50

Examples:
  python data/setup_benchmark_inputs.py
  python data/setup_benchmark_inputs.py --total-undirected 2000 --total-directed 2000 --seed 7
  python data/setup_benchmark_inputs.py --no-clean
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

REAL_WORLD_SCRIPT = DATA_DIR / "download_real_world.py"
SYNTH_SCRIPT = DATA_DIR / "generate_synthetic.py"

PRESERVE_DIRS = {"pace2022", "__pycache__"}
PRESERVE_FILES = {".py", ".md", ".gitkeep"}


def _run(cmd: Sequence[str]) -> None:
    print("[RUN] " + " ".join(str(x) for x in cmd))
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(str(x) for x in cmd)}")


def _clean_data_subfolders() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    removed_paths = 0

    for child in DATA_DIR.iterdir():
        if child.is_dir():
            if child.name in PRESERVE_DIRS:
                continue
            shutil.rmtree(child)
            removed_paths += 1
            continue

        if child.is_file() and child.suffix.lower() in PRESERVE_FILES:
            continue

        if child.is_file():
            child.unlink()
            removed_paths += 1

    print(f"[CLEAN] Removed {removed_paths} data item(s), preserved: {sorted(PRESERVE_DIRS)} and script/docs files")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full benchmark inputs into data/")
    parser.add_argument("--total-undirected", type=int, default=100_000)
    parser.add_argument("--total-directed", type=int, default=100_000)
    parser.add_argument("--exact-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not clean data/ before setup",
    )
    parser.add_argument(
        "--family",
        choices=["all", "undirected", "directed"],
        default="all",
        help="Restrict setup to one graph family",
    )
    args = parser.parse_args()

    if not args.no_clean:
        _clean_data_subfolders()

    _run(
        [
            sys.executable,
            str(REAL_WORLD_SCRIPT),
            "--total-undirected",
            str(args.total_undirected),
            "--total-directed",
            str(args.total_directed),
            "--exact-ratio",
            str(args.exact_ratio),
            "--seed",
            str(args.seed),
            "--family",
            args.family,
            "--force",
        ]
    )

    _run(
        [
            sys.executable,
            str(SYNTH_SCRIPT),
            "--total-undirected",
            str(args.total_undirected),
            "--total-directed",
            str(args.total_directed),
            "--exact-ratio",
            str(args.exact_ratio),
            "--seed",
            str(args.seed),
            "--family",
            args.family,
        ]
    )

    print("\n[DONE] Benchmark input setup completed.")
    print(f"Output root: {DATA_DIR / 'synthetic'}")


if __name__ == "__main__":
    main()
