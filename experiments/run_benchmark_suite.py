#!/usr/bin/env python3
"""
Run the full benchmark suite in one command with resume support.

Default behavior:
- exact_track      -> BST, IC, MA, KMA, GNN-KMA
- heuristic_track  -> MA, KMA, GNN-KMA
- families         -> undirected + directed

For each (family, track, algorithm), results are saved to a dedicated CSV.
If a CSV already contains a file row, that file is skipped for that algorithm unless --rerun is used.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "synthetic"
RESULTS_ROOT = PROJECT_ROOT / "results" / "suite"

UND_BENCH = PROJECT_ROOT / "experiments" / "benchmark_undirected.py"
DIR_BENCH = PROJECT_ROOT / "experiments" / "benchmark_directed.py"

GRAPH_EXTENSIONS = {".txt", ".gr", ".edges", ".graph", ".dimacs", ".mtx"}

EXACT_ALGOS = ["BST", "IC", "MA", "KMA", "GNN-KMA", "GNN-KMA-2"]
HEURISTIC_ALGOS = ["MA", "KMA", "GNN-KMA", "GNN-KMA-2"]


@dataclass
class Task:
    family: str
    track: str
    algo: str


def collect_graph_files(root: Path) -> List[Path]:
    if not root.exists() or not root.is_dir():
        return []

    files: List[Path] = []
    for item in sorted(root.rglob("*")):
        if not item.is_file():
            continue
        if item.suffix.lower() in GRAPH_EXTENSIONS:
            files.append(item)
        elif item.suffix == "" and not item.name.startswith("."):
            files.append(item)
    return files


def run_command(cmd: Sequence[str]) -> int:
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return proc.returncode


def read_done_file_set(csv_path: Path) -> Set[str]:
    if not csv_path.exists():
        return set()

    done: Set[str] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_path = row.get("file_path", "").strip()
            if file_path:
                done.add(file_path)
    return done


def ensure_output_header(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "file_path",
                "n",
                "m",
                "family",
                "track",
                "algorithm",
                "fvs_size",
                "runtime_ms",
                "valid",
                "status",
                "executed_at",
            ],
        )
        writer.writeheader()


def extract_algo_fields(row: Dict[str, str], algo: str) -> Dict[str, str]:
    size_key = f"{algo}_size"
    ms_key = f"{algo}_ms"
    valid_key = f"{algo}_valid"

    fvs_size = row.get(size_key, "")
    runtime_ms = row.get(ms_key, "")
    valid = row.get(valid_key, "")

    status = "OK"
    marker = str(fvs_size).upper()
    if marker in {"TIMEOUT", "ERROR"}:
        status = marker

    return {
        "fvs_size": fvs_size,
        "runtime_ms": runtime_ms,
        "valid": valid,
        "status": status,
    }


def append_result_row(csv_path: Path, result_row: Dict[str, str]) -> None:
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "file_path",
                "n",
                "m",
                "family",
                "track",
                "algorithm",
                "fvs_size",
                "runtime_ms",
                "valid",
                "status",
                "executed_at",
            ],
        )
        writer.writerow(result_row)


def run_single_file(
    family: str,
    algo: str,
    file_path: Path,
    pop: int,
    gens: int,
    timeout: int,
    earlystop: int,
    quiet: bool,
    temp_csv: Path,
) -> Dict[str, str]:
    bench_script = UND_BENCH if family == "undirected" else DIR_BENCH

    cmd = [
        sys.executable,
        str(bench_script),
        "--algo",
        algo,
        "--test",
        str(file_path),
        "--output",
        str(temp_csv),
        "--pop",
        str(pop),
        "--gens",
        str(gens),
        "--timeout",
        str(timeout),
        "--earlystop",
        str(earlystop),
    ]
    if quiet:
        cmd.append("--quiet")

    rc = run_command(cmd)
    if rc != 0:
        return {
            "n": "",
            "m": "",
            "fvs_size": "ERROR",
            "runtime_ms": "ERROR",
            "valid": "False",
            "status": f"BENCH_EXIT_{rc}",
        }

    if not temp_csv.exists() or temp_csv.stat().st_size == 0:
        return {
            "n": "",
            "m": "",
            "fvs_size": "ERROR",
            "runtime_ms": "ERROR",
            "valid": "False",
            "status": "NO_RESULT",
        }

    with temp_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return {
            "n": "",
            "m": "",
            "fvs_size": "ERROR",
            "runtime_ms": "ERROR",
            "valid": "False",
            "status": "NO_RESULT",
        }

    row = rows[0]
    algo_fields = extract_algo_fields(row, algo)
    return {
        "n": row.get("n", ""),
        "m": row.get("m", ""),
        **algo_fields,
    }


def build_tasks(mode: str, profile: str) -> List[Task]:
    tasks: List[Task] = []

    if profile == "requested":
        # Requested combo:
        # 1) all directed algorithms on exact_track
        # 2) heuristic undirected algorithms on heuristic_track
        for algo in EXACT_ALGOS:
            tasks.append(Task(family="directed", track="exact_track", algo=algo))
        for algo in HEURISTIC_ALGOS:
            tasks.append(Task(family="undirected", track="heuristic_track", algo=algo))
    else:
        families = ["undirected", "directed"] if mode == "all" else [mode]
        for fam in families:
            for algo in EXACT_ALGOS:
                tasks.append(Task(family=fam, track="exact_track", algo=algo))
            for algo in HEURISTIC_ALGOS:
                tasks.append(Task(family=fam, track="heuristic_track", algo=algo))

    if mode == "all":
        return tasks
    return [t for t in tasks if t.family == mode]


def task_input_dir(task: Task) -> Path:
    return DATA_ROOT / task.family / task.track


def task_output_csv(task: Task) -> Path:
    return RESULTS_ROOT / f"{task.family}_{task.track}_{task.algo}.csv"


def run_task(
    task: Task,
    pop: int,
    gens: int,
    timeout: int,
    earlystop: int,
    quiet: bool,
    rerun: bool,
    max_files: int,
) -> Dict[str, int]:
    input_dir = task_input_dir(task)
    output_csv = task_output_csv(task)

    files = collect_graph_files(input_dir)
    if max_files > 0:
        files = files[:max_files]
    ensure_output_header(output_csv)
    done_set = set() if rerun else read_done_file_set(output_csv)

    temp_csv = RESULTS_ROOT / ".tmp_single_result.csv"
    if temp_csv.exists():
        temp_csv.unlink()

    created = 0
    skipped = 0
    failed = 0

    print(f"\n[START] {task.family}/{task.track}/{task.algo}")
    print(f"        input: {input_dir}")
    print(f"        output: {output_csv}")
    print(f"        files found: {len(files)}")

    for fp in files:
        fp_str = str(fp)
        if fp_str in done_set:
            skipped += 1
            continue

        if temp_csv.exists():
            temp_csv.unlink()

        result = run_single_file(
            family=task.family,
            algo=task.algo,
            file_path=fp,
            pop=pop,
            gens=gens,
            timeout=timeout,
            earlystop=earlystop,
            quiet=quiet,
            temp_csv=temp_csv,
        )

        now = datetime.now().isoformat(timespec="seconds")
        out_row = {
            "file": fp.name,
            "file_path": fp_str,
            "n": result.get("n", ""),
            "m": result.get("m", ""),
            "family": task.family,
            "track": task.track,
            "algorithm": task.algo,
            "fvs_size": result.get("fvs_size", ""),
            "runtime_ms": result.get("runtime_ms", ""),
            "valid": result.get("valid", ""),
            "status": result.get("status", ""),
            "executed_at": now,
        }
        append_result_row(output_csv, out_row)
        created += 1

        if str(out_row["status"]).startswith("BENCH_EXIT") or out_row["status"] in {"NO_RESULT", "ERROR"}:
            failed += 1

    if temp_csv.exists():
        temp_csv.unlink()

    print(f"[DONE] {task.family}/{task.track}/{task.algo}: ran={created}, skipped={skipped}, failed={failed}")
    return {"ran": created, "skipped": skipped, "failed": failed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full benchmark suite with resume support")
    parser.add_argument(
        "--mode",
        choices=["all", "undirected", "directed"],
        default="all",
        help="Which family to run",
    )
    parser.add_argument(
        "--profile",
        choices=["requested", "full"],
        default="requested",
        help="Task profile: requested (directed exact all + undirected heuristic only) or full",
    )
    parser.add_argument("--pop", type=int, default=20, help="Population size for MA/KMA/GNN-KMA variants")
    parser.add_argument("--gens", type=int, default=100, help="Max generations for MA/KMA/GNN-KMA variants")
    parser.add_argument("--timeout", type=int, default=600, help="Hard wall-clock timeout in seconds for MA/KMA/GNN-KMA variants")
    parser.add_argument("--earlystop", type=int, default=20, help="Patience / early-stopping generations without improvement (default: 20)")
    parser.add_argument("--quiet", action="store_true", help="Forward quiet mode to benchmark scripts")
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Ignore existing CSV rows and rerun all files",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap of files per (family, track, algorithm) task; 0 means no cap",
    )
    args = parser.parse_args()

    if args.timeout <= 0:
        raise ValueError("--timeout must be a positive integer")
    if args.earlystop <= 0:
        raise ValueError("--earlystop must be a positive integer")

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(args.mode, args.profile)

    total_ran = 0
    total_skipped = 0
    total_failed = 0

    for task in tasks:
        stats = run_task(
            task,
            pop=args.pop,
            gens=args.gens,
            timeout=args.timeout,
            earlystop=args.earlystop,
            quiet=args.quiet,
            rerun=args.rerun,
            max_files=args.max_files,
        )
        total_ran += stats["ran"]
        total_skipped += stats["skipped"]
        total_failed += stats["failed"]

    print("\nSuite summary")
    print("-------------")
    print(f"Rows added:    {total_ran}")
    print(f"Rows skipped:  {total_skipped}")
    print(f"Rows failed:   {total_failed}")
    print(f"CSV output dir: {RESULTS_ROOT}")


if __name__ == "__main__":
    main()
