#!/usr/bin/env python3
"""
Pure brute-force FVS runner for synthetic exact-track datasets.

Behavior:
- Runs all exact-track testcases for undirected and directed.
- Does not check IC/BST CSV existence.
- Uses 60s timeout per testcase by default.

Outputs:
- results/undirected_brute_force.csv
- results/directed_brute_force.csv

CSV columns:
file,n,m,FVS_size,runtime,validity
"""

from __future__ import annotations

import argparse
import csv
import itertools
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "data" / "synthetic"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
GRAPH_EXTENSIONS = {".txt", ".gr", ".edges", ".graph", ".dimacs", ".mtx"}


def collect_graph_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() in GRAPH_EXTENSIONS:
            files.append(p)
        elif p.suffix == "" and not p.name.startswith("."):
            files.append(p)
    return files


def parse_metis_directed(filepath: Path) -> Tuple[int, List[Tuple[int, int]]]:
    edges: List[Tuple[int, int]] = []
    n: Optional[int] = None
    vertex_idx = 0

    with filepath.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            s = line.strip()
            if not s or s.startswith("%"):
                continue

            parts = s.split()
            if n is None:
                if len(parts) < 3:
                    raise ValueError(f"Invalid METIS header at line {line_num}: {s}")
                n = int(parts[0])
                continue

            u = vertex_idx
            vertex_idx += 1
            for tok in parts:
                edges.append((u, int(tok) - 1))

    if n is None:
        raise ValueError(f"No METIS header in {filepath}")

    return n, edges


def parse_edge_list(filepath: Path) -> Tuple[int, List[Tuple[int, int]]]:
    edges: List[Tuple[int, int]] = []
    n_hint: Optional[int] = None

    with filepath.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith(("#", "%", "c ")):
                continue

            parts = s.split()
            if not parts:
                continue

            if parts[0].lower() == "p" and len(parts) >= 4:
                try:
                    n_hint = int(parts[2])
                except ValueError:
                    pass
                continue

            if not parts[0].lstrip("-").isdigit() or len(parts) < 2:
                continue

            edges.append((int(parts[0]), int(parts[1])))

    if not edges:
        raise ValueError(f"No edges found in {filepath}")

    verts = {v for e in edges for v in e}
    min_v, max_v = min(verts), max(verts)

    if min_v == 1:
        edges = [(u - 1, v - 1) for u, v in edges]
        max_v -= 1

    n = n_hint if n_hint is not None else max_v + 1
    n = max(n, max_v + 1)
    return n, edges


def parse_graph(filepath: Path, directed: bool) -> Tuple[int, List[Tuple[int, int]]]:
    if directed:
        first_data_line = None
        with filepath.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith(("#", "%", "c ")):
                    first_data_line = s
                    break

        if first_data_line:
            parts = first_data_line.split()
            if len(parts) == 3 and all(p.lstrip("-").isdigit() for p in parts):
                return parse_metis_directed(filepath)

    return parse_edge_list(filepath)


def is_acyclic_directed(n: int, edges: Sequence[Tuple[int, int]], removed: Set[int]) -> bool:
    remaining = [v for v in range(n) if v not in removed]
    indeg = {v: 0 for v in remaining}
    out = {v: [] for v in remaining}

    for u, v in edges:
        if u in removed or v in removed:
            continue
        out[u].append(v)
        indeg[v] += 1

    q = [v for v in remaining if indeg[v] == 0]
    seen = 0
    head = 0
    while head < len(q):
        u = q[head]
        head += 1
        seen += 1
        for w in out[u]:
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)

    return seen == len(remaining)


def is_acyclic_undirected(n: int, edges: Sequence[Tuple[int, int]], removed: Set[int]) -> bool:
    adj = {v: set() for v in range(n) if v not in removed}
    for u, v in edges:
        if u in removed or v in removed:
            continue
        adj[u].add(v)
        adj[v].add(u)

    visited: Set[int] = set()
    for start in adj:
        if start in visited:
            continue

        stack = [(start, -1)]
        while stack:
            u, parent = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            for w in adj[u]:
                if w == parent:
                    continue
                if w in visited:
                    return False
                stack.append((w, u))

    return True


def brute_force_fvs(
    n: int,
    edges: Sequence[Tuple[int, int]],
    directed: bool,
    timeout_seconds: float,
) -> Tuple[str, float, str]:
    start = time.perf_counter()

    candidate_vertices = sorted({v for e in edges for v in e})
    if not candidate_vertices:
        return "0", 0.0, "True"

    check_fn = is_acyclic_directed if directed else is_acyclic_undirected

    checks = 0
    for k in range(len(candidate_vertices) + 1):
        for subset in itertools.combinations(candidate_vertices, k):
            checks += 1
            if checks % 2048 == 0 and (time.perf_counter() - start) > timeout_seconds:
                return "TIMEOUT", time.perf_counter() - start, "TIMEOUT"

            if check_fn(n, edges, set(subset)):
                return str(len(subset)), time.perf_counter() - start, "True"

            if (time.perf_counter() - start) > timeout_seconds:
                return "TIMEOUT", time.perf_counter() - start, "TIMEOUT"

    return "ERROR", time.perf_counter() - start, "False"


def write_rows(csv_path: Path, rows: List[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "n", "m", "FVS_size", "runtime", "validity"])
        writer.writeheader()
        writer.writerows(rows)


def load_existing_rows(csv_path: Path) -> Dict[str, dict]:
    if not csv_path.exists():
        return {}

    rows: Dict[str, dict] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = str(row.get("file", "")).strip()
            if key:
                rows[key] = row
    return rows


def is_completed_non_timeout(row: Optional[dict]) -> bool:
    if not row:
        return False
    fvs_value = str(row.get("FVS_size", "")).strip().upper()
    return fvs_value not in {"", "TIMEOUT", "ERROR"}


def persist_rows(csv_path: Path, rows_by_file: Dict[str, dict]) -> None:
    ordered = [rows_by_file[name] for name in sorted(rows_by_file.keys())]
    write_rows(csv_path, ordered)


def run_family(input_root: Path, results_dir: Path, directed: bool, timeout_seconds: float) -> Path:
    family = "directed" if directed else "undirected"
    exact_root = input_root / family / "exact_track"
    files = collect_graph_files(exact_root) if exact_root.exists() else []
    out_csv = results_dir / f"{family}_brute_force.csv"
    rows_by_file = load_existing_rows(out_csv)

    if not files:
        print(f"[WARN] No files found in {exact_root}; writing empty report")
        write_rows(out_csv, [])
        return out_csv

    print(f"[INFO] Brute force on {family}: {len(files)} file(s)")
    for fp in files:
        rel_name = fp.relative_to(exact_root).as_posix()
        existing = rows_by_file.get(fp.name)
        if is_completed_non_timeout(existing):
            print(f"  {family[:3].upper()} | {rel_name} | [SKIPPED] Already stored result")
            continue

        try:
            n, edges = parse_graph(fp, directed=directed)
            fvs_size, runtime_s, validity = brute_force_fvs(n, edges, directed, timeout_seconds)
        except Exception as ex:
            n, edges = 0, []
            fvs_size = "ERROR"
            runtime_s = 0.0
            validity = f"ERROR:{ex}"

        row = {
            "file": fp.name,
            "n": n,
            "m": len(edges),
            "FVS_size": fvs_size,
            "runtime": round(runtime_s, 6),
            "validity": validity,
        }
        rows_by_file[fp.name] = row
        # Save progress after each testcase so interrupted runs can resume.
        persist_rows(out_csv, rows_by_file)
        print(f"  {family[:3].upper()} | {rel_name} | FVS={fvs_size} | t={runtime_s:.3f}s | validity={validity}")

    persist_rows(out_csv, rows_by_file)
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Pure brute-force FVS runner")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-instance timeout in seconds (default: 60)")
    args = parser.parse_args()

    out_und = run_family(args.input_root, args.results_dir, directed=False, timeout_seconds=args.timeout)
    out_dir = run_family(args.input_root, args.results_dir, directed=True, timeout_seconds=args.timeout)

    print("\n[OK] Brute-force run complete")
    print(f"  - {out_und}")
    print(f"  - {out_dir}")


if __name__ == "__main__":
    main()
