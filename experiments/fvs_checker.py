#!/usr/bin/env python3
"""
Brute-force FVS checker on synthetic_selected exact-track datasets.

Runs both:
- undirected exact-track instances
- directed exact-track instances

Outputs:
- results/undirected_BRUTE_FORCE.csv
- results/directed_BRUTE_FORCE.csv

CSV schema (uniform):
file,n,m,FVS_size,runtime,validity
"""

from __future__ import annotations

import argparse
import csv
import itertools
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "data" / "synthetic_selected"
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
                v = int(tok) - 1
                edges.append((u, v))

    if n is None:
        raise ValueError(f"No METIS header in {filepath}")

    return n, edges


def _parse_edge_list(filepath: Path) -> Tuple[int, List[Tuple[int, int]]]:
    edges: List[Tuple[int, int]] = []
    n_hint: Optional[int] = None

    with filepath.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith(("#", "%", "c ")):
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

            if not parts[0].lstrip("-").isdigit():
                continue

            if len(parts) < 2:
                continue

            u = int(parts[0])
            v = int(parts[1])
            edges.append((u, v))

    if not edges:
        raise ValueError(f"No edges found in {filepath}")

    vertices = set()
    for u, v in edges:
        vertices.add(u)
        vertices.add(v)

    min_v = min(vertices)
    max_v = max(vertices)

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

    return _parse_edge_list(filepath)


def is_acyclic_directed(n: int, edges: Sequence[Tuple[int, int]], removed: Set[int]) -> bool:
    remaining = [v for v in range(n) if v not in removed]
    indeg = {v: 0 for v in remaining}
    out = {v: [] for v in remaining}

    for u, v in edges:
        if u in removed or v in removed:
            continue
        out[u].append(v)
        indeg[v] += 1

    queue = [v for v in remaining if indeg[v] == 0]
    seen = 0
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        seen += 1
        for w in out[u]:
            indeg[w] -= 1
            if indeg[w] == 0:
                queue.append(w)

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

            removed = set(subset)
            if check_fn(n, edges, removed):
                return str(len(subset)), time.perf_counter() - start, "True"

            if (time.perf_counter() - start) > timeout_seconds:
                return "TIMEOUT", time.perf_counter() - start, "TIMEOUT"

    return "ERROR", time.perf_counter() - start, "False"


def write_results(csv_path: Path, rows: List[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "n", "m", "FVS_size", "runtime", "validity"])
        writer.writeheader()
        writer.writerows(rows)


def run_family(input_root: Path, results_dir: Path, directed: bool, timeout_seconds: float) -> Path:
    family = "directed" if directed else "undirected"
    exact_root = input_root / family / "exact_track"
    files = collect_graph_files(exact_root)

    if not files:
        raise RuntimeError(f"No files found in {exact_root}")

    rows: List[dict] = []
    print(f"[INFO] Running brute-force for {family}: {len(files)} file(s)")

    for fp in files:
        rel_name = fp.relative_to(exact_root).as_posix()
        try:
            n, edges = parse_graph(fp, directed=directed)
            fvs_size, runtime_s, validity = brute_force_fvs(n, edges, directed, timeout_seconds)
        except Exception as ex:
            n, edges = 0, []
            fvs_size = "ERROR"
            runtime_s = 0.0
            validity = f"ERROR:{ex}"

        row = {
            "file": rel_name,
            "n": n,
            "m": len(edges),
            "FVS_size": fvs_size,
            "runtime": round(runtime_s, 6),
            "validity": validity,
        }
        rows.append(row)
        print(f"  {family[:3].upper()} | {rel_name} | FVS={fvs_size} | t={runtime_s:.3f}s | validity={validity}")

    out_csv = results_dir / f"{family}_BRUTE_FORCE.csv"
    write_results(out_csv, rows)
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Brute-force FVS checker for synthetic_selected exact-track datasets")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT, help="Root dataset folder (default: data/synthetic_selected)")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Results folder (default: results)")
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-instance timeout in seconds (default: 60)")
    args = parser.parse_args()

    out_und = run_family(args.input_root, args.results_dir, directed=False, timeout_seconds=args.timeout)
    out_dir = run_family(args.input_root, args.results_dir, directed=True, timeout_seconds=args.timeout)

    print("\n[OK] Brute-force checking complete")
    print(f"  - {out_und}")
    print(f"  - {out_dir}")


if __name__ == "__main__":
    main()
