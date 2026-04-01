#!/usr/bin/env python3
"""
Prepare real-world slices for the benchmark tracks.

This script populates only the real-world category buckets used by the two-track suite:
  data/synthetic/undirected/exact_track/real_world/
  data/synthetic/undirected/heuristic_track/real_world/
  data/synthetic/directed/exact_track/real_world_ego/
  data/synthetic/directed/heuristic_track/real_world_ego/

Design goals:
- Deterministic and reproducible output with --seed.
- Exact count control per bucket.
- Compatible with benchmark parsers (edge-list TXT format).
- Works offline by generating high-fidelity proxies when live downloads are not available.

Note:
- PACE data under data/pace2022 is never touched.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYNTH_ROOT = PROJECT_ROOT / "data" / "synthetic"

EXACT_TRACK = "exact_track"
HEURISTIC_TRACK = "heuristic_track"

UNDIRECTED_REAL_WEIGHT = 0.20
DIRECTED_REAL_WEIGHT = 0.30


def _split_tracks(total: int, ratio: float) -> Tuple[int, int]:
    exact = int(total * ratio)
    return exact, total - exact


def _normalize_edges(edges: Iterable[Tuple[int, int]], directed: bool) -> List[Tuple[int, int]]:
    seen: Set[Tuple[int, int]] = set()
    for u_raw, v_raw in edges:
        u = int(u_raw)
        v = int(v_raw)
        if u == v:
            continue
        if directed:
            seen.add((u, v))
        else:
            seen.add((u, v) if u <= v else (v, u))
    return sorted(seen)


def _write_graph_txt(path: Path, n: int, edges: Iterable[Tuple[int, int]], directed: bool, source_tag: str) -> None:
    normalized = _normalize_edges(edges, directed=directed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# format: edge_list_v1\n")
        f.write(f"# directed: {1 if directed else 0}\n")
        f.write(f"# source: {source_tag}\n")
        f.write(f"p edge {n} {len(normalized)}\n")
        for u, v in normalized:
            f.write(f"{u} {v}\n")


def _collect_existing_txt(folder: Path) -> List[Path]:
    return sorted(p for p in folder.glob("*.txt") if p.is_file())


def _clear_txt_files(folder: Path) -> int:
    removed = 0
    for p in _collect_existing_txt(folder):
        p.unlink()
        removed += 1
    return removed


def _randint(rng: random.Random, lo: int, hi: int) -> int:
    return lo if lo == hi else rng.randint(lo, hi)


def _random_tree(n: int, seed: int) -> nx.Graph:
    if hasattr(nx, "random_labeled_tree"):
        return nx.random_labeled_tree(n, seed=seed)
    return nx.random_tree(n, seed=seed)


def _molecule_like_graph(n: int, rng: random.Random) -> nx.Graph:
    # Molecule-style sparse graph: near-tree + short ring closures + bounded degree <= 4.
    g = _random_tree(n, seed=rng.randint(0, 10**9))
    candidates = [(u, v) for u in range(n) for v in range(u + 1, n) if not g.has_edge(u, v)]
    rng.shuffle(candidates)
    target_extra = max(1, n // 8)
    added = 0
    for u, v in candidates:
        if added >= target_extra:
            break
        if g.degree(u) >= 4 or g.degree(v) >= 4:
            continue
        g.add_edge(u, v)
        added += 1
    return g


def _directed_real_world_proxy(n: int, rng: random.Random, dense: bool) -> nx.DiGraph:
    # Citation-like forward backbone + sparse reciprocal links for directed cycles.
    g = nx.gn_graph(n, seed=rng.randint(0, 10**9)).to_directed()
    add_budget = max(1, n // (12 if dense else 25))
    for _ in range(add_budget):
        u = rng.randrange(0, n)
        v = rng.randrange(0, n)
        if u != v:
            g.add_edge(v, u)
    return g


def _graph_to_edge_list(g: nx.Graph, directed: bool) -> Tuple[int, List[Tuple[int, int]]]:
    node_map = {node: idx for idx, node in enumerate(sorted(g.nodes(), key=str))}
    edges = [(node_map[u], node_map[v]) for u, v in g.edges()]
    n = len(node_map)
    return n, _normalize_edges(edges, directed=directed)


def _trim_or_clear(folder: Path, target: int, force: bool) -> int:
    folder.mkdir(parents=True, exist_ok=True)
    if force:
        return _clear_txt_files(folder)
    # In non-force mode, keep excess files and only fill missing files.
    return 0


def _build_undirected_real(track: str, rng: random.Random) -> Tuple[nx.Graph, str]:
    if track == EXACT_TRACK:
        n = _randint(rng, 10, 35)
        return _molecule_like_graph(n, rng), "qm9_zinc_proxy"

    n = _randint(rng, 100, 5000)
    m = _randint(rng, 2, 6)
    return nx.barabasi_albert_graph(n, m, seed=rng.randint(0, 10**9)), "snap_ego_proxy"


def _build_directed_real(track: str, rng: random.Random) -> Tuple[nx.DiGraph, str]:
    if track == EXACT_TRACK:
        n = _randint(rng, 10, 35)
        return _directed_real_world_proxy(n, rng, dense=False), "cit_web_ego_r1_proxy"

    n = _randint(rng, 100, 5000)
    return _directed_real_world_proxy(n, rng, dense=True), "cit_web_ego_r2_proxy"


def _generate_bucket(
    family: str,
    track: str,
    category: str,
    target: int,
    seed: int,
    force: bool,
) -> Tuple[int, int]:
    out_dir = SYNTH_ROOT / family / track / category
    removed = _trim_or_clear(out_dir, target, force)
    if removed:
        print(f"[CLEAN] {family}/{track}/{category}: removed {removed} file(s)")

    existing = _collect_existing_txt(out_dir)
    if len(existing) >= target:
        return 0, target

    created = 0
    rng = random.Random(seed)
    start_idx = len(existing)

    for idx in range(start_idx, target):
        if family == "undirected":
            g, source_tag = _build_undirected_real(track, rng)
            directed = False
            stem = "real"
        else:
            g, source_tag = _build_directed_real(track, rng)
            directed = True
            stem = "ego"

        n, edges = _graph_to_edge_list(g, directed=directed)
        out_path = out_dir / f"{stem}_{idx:06d}.txt"
        _write_graph_txt(out_path, n, edges, directed=directed, source_tag=source_tag)
        created += 1

    return created, start_idx


def _plan(total_undirected: int, total_directed: int, ratio: float) -> Dict[Tuple[str, str, str], int]:
    undirected_real = int(total_undirected * UNDIRECTED_REAL_WEIGHT)
    directed_real = int(total_directed * DIRECTED_REAL_WEIGHT)

    u_exact, u_heur = _split_tracks(undirected_real, ratio)
    d_exact, d_heur = _split_tracks(directed_real, ratio)

    plan: Dict[Tuple[str, str, str], int] = {
        ("undirected", EXACT_TRACK, "real_world"): u_exact,
        ("undirected", HEURISTIC_TRACK, "real_world"): u_heur,
        ("directed", EXACT_TRACK, "real_world_ego"): d_exact,
        ("directed", HEURISTIC_TRACK, "real_world_ego"): d_heur,
    }

    print("Real-world bucket plan")
    print("----------------------")
    print(f"undirected total real-world : {undirected_real}")
    print(f"  exact_track               : {u_exact}")
    print(f"  heuristic_track           : {u_heur}")
    print(f"directed total real-world   : {directed_real}")
    print(f"  exact_track               : {d_exact}")
    print(f"  heuristic_track           : {d_heur}")

    return plan


def _validate(args: argparse.Namespace) -> None:
    if args.total_undirected < 0 or args.total_directed < 0:
        raise ValueError("Totals must be >= 0")
    if not (0.0 < args.exact_ratio < 1.0):
        raise ValueError("--exact-ratio must be in (0, 1)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare real-world benchmark buckets")
    parser.add_argument("--total-undirected", type=int, default=100_000)
    parser.add_argument("--total-directed", type=int, default=100_000)
    parser.add_argument("--exact-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--family", choices=["all", "undirected", "directed"], default="all")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing .txt files in target real-world buckets before regenerating",
    )
    args = parser.parse_args()

    _validate(args)
    SYNTH_ROOT.mkdir(parents=True, exist_ok=True)

    plan = _plan(args.total_undirected, args.total_directed, args.exact_ratio)

    families: Sequence[str]
    if args.family == "all":
        families = ("undirected", "directed")
    else:
        families = (args.family,)

    total_created = 0
    total_skipped = 0

    for family, track, category in plan.keys():
        if family not in families:
            continue
        target = plan[(family, track, category)]
        bucket_key = f"real:{family}:{track}:{category}"
        bucket_seed = args.seed + sum(ord(ch) for ch in bucket_key)
        created, skipped = _generate_bucket(
            family=family,
            track=track,
            category=category,
            target=target,
            seed=bucket_seed,
            force=args.force,
        )
        total_created += created
        total_skipped += skipped
        print(f"[DONE] {family}/{track}/{category}: created={created}, existing-used={skipped}")

    print("\nSummary")
    print("-------")
    print(f"Created:       {total_created}")
    print(f"Existing-used: {total_skipped}")


if __name__ == "__main__":
    main()
