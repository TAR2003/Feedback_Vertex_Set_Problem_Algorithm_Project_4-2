#!/usr/bin/env python3
"""
Generate synthetic datasets for the two-track benchmark suite.

Layout:
  data/synthetic/<family>/<track>/<category>/*.txt

Tracks:
  - exact_track      (small graphs; all algorithms)
  - heuristic_track  (large graphs; MA/KME/HYBRID)
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


def _allocate_counts(total: int, weights: Dict[str, float]) -> Dict[str, int]:
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


def _split_tracks(total: int, exact_ratio: float) -> Tuple[int, int]:
    exact = int(total * exact_ratio)
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


def _write_graph_txt(path: Path, n: int, edges: Iterable[Tuple[int, int]], directed: bool) -> None:
    norm = _normalize_edges(edges, directed=directed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# format: edge_list_v1\n")
        f.write(f"# directed: {1 if directed else 0}\n")
        f.write(f"p edge {n} {len(norm)}\n")
        for u, v in norm:
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


def _directed_scale_free(n: int, rng: random.Random) -> nx.DiGraph:
    base = nx.scale_free_graph(n, seed=rng.randint(0, 10**9))
    d = nx.DiGraph()
    d.add_nodes_from(range(n))
    for u, v in base.edges():
        if u != v:
            d.add_edge(int(u), int(v))
    return d


def _directed_grid(rows: int, cols: int, rng: random.Random) -> nx.DiGraph:
    g = nx.grid_2d_graph(rows, cols)
    g = nx.convert_node_labels_to_integers(g)
    d = nx.DiGraph()
    d.add_nodes_from(g.nodes())
    for u, v in g.edges():
        if rng.random() < 0.5:
            d.add_edge(u, v)
        else:
            d.add_edge(v, u)
        if rng.random() < 0.08:
            d.add_edge(v, u)
    return d


def _directed_real_world_proxy(n: int, rng: random.Random, dense: bool) -> nx.DiGraph:
    g = nx.gn_graph(n, seed=rng.randint(0, 10**9)).to_directed()
    add_budget = max(1, n // (12 if dense else 25))
    for _ in range(add_budget):
        u = rng.randrange(0, n)
        v = rng.randrange(0, n)
        if u != v:
            g.add_edge(v, u)
    return g


def _build_undirected(category: str, track: str, rng: random.Random, idx: int) -> Tuple[nx.Graph, str]:
    if category == "real_world":
        if track == EXACT_TRACK:
            n = _randint(rng, 10, 35)
            return _molecule_like_graph(n, rng), "chem_proxy"
        n = _randint(rng, 100, 5000)
        m = _randint(rng, 2, 6)
        return nx.barabasi_albert_graph(n, m, seed=rng.randint(0, 10**9)), "ego_proxy"

    if category == "scale_free":
        if track == EXACT_TRACK:
            n = _randint(rng, 15, 35)
            m = _randint(rng, 2, 3)
        else:
            n = _randint(rng, 100, 2000)
            m = _randint(rng, 2, 5)
        return nx.barabasi_albert_graph(n, m, seed=rng.randint(0, 10**9)), "ba"

    if category == "small_world":
        if track == EXACT_TRACK:
            n = _randint(rng, 15, 35)
            k, p = 4, 0.1
        else:
            n = _randint(rng, 100, 1000)
            k, p = 6, 0.2
        return nx.watts_strogatz_graph(n, k, p, seed=rng.randint(0, 10**9)), "ws"

    if category == "random_er":
        if track == EXACT_TRACK:
            n = _randint(rng, 10, 30)
            p = rng.uniform(0.1, 0.4)
        else:
            n = _randint(rng, 50, 1000)
            p = rng.uniform(0.05, 0.2)
        return nx.erdos_renyi_graph(n, p, seed=rng.randint(0, 10**9)), "er"

    if category == "grids_trees":
        subtype = "grid" if idx % 2 == 0 else "tree"
        if subtype == "grid":
            if track == EXACT_TRACK:
                rows = _randint(rng, 3, 6)
                cols = _randint(rng, 3, 5)
            else:
                rows = _randint(rng, 10, 50)
                cols = _randint(rng, 10, 50)
            g = nx.grid_2d_graph(rows, cols)
            return nx.convert_node_labels_to_integers(g), "grid"

        if track == EXACT_TRACK:
            n = _randint(rng, 10, 35)
        else:
            n = _randint(rng, 100, 5000)
        return _random_tree(n, seed=rng.randint(0, 10**9)), "tree"

    raise ValueError(f"Unknown undirected category: {category}")


def _build_directed(category: str, track: str, rng: random.Random) -> Tuple[nx.DiGraph, str]:
    if category == "real_world_ego":
        if track == EXACT_TRACK:
            n = _randint(rng, 10, 35)
            return _directed_real_world_proxy(n, rng, dense=False), "ego_r1_proxy"
        n = _randint(rng, 100, 5000)
        return _directed_real_world_proxy(n, rng, dense=True), "ego_r2_proxy"

    if category == "scale_free":
        if track == EXACT_TRACK:
            n = _randint(rng, 15, 35)
        else:
            n = _randint(rng, 100, 2000)
        return _directed_scale_free(n, rng), "scale_free"

    if category == "random_er":
        if track == EXACT_TRACK:
            n = _randint(rng, 10, 30)
            p = rng.uniform(0.1, 0.4)
        else:
            n = _randint(rng, 50, 1000)
            p = rng.uniform(0.05, 0.2)
        return nx.erdos_renyi_graph(n, p, directed=True, seed=rng.randint(0, 10**9)), "er"

    if category == "directed_grids":
        if track == EXACT_TRACK:
            side = _randint(rng, 3, 6)
            rows, cols = side, side
        else:
            rows = _randint(rng, 10, 50)
            cols = _randint(rng, 10, 50)
        return _directed_grid(rows, cols, rng), "dgrid"

    if category == "dags":
        if track == EXACT_TRACK:
            n = _randint(rng, 10, 35)
        else:
            n = _randint(rng, 100, 5000)
        return nx.gn_graph(n, seed=rng.randint(0, 10**9)), "dag"

    raise ValueError(f"Unknown directed category: {category}")


def _graph_to_edge_list(g: nx.Graph, directed: bool) -> Tuple[int, List[Tuple[int, int]]]:
    node_map = {node: idx for idx, node in enumerate(sorted(g.nodes(), key=str))}
    edges = [(node_map[u], node_map[v]) for u, v in g.edges()]
    n = len(node_map)
    return n, _normalize_edges(edges, directed=directed)


def _generate_bucket(family: str, track: str, category: str, target: int, seed: int, force: bool) -> Tuple[int, int]:
    out_dir = SYNTH_ROOT / family / track / category
    out_dir.mkdir(parents=True, exist_ok=True)

    if force:
        removed = _clear_txt_files(out_dir)
        if removed:
            print(f"[CLEAN] {family}/{track}/{category}: removed {removed} old file(s)")

    existing = _collect_existing_txt(out_dir)
    if len(existing) > target:
        for stale in existing[target:]:
            stale.unlink()
        print(f"[TRIM] {family}/{track}/{category}: removed {len(existing) - target} excess file(s)")
        existing = existing[:target]

    if len(existing) >= target:
        return 0, target

    created = 0
    rng = random.Random(seed)
    start_idx = len(existing)

    for idx in range(start_idx, target):
        if family == "undirected":
            g, stem = _build_undirected(category, track, rng, idx)
            directed = False
        else:
            g, stem = _build_directed(category, track, rng)
            directed = True

        n, edges = _graph_to_edge_list(g, directed)
        out_path = out_dir / f"{stem}_{idx:06d}.txt"
        _write_graph_txt(out_path, n, edges, directed)
        created += 1

    return created, start_idx


def _print_plan(family: str, total: int, ratio: float, weights: Dict[str, float]) -> Dict[Tuple[str, str], int]:
    per_category = _allocate_counts(total, weights)
    plan: Dict[Tuple[str, str], int] = {}

    print(f"\n{family.upper()} plan (total={total}, exact_ratio={ratio:.2f})")
    print("-" * 72)
    for category, cat_total in per_category.items():
        exact, heuristic = _split_tracks(cat_total, ratio)
        plan[(EXACT_TRACK, category)] = exact
        plan[(HEURISTIC_TRACK, category)] = heuristic
        print(f"{category:<18} total={cat_total:>8} exact={exact:>8} heuristic={heuristic:>8}")

    return plan


def _run_family(family: str, total: int, ratio: float, force: bool, seed: int) -> Tuple[int, int]:
    weights = UNDIRECTED_WEIGHTS if family == "undirected" else DIRECTED_WEIGHTS
    plan = _print_plan(family, total, ratio, weights)

    created = 0
    skipped = 0
    for track, category in plan.keys():
        target = plan[(track, category)]
        bucket_seed = seed + sum(ord(ch) for ch in f"{family}:{track}:{category}")
        c, s = _generate_bucket(family, track, category, target, bucket_seed, force)
        created += c
        skipped += s
        print(f"[DONE] {family}/{track}/{category}: created={c}, existing-used={s}")

    return created, skipped


def _validate(args: argparse.Namespace) -> None:
    if args.total_undirected < 0 or args.total_directed < 0:
        raise ValueError("Totals must be >= 0")
    if not (0.0 < args.exact_ratio < 1.0):
        raise ValueError("--exact-ratio must be in (0, 1)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate two-track synthetic benchmark datasets")
    parser.add_argument("--total-undirected", type=int, default=100_000)
    parser.add_argument("--total-directed", type=int, default=100_000)
    parser.add_argument("--exact-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--family", choices=["all", "undirected", "directed"], default="all")
    parser.add_argument("--force", action="store_true", help="Regenerate buckets by deleting existing .txt files")
    args = parser.parse_args()

    _validate(args)
    SYNTH_ROOT.mkdir(parents=True, exist_ok=True)

    families: Sequence[str] = ("undirected", "directed") if args.family == "all" else (args.family,)

    total_created = 0
    total_skipped = 0
    for family in families:
        total = args.total_undirected if family == "undirected" else args.total_directed
        c, s = _run_family(family, total, args.exact_ratio, args.force, args.seed)
        total_created += c
        total_skipped += s

    print("\nSummary")
    print("-------")
    print(f"Created:       {total_created}")
    print(f"Existing-used: {total_skipped}")
    print(f"Output root:   {SYNTH_ROOT}")


if __name__ == "__main__":
    main()
