#!/usr/bin/env python3
"""
Generate benchmark-style PT datasets for GNN training.

This script mirrors the benchmark distribution logic used in data/generate_synthetic.py:
- Undirected categories: real_world 20%, scale_free 20%, small_world 20%, random_er 20%, grids_trees 20%
- Directed categories: real_world_ego 30%, scale_free 20%, random_er 20%, directed_grids 15%, dags 15%
- Track split per category: exact_track / heuristic_track via --exact-ratio

Output layout:
  gnn_model/datasets/pt/<family>/<track>/<category>/*.pt

Each .pt file is a torch_geometric Data object with:
  data.x, data.edge_index, data.y, data.fvs_size
"""

from __future__ import annotations

import argparse
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for candidate in ("build-linux", "build-macos", "build-win", "build"):
    sys.path.insert(0, str(PROJECT_ROOT / "cpp_engine" / candidate))

try:
    import cpp_engine

    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False
    print("WARNING: cpp_engine not found. Using Python fallback solver (slow).")

try:
    import torch
    from torch_geometric.data import Data

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _log(msg: str) -> None:
    """Always flush output so progress is visible during long runs."""
    print(msg, flush=True)


OUTPUT_ROOT = PROJECT_ROOT / "gnn_model" / "datasets" / "pt"

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


def _build_undirected(category: str, track: str, rng: random.Random, idx: int) -> nx.Graph:
    if category == "real_world":
        if track == EXACT_TRACK:
            n = _randint(rng, 10, 35)
            return _molecule_like_graph(n, rng)
        n = _randint(rng, 100, 5000)
        m = _randint(rng, 2, 6)
        return nx.barabasi_albert_graph(n, m, seed=rng.randint(0, 10**9))

    if category == "scale_free":
        if track == EXACT_TRACK:
            n = _randint(rng, 15, 35)
            m = _randint(rng, 2, 3)
        else:
            n = _randint(rng, 100, 2000)
            m = _randint(rng, 2, 5)
        return nx.barabasi_albert_graph(n, m, seed=rng.randint(0, 10**9))

    if category == "small_world":
        if track == EXACT_TRACK:
            n = _randint(rng, 15, 35)
            k, p = 4, 0.1
        else:
            n = _randint(rng, 100, 1000)
            k, p = 6, 0.2
        return nx.watts_strogatz_graph(n, k, p, seed=rng.randint(0, 10**9))

    if category == "random_er":
        if track == EXACT_TRACK:
            n = _randint(rng, 10, 30)
            p = rng.uniform(0.1, 0.4)
        else:
            n = _randint(rng, 50, 1000)
            p = rng.uniform(0.05, 0.2)
        return nx.erdos_renyi_graph(n, p, seed=rng.randint(0, 10**9))

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
            return nx.convert_node_labels_to_integers(g)

        if track == EXACT_TRACK:
            n = _randint(rng, 10, 35)
        else:
            n = _randint(rng, 100, 5000)
        return _random_tree(n, seed=rng.randint(0, 10**9))

    raise ValueError(f"Unknown undirected category: {category}")


def _build_directed(category: str, track: str, rng: random.Random) -> nx.DiGraph:
    if category == "real_world_ego":
        if track == EXACT_TRACK:
            n = _randint(rng, 10, 35)
            return _directed_real_world_proxy(n, rng, dense=False)
        n = _randint(rng, 100, 5000)
        return _directed_real_world_proxy(n, rng, dense=True)

    if category == "scale_free":
        if track == EXACT_TRACK:
            n = _randint(rng, 15, 35)
        else:
            n = _randint(rng, 100, 2000)
        return _directed_scale_free(n, rng)

    if category == "random_er":
        if track == EXACT_TRACK:
            n = _randint(rng, 10, 30)
            p = rng.uniform(0.1, 0.4)
        else:
            n = _randint(rng, 50, 1000)
            p = rng.uniform(0.05, 0.2)
        return nx.erdos_renyi_graph(n, p, directed=True, seed=rng.randint(0, 10**9))

    if category == "directed_grids":
        if track == EXACT_TRACK:
            side = _randint(rng, 3, 6)
            rows, cols = side, side
        else:
            rows = _randint(rng, 10, 50)
            cols = _randint(rng, 10, 50)
        return _directed_grid(rows, cols, rng)

    if category == "dags":
        if track == EXACT_TRACK:
            n = _randint(rng, 10, 35)
        else:
            n = _randint(rng, 100, 5000)
        return nx.gn_graph(n, seed=rng.randint(0, 10**9))

    raise ValueError(f"Unknown directed category: {category}")


def _graph_to_edge_list(g: nx.Graph, directed: bool) -> Tuple[int, List[Tuple[int, int]]]:
    node_map = {node: idx for idx, node in enumerate(sorted(g.nodes(), key=str))}
    edges = [(node_map[u], node_map[v]) for u, v in g.edges()]
    n = len(node_map)
    return n, _normalize_edges(edges, directed=directed)


def _cap_graph_size(g: nx.Graph, max_nodes: int, seed: int) -> nx.Graph:
    """Downsample very large graphs to keep PT label generation practical."""
    if max_nodes <= 0 or g.number_of_nodes() <= max_nodes:
        return g

    rng = random.Random(seed)
    nodes = list(g.nodes())
    rng.shuffle(nodes)
    keep = set(nodes[:max_nodes])
    sub = g.subgraph(keep).copy()
    return nx.convert_node_labels_to_integers(sub)


def compute_node_features_undirected(n: int, edges: List[Tuple[int, int]]) -> List[List[float]]:
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)

    degrees = dict(g.degree())
    clust = nx.clustering(g)
    feats: List[List[float]] = []
    for v in range(n):
        deg = degrees.get(v, 0)
        feats.append(
            [
                deg / max(n - 1, 1),
                clust.get(v, 0.0),
                math.log(deg + 1) / math.log(n + 1),
            ]
        )
    return feats


def compute_node_features_directed(n: int, edges: List[Tuple[int, int]]) -> List[List[float]]:
    in_deg = [0] * n
    out_deg = [0] * n
    for u, v in edges:
        out_deg[u] += 1
        in_deg[v] += 1

    feats: List[List[float]] = []
    for v in range(n):
        ind = in_deg[v]
        outd = out_deg[v]
        feats.append(
            [
                ind / max(n - 1, 1),
                outd / max(n - 1, 1),
                min(ind, outd) / max(n - 1, 1),
            ]
        )
    return feats


def solve_undirected(n: int, edges: List[Tuple[int, int]], solver_mode: str = "auto") -> List[int]:
    if HAS_ENGINE:
        if solver_mode == "ma":
            return cpp_engine.solve_undirected_MA(n, edges, 30, 80)
        if solver_mode == "ic":
            return cpp_engine.solve_undirected_IC(n, edges)
        if n <= 120:
            return cpp_engine.solve_undirected_IC(n, edges)
        return cpp_engine.solve_undirected_MA(n, edges, 30, 80)

    adj = {v: set() for v in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    fvs: List[int] = []
    removed: Set[int] = set()

    def has_cycle() -> bool:
        visited: Set[int] = set()

        def dfs(cur: int, parent: int) -> bool:
            visited.add(cur)
            for nb in adj[cur]:
                if nb in removed:
                    continue
                if nb == parent:
                    continue
                if nb in visited:
                    return True
                if dfs(nb, cur):
                    return True
            return False

        for node in range(n):
            if node not in removed and node not in visited and dfs(node, -1):
                return True
        return False

    while has_cycle():
        best = max((v for v in range(n) if v not in removed), key=lambda v: len(adj[v] - removed))
        removed.add(best)
        fvs.append(best)
    return fvs


def solve_directed(n: int, edges: List[Tuple[int, int]], solver_mode: str = "auto") -> List[int]:
    if HAS_ENGINE:
        if solver_mode == "ma":
            return cpp_engine.solve_directed_MA(n, edges, 30, 80)
        if solver_mode == "ic":
            return cpp_engine.solve_directed_IC(n, edges)
        if n <= 120:
            return cpp_engine.solve_directed_IC(n, edges)
        return cpp_engine.solve_directed_MA(n, edges, 30, 80)

    out_adj = {v: set() for v in range(n)}
    for u, v in edges:
        out_adj[u].add(v)

    fvs: List[int] = []
    removed: Set[int] = set()

    def has_dcycle() -> bool:
        color: Dict[int, int] = {}

        def dfs(cur: int) -> bool:
            color[cur] = 1
            for nb in out_adj[cur]:
                if nb in removed:
                    continue
                c = color.get(nb, 0)
                if c == 1:
                    return True
                if c == 0 and dfs(nb):
                    return True
            color[cur] = 2
            return False

        for node in range(n):
            if node not in removed and color.get(node, 0) == 0 and dfs(node):
                return True
        return False

    while has_dcycle():
        best = max((v for v in range(n) if v not in removed), key=lambda v: len(out_adj[v] - removed))
        removed.add(best)
        fvs.append(best)
    return fvs


def _build_pt_sample(
    graph_type: str,
    n: int,
    edges: List[Tuple[int, int]],
    solver_mode: str,
) -> Data:
    if not HAS_TORCH:
        raise RuntimeError("torch and torch_geometric are required for PT generation")

    if graph_type == "undirected":
        feats = compute_node_features_undirected(n, edges)
        fvs = solve_undirected(n, edges, solver_mode=solver_mode)
    else:
        feats = compute_node_features_directed(n, edges)
        fvs = solve_directed(n, edges, solver_mode=solver_mode)

    x = torch.tensor(feats, dtype=torch.float)
    y = torch.zeros(n, dtype=torch.long)
    for v in fvs:
        if 0 <= v < n:
            y[v] = 1

    if edges:
        ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
        if graph_type == "undirected":
            ei = torch.cat([ei, ei.flip(0)], dim=1)
    else:
        ei = torch.zeros((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=ei, y=y)
    data.fvs_size = len(fvs)
    return data


def _list_existing_pt(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.glob("*.pt") if p.is_file())


def _trim_excess(folder: Path, target: int) -> int:
    existing = _list_existing_pt(folder)
    if len(existing) <= target:
        return 0
    for stale in existing[target:]:
        stale.unlink()
    return len(existing) - target


def _generate_bucket(
    family: str,
    track: str,
    category: str,
    target: int,
    seed: int,
    force: bool,
    progress_every: int,
    max_nodes: int,
    solver_mode: str,
) -> Tuple[int, int]:
    out_dir = OUTPUT_ROOT / family / track / category
    out_dir.mkdir(parents=True, exist_ok=True)

    if force:
        removed = _list_existing_pt(out_dir)
        for p in removed:
            p.unlink()
        if removed:
            _log(f"[CLEAN] {family}/{track}/{category}: removed {len(removed)} old .pt files")

    trimmed = _trim_excess(out_dir, target)
    if trimmed:
        _log(f"[TRIM] {family}/{track}/{category}: removed {trimmed} excess .pt files")

    existing = _list_existing_pt(out_dir)
    if len(existing) >= target:
        return 0, target

    created = 0
    rng = random.Random(seed)
    start_idx = len(existing)
    bucket_total = max(target - start_idx, 1)

    for idx in range(start_idx, target):
        if family == "undirected":
            g = _build_undirected(category, track, rng, idx)
            g = _cap_graph_size(g, max_nodes=max_nodes, seed=seed + idx)
            graph_type = "undirected"
            stem = category.replace("_", "")
        else:
            g = _build_directed(category, track, rng)
            g = _cap_graph_size(g, max_nodes=max_nodes, seed=seed + idx)
            graph_type = "directed"
            stem = category.replace("_", "")

        progress_idx = created + 1
        n, edges = _graph_to_edge_list(g, directed=(graph_type == "directed"))
        pct_before = 100.0 * progress_idx / bucket_total
        _log(
            f"  [{family}/{track}/{category}] graph {progress_idx}/{bucket_total} "
            f"({pct_before:.1f}%) start | n={n} m={len(edges)} solver={solver_mode}"
        )

        t0 = time.perf_counter()
        data = _build_pt_sample(graph_type, n, edges, solver_mode=solver_mode)
        dt = time.perf_counter() - t0
        data.family = family
        data.track = track
        data.category = category

        out_path = out_dir / f"{stem}_{idx:06d}.pt"
        torch.save(data, out_path)
        created += 1

        _log(
            f"    done in {dt:.2f}s | fvs_size={int(data.fvs_size)} | saved={out_path.name}"
        )

        if (created % progress_every) == 0 or created == (target - start_idx):
            pct = 100.0 * created / bucket_total
            _log(
                f"  [{family}/{track}/{category}] created {created} / {bucket_total} "
                f"({pct:.1f}%)"
            )

    return created, start_idx


def _print_plan(family: str, total: int, ratio: float, weights: Dict[str, float]) -> Dict[Tuple[str, str], int]:
    per_category = _allocate_counts(total, weights)
    plan: Dict[Tuple[str, str], int] = {}

    _log(f"\n{family.upper()} PT plan (total={total}, exact_ratio={ratio:.2f})")
    _log("-" * 76)
    for category, cat_total in per_category.items():
        exact, heuristic = _split_tracks(cat_total, ratio)
        plan[(EXACT_TRACK, category)] = exact
        plan[(HEURISTIC_TRACK, category)] = heuristic
        _log(f"{category:<18} total={cat_total:>8} exact={exact:>8} heuristic={heuristic:>8}")

    return plan


def _run_family(
    family: str,
    total: int,
    ratio: float,
    force: bool,
    seed: int,
    progress_every: int,
    max_nodes: int,
    solver_mode: str,
) -> Tuple[int, int]:
    weights = UNDIRECTED_WEIGHTS if family == "undirected" else DIRECTED_WEIGHTS
    plan = _print_plan(family, total, ratio, weights)

    created = 0
    skipped = 0
    for track, category in plan.keys():
        target = plan[(track, category)]
        bucket_seed = seed + sum(ord(ch) for ch in f"{family}:{track}:{category}")
        c, s = _generate_bucket(
            family,
            track,
            category,
            target,
            bucket_seed,
            force,
            progress_every,
            max_nodes,
            solver_mode,
        )
        created += c
        skipped += s
        _log(f"[DONE] {family}/{track}/{category}: created={c}, existing-used={s}")
    return created, skipped


def _validate(args: argparse.Namespace) -> None:
    if args.total_undirected < 0 or args.total_directed < 0:
        raise ValueError("Totals must be >= 0")
    if not (0.0 < args.exact_ratio < 1.0):
        raise ValueError("--exact-ratio must be in (0, 1)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark-style PT datasets for GNN training")
    parser.add_argument("--total-undirected", type=int, default=100_000)
    parser.add_argument("--total-directed", type=int, default=100_000)
    parser.add_argument("--exact-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--family", choices=["all", "undirected", "directed"], default="all")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate buckets by deleting existing PT files before generation",
    )
    parser.add_argument(
        "--clean-root",
        action="store_true",
        help="Delete the full PT output root before generation",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N created files per bucket (default: 10)",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=300,
        help="Maximum nodes per generated graph before downsampling (default: 300)",
    )
    parser.add_argument(
        "--solver-mode",
        choices=["auto", "ic", "ma"],
        default="auto",
        help="Label solver mode: auto (default), ic (exact-ish), ma (faster heuristic)",
    )
    args = parser.parse_args()

    if not HAS_TORCH:
        _log("ERROR: torch and torch_geometric are required for PT generation.")
        _log("Install with: pip install torch torch-geometric")
        sys.exit(1)

    _validate(args)

    if args.clean_root and OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    progress_every = max(1, args.progress_every)
    max_nodes = max(0, args.max_nodes)

    families: Sequence[str] = ("undirected", "directed") if args.family == "all" else (args.family,)

    total_created = 0
    total_existing_used = 0
    for family in families:
        total = args.total_undirected if family == "undirected" else args.total_directed
        c, s = _run_family(
            family,
            total,
            args.exact_ratio,
            args.force,
            args.seed,
            progress_every,
            max_nodes,
            args.solver_mode,
        )
        total_created += c
        total_existing_used += s

    _log("\nSummary")
    _log("-------")
    _log(f"Created:       {total_created}")
    _log(f"Existing-used: {total_existing_used}")
    _log(f"Output root:   {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
