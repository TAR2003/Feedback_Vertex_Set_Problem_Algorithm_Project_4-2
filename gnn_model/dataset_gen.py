#!/usr/bin/env python3
"""
Generate exact-track PT datasets for GNN training.

Exact-track behavior:
- Reuse all existing graph files from data/synthetic/<family>/exact_track/<category>/*.txt
- Label them with the IC solver and save as .pt
- Do this for both undirected and directed families

Output layout:
    gnn_model/datasets/pt/<family>/exact_track/<category>/*.pt

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
SYNTHETIC_ROOT = PROJECT_ROOT / "data" / "synthetic"

EXACT_TRACK = "exact_track"
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


def _list_existing_txt(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.glob("*.txt") if p.is_file())


def _parse_edge_list_txt(path: Path) -> Tuple[int, List[Tuple[int, int]]]:
    edges: List[Tuple[int, int]] = []
    n_hint: int | None = None

    with path.open("r", encoding="utf-8") as f:
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

            if len(parts) < 2 or not parts[0].lstrip("-").isdigit() or not parts[1].lstrip("-").isdigit():
                continue

            edges.append((int(parts[0]), int(parts[1])))

    if not edges:
        raise ValueError(f"No edges found in {path}")

    verts = {v for e in edges for v in e}
    min_v = min(verts)
    max_v = max(verts)

    # Existing corpora may be 1-based. Normalize to 0-based for engine/model inputs.
    if min_v == 1:
        edges = [(u - 1, v - 1) for u, v in edges]
        max_v -= 1

    n = (n_hint if n_hint is not None else (max_v + 1))
    n = max(n, max_v + 1)
    return n, edges


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


def solve_undirected(n: int, edges: List[Tuple[int, int]]) -> List[int]:
    if HAS_ENGINE:
        return cpp_engine.solve_undirected_IC(n, edges)

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


def solve_directed(n: int, edges: List[Tuple[int, int]]) -> List[int]:
    if HAS_ENGINE:
        return cpp_engine.solve_directed_IC(n, edges)

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
) -> Data:
    if not HAS_TORCH:
        raise RuntimeError("torch and torch_geometric are required for PT generation")

    if graph_type == "undirected":
        feats = compute_node_features_undirected(n, edges)
        fvs = solve_undirected(n, edges)
    else:
        feats = compute_node_features_directed(n, edges)
        fvs = solve_directed(n, edges)

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


def _trim_non_source_pt(folder: Path, source_stems: Set[str]) -> int:
    removed = 0
    for p in _list_existing_pt(folder):
        if p.stem not in source_stems:
            p.unlink()
            removed += 1
    return removed


def _generate_exact_bucket_from_sources(
    family: str,
    category: str,
    force: bool,
    progress_every: int,
) -> Tuple[int, int]:
    src_dir = SYNTHETIC_ROOT / family / EXACT_TRACK / category
    src_files = _list_existing_txt(src_dir)
    if not src_files:
        _log(f"[WARN] No exact source graphs found in {src_dir}")
        return 0, 0

    out_dir = OUTPUT_ROOT / family / EXACT_TRACK / category
    out_dir.mkdir(parents=True, exist_ok=True)

    if force:
        removed = _list_existing_pt(out_dir)
        for p in removed:
            p.unlink()
        if removed:
            _log(f"[CLEAN] {family}/{EXACT_TRACK}/{category}: removed {len(removed)} old .pt files")

    source_stems = {p.stem for p in src_files}
    trimmed = _trim_non_source_pt(out_dir, source_stems)
    if trimmed:
        _log(f"[TRIM] {family}/{EXACT_TRACK}/{category}: removed {trimmed} non-source .pt files")

    created = 0
    existing_used = 0
    total = len(src_files)
    graph_type = "directed" if family == "directed" else "undirected"

    for idx, src_path in enumerate(src_files, start=1):
        out_path = out_dir / f"{src_path.stem}.pt"
        if out_path.exists():
            existing_used += 1
            continue

        n, edges = _parse_edge_list_txt(src_path)
        edges = _normalize_edges(edges, directed=(graph_type == "directed"))

        _log(
            f"  [{family}/{EXACT_TRACK}/{category}] graph {idx}/{total} "
            f"start | n={n} m={len(edges)} solver=ic source={src_path.name}"
        )
        t0 = time.perf_counter()
        data = _build_pt_sample(graph_type, n, edges)
        dt = time.perf_counter() - t0
        data.family = family
        data.track = EXACT_TRACK
        data.category = category
        data.source_file = src_path.name
        torch.save(data, out_path)
        created += 1

        _log(f"    done in {dt:.2f}s | fvs_size={int(data.fvs_size)} | saved={out_path.name}")
        if (created % progress_every) == 0 or (created + existing_used) == total:
            done = created + existing_used
            pct = 100.0 * done / max(total, 1)
            _log(f"  [{family}/{EXACT_TRACK}/{category}] ready {done}/{total} ({pct:.1f}%)")

    return created, existing_used


def _run_family(
    family: str,
    force: bool,
    progress_every: int,
) -> Tuple[int, int]:
    weights = UNDIRECTED_WEIGHTS if family == "undirected" else DIRECTED_WEIGHTS
    categories = list(weights.keys())

    _log(f"\n{family.upper()} PT plan (exact-only)")
    _log("-" * 76)
    for category in categories:
        src_dir = SYNTHETIC_ROOT / family / EXACT_TRACK / category
        count = len(_list_existing_txt(src_dir))
        _log(f"{category:<18} source-exact-files={count:>8}")

    created = 0
    skipped = 0
    for category in categories:
        c, s = _generate_exact_bucket_from_sources(
            family=family,
            category=category,
            force=force,
            progress_every=progress_every,
        )
        created += c
        skipped += s
        _log(f"[DONE] {family}/{EXACT_TRACK}/{category}: created={c}, existing-used={s}")
    return created, skipped


def _remove_heuristic_outputs(families: Sequence[str]) -> int:
    removed = 0
    for family in families:
        path = OUTPUT_ROOT / family / "heuristic_track"
        if path.exists():
            shutil.rmtree(path)
            removed += 1
            _log(f"[CLEAN] removed stale heuristic outputs at {path}")
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate exact-track PT datasets for GNN training")
    parser.add_argument("--family", choices=["all", "undirected", "directed"], default="all")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate exact-track buckets by deleting existing PT files before generation",
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
    args = parser.parse_args()

    if not HAS_TORCH:
        _log("ERROR: torch and torch_geometric are required for PT generation.")
        _log("Install with: pip install torch torch-geometric")
        sys.exit(1)

    if args.clean_root and OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    progress_every = max(1, args.progress_every)

    families: Sequence[str] = ("undirected", "directed") if args.family == "all" else (args.family,)
    _remove_heuristic_outputs(families)

    total_created = 0
    total_existing_used = 0
    for family in families:
        c, s = _run_family(
            family,
            args.force,
            progress_every,
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
