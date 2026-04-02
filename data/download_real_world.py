#!/usr/bin/env python3
"""
Prepare real-world slices for the benchmark tracks using real empirical datasets.

This script populates only the real-world category buckets used by the two-track suite:
    data/synthetic/undirected/exact_track/real_world/
    data/synthetic/undirected/heuristic_track/real_world/
    data/synthetic/directed/exact_track/real_world_ego/
    data/synthetic/directed/heuristic_track/real_world_ego/

Data sources (best effort):
- NetworkX built-in classic social networks.
- PyTorch Geometric datasets (Planetoid, Amazon, Coauthor, TU).
- Open Graph Benchmark (ogbn-arxiv).
- SNAP edge-list archives (downloaded via urllib).

All outputs are normalized to the benchmark TXT edge-list format:
    # format: edge_list_v1
    # directed: 0|1
    # source: ...
    p edge N M
    u v

Notes:
- Deterministic and reproducible output with --seed.
- Exact file-count control per bucket.
- Real datasets are cached under .cache/real_graphs to avoid repeated downloads.
- If a source is unavailable (missing package/network), it is skipped gracefully.
"""

from __future__ import annotations

import argparse
import gzip
import random
import tarfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYNTH_ROOT = PROJECT_ROOT / "data" / "synthetic"
CACHE_ROOT = PROJECT_ROOT / ".cache" / "real_graphs"

EXACT_TRACK = "exact_track"
HEURISTIC_TRACK = "heuristic_track"

UNDIRECTED_REAL_WEIGHT = 0.20
DIRECTED_REAL_WEIGHT = 0.30


@dataclass
class GraphRecord:
    name: str
    directed: bool
    graph: nx.Graph
    source: str


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


def _normalize_graph_nodes(g: nx.Graph) -> nx.Graph:
    # Keep node ids compact and deterministic before writing edge lists.
    return nx.convert_node_labels_to_integers(g, ordering="sorted")


def _largest_component_subgraph(g: nx.Graph) -> nx.Graph:
    if g.number_of_nodes() == 0:
        return g.copy()
    if g.is_directed():
        comps = nx.weakly_connected_components(g)  # type: ignore[arg-type]
    else:
        comps = nx.connected_components(g)  # type: ignore[arg-type]
    largest = max(comps, key=len)
    return g.subgraph(largest).copy()


def _sample_subgraph(g: nx.Graph, target_n: int, rng: random.Random) -> nx.Graph:
    """Take an induced BFS sample to keep local structure from real graphs."""
    if g.number_of_nodes() <= target_n:
        return g.copy()

    nodes = list(g.nodes())
    start = nodes[rng.randrange(0, len(nodes))]
    selected: Set[int] = set([start])

    if g.is_directed():
        frontier = [start]
        while frontier and len(selected) < target_n:
            cur = frontier.pop(0)
            neighbors = list(g.successors(cur)) + list(g.predecessors(cur))  # type: ignore[attr-defined]
            rng.shuffle(neighbors)
            for nb in neighbors:
                if nb not in selected:
                    selected.add(nb)
                    frontier.append(nb)
                    if len(selected) >= target_n:
                        break
    else:
        frontier = [start]
        while frontier and len(selected) < target_n:
            cur = frontier.pop(0)
            neighbors = list(g.neighbors(cur))
            rng.shuffle(neighbors)
            for nb in neighbors:
                if nb not in selected:
                    selected.add(nb)
                    frontier.append(nb)
                    if len(selected) >= target_n:
                        break

    if len(selected) < target_n:
        remaining = [n for n in nodes if n not in selected]
        rng.shuffle(remaining)
        for n in remaining:
            selected.add(n)
            if len(selected) >= target_n:
                break

    return g.subgraph(selected).copy()


def _download_file(url: str, dst: Path, timeout: int = 120) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return True
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, dst.open("wb") as out:
            out.write(resp.read())
        return True
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"[WARN] Failed download: {url} ({exc})")
        return False


def _iter_text_lines(path: Path) -> Iterator[str]:
    name = path.name.lower()
    if name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                yield line
        return
    if name.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            for member in zf.namelist():
                if member.endswith("/"):
                    continue
                with zf.open(member, "r") as f:
                    for raw in f:
                        yield raw.decode("utf-8", errors="ignore")
        return
    if name.endswith(".tar.gz") or name.endswith(".tgz") or name.endswith(".tar.bz2"):
        with tarfile.open(path, "r:*") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                extracted = tf.extractfile(member)
                if extracted is None:
                    continue
                for raw in extracted:
                    yield raw.decode("utf-8", errors="ignore")
        return
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            yield line


def _parse_edge_lines(lines: Iterable[str], directed: bool) -> nx.Graph:
    g: nx.Graph = nx.DiGraph() if directed else nx.Graph()
    for line in lines:
        line = line.strip()
        if not line or line.startswith(("#", "%", "c", "p")):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        if not parts[0].lstrip("-").isdigit() or not parts[1].lstrip("-").isdigit():
            continue
        u = int(parts[0])
        v = int(parts[1])
        if u != v:
            g.add_edge(u, v)
    return g


def _load_networkx_builtins() -> List[GraphRecord]:
    records: List[GraphRecord] = []

    undirected_builders = [
        ("karate_club", nx.karate_club_graph),
        ("davis_southern_women", nx.davis_southern_women_graph),
        ("florentine_families", nx.florentine_families_graph),
        ("les_miserables", nx.les_miserables_graph),
    ]
    for name, builder in undirected_builders:
        try:
            g = _normalize_graph_nodes(builder())
            records.append(GraphRecord(name=name, directed=False, graph=g, source="networkx_builtin"))
        except Exception as exc:
            print(f"[WARN] NetworkX builtin {name} failed: {exc}")

    # Directed variants from built-ins where original graph is undirected.
    for name, builder in undirected_builders:
        try:
            base = builder()
            d = nx.DiGraph()
            d.add_nodes_from(base.nodes())
            d.add_edges_from(base.edges())
            for u, v in list(base.edges()):
                parity_key = sum(ord(ch) for ch in (str(u) + "|" + str(v)))
                if parity_key % 3 == 0:
                    d.add_edge(v, u)
            d = _normalize_graph_nodes(d)
            records.append(GraphRecord(name=f"{name}_oriented", directed=True, graph=d, source="networkx_builtin"))
        except Exception as exc:
            print(f"[WARN] NetworkX directed variant {name} failed: {exc}")

    return records


def _to_networkx_from_pyg(data, directed: bool) -> nx.Graph:
    from torch_geometric.utils import to_networkx

    g = to_networkx(
        data,
        to_undirected=(not directed),
        remove_self_loops=True,
    )
    if directed:
        h = nx.DiGraph()
        h.add_nodes_from(g.nodes())
        h.add_edges_from(g.edges())
        return _normalize_graph_nodes(h)

    h = nx.Graph()
    h.add_nodes_from(g.nodes())
    h.add_edges_from(g.edges())
    return _normalize_graph_nodes(h)


def _load_pyg_datasets() -> List[GraphRecord]:
    records: List[GraphRecord] = []
    try:
        from torch_geometric.datasets import Amazon
        from torch_geometric.datasets import Coauthor
        from torch_geometric.datasets import Planetoid
        from torch_geometric.datasets import TUDataset
    except Exception as exc:
        print(f"[WARN] torch-geometric not available, skipping PyG datasets ({exc})")
        return records

    pyg_root = CACHE_ROOT / "pyg"

    # Citation networks (directed, real).
    for name in ("Cora", "CiteSeer", "PubMed"):
        try:
            ds = Planetoid(root=str(pyg_root), name=name)
            g = _to_networkx_from_pyg(ds[0], directed=True)
            records.append(GraphRecord(name=name.lower(), directed=True, graph=g, source="pyg_planetoid"))
        except Exception as exc:
            print(f"[WARN] PyG Planetoid {name} failed: {exc}")

    # Co-purchase / coauthor networks (treated as undirected).
    for name in ("Computers", "Photo"):
        try:
            ds = Amazon(root=str(pyg_root), name=name)
            g = _to_networkx_from_pyg(ds[0], directed=False)
            records.append(GraphRecord(name=f"amazon_{name.lower()}", directed=False, graph=g, source="pyg_amazon"))
        except Exception as exc:
            print(f"[WARN] PyG Amazon {name} failed: {exc}")

    for name in ("CS", "Physics"):
        try:
            ds = Coauthor(root=str(pyg_root), name=name)
            g = _to_networkx_from_pyg(ds[0], directed=False)
            records.append(GraphRecord(name=f"coauthor_{name.lower()}", directed=False, graph=g, source="pyg_coauthor"))
        except Exception as exc:
            print(f"[WARN] PyG Coauthor {name} failed: {exc}")

    # Small molecule/protein graph collections for exact-track sampling.
    tu_specs = [
        ("MUTAG", 3),
        ("PROTEINS", 3),
        ("ENZYMES", 3),
    ]
    for name, take_n in tu_specs:
        try:
            ds = TUDataset(root=str(pyg_root), name=name)
            for idx in range(min(take_n, len(ds))):
                g = _to_networkx_from_pyg(ds[idx], directed=False)
                records.append(
                    GraphRecord(
                        name=f"{name.lower()}_{idx}",
                        directed=False,
                        graph=g,
                        source="pyg_tu",
                    )
                )
        except Exception as exc:
            print(f"[WARN] PyG TUDataset {name} failed: {exc}")

    return records


def _load_ogb_datasets() -> List[GraphRecord]:
    records: List[GraphRecord] = []
    try:
        from ogb.nodeproppred import NodePropPredDataset
    except Exception as exc:
        print(f"[WARN] ogb not available, skipping OGB datasets ({exc})")
        return records

    ogb_root = CACHE_ROOT / "ogb"
    try:
        ds = NodePropPredDataset(name="ogbn-arxiv", root=str(ogb_root))
        graph_obj, _ = ds[0]
        edge_index = graph_obj["edge_index"]
        d = nx.DiGraph()
        d.add_nodes_from(range(int(graph_obj["num_nodes"])))
        for u, v in zip(edge_index[0], edge_index[1]):
            u_i = int(u)
            v_i = int(v)
            if u_i != v_i:
                d.add_edge(u_i, v_i)
        records.append(GraphRecord(name="ogbn_arxiv", directed=True, graph=_normalize_graph_nodes(d), source="ogb"))
    except Exception as exc:
        print(f"[WARN] OGB ogbn-arxiv failed: {exc}")

    return records


def _load_snap_datasets() -> List[GraphRecord]:
    records: List[GraphRecord] = []
    snap_root = CACHE_ROOT / "snap"
    specs = [
        ("wiki_vote", True, "https://snap.stanford.edu/data/wiki-Vote.txt.gz"),
        ("email_enron", True, "https://snap.stanford.edu/data/email-Enron.txt.gz"),
        ("p2p_gnutella04", True, "https://snap.stanford.edu/data/p2p-Gnutella04.txt.gz"),
        ("ca_grqc", False, "https://snap.stanford.edu/data/ca-GrQc.txt.gz"),
        ("ca_hepth", False, "https://snap.stanford.edu/data/ca-HepTh.txt.gz"),
        ("ca_astroph", False, "https://snap.stanford.edu/data/ca-AstroPh.txt.gz"),
    ]

    for name, directed, url in specs:
        dst = snap_root / url.split("/")[-1]
        if not _download_file(url, dst):
            continue
        try:
            g = _parse_edge_lines(_iter_text_lines(dst), directed=directed)
            if g.number_of_nodes() == 0 or g.number_of_edges() == 0:
                continue
            g = _largest_component_subgraph(g)
            g = _normalize_graph_nodes(g)
            records.append(GraphRecord(name=name, directed=directed, graph=g, source="snap"))
        except Exception as exc:
            print(f"[WARN] SNAP parse failed for {name}: {exc}")

    return records


def _collect_real_graphs() -> List[GraphRecord]:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    all_records: List[GraphRecord] = []
    all_records.extend(_load_networkx_builtins())
    all_records.extend(_load_pyg_datasets())
    all_records.extend(_load_ogb_datasets())
    all_records.extend(_load_snap_datasets())

    usable: List[GraphRecord] = []
    for rec in all_records:
        g = rec.graph
        if g.number_of_nodes() < 5 or g.number_of_edges() < 5:
            continue
        usable.append(rec)

    print(f"[INFO] Loaded real graphs: {len(usable)} usable instance(s)")
    directed_count = sum(1 for r in usable if r.directed)
    print(f"[INFO] Directed={directed_count}, Undirected={len(usable) - directed_count}")
    return usable


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


def _pick_real_record(records: List[GraphRecord], directed: bool, rng: random.Random) -> Optional[GraphRecord]:
    pool = [r for r in records if r.directed == directed]
    if not pool:
        return None
    return pool[rng.randrange(0, len(pool))]


def _build_real_slice(
    records: List[GraphRecord],
    directed: bool,
    track: str,
    rng: random.Random,
) -> Tuple[nx.Graph, str]:
    rec = _pick_real_record(records, directed=directed, rng=rng)
    if rec is None:
        # Last-resort fallback if no real source loaded at all.
        n = _randint(rng, 10, 35) if track == EXACT_TRACK else _randint(rng, 100, 2000)
        if directed:
            g = nx.gn_graph(n, seed=rng.randint(0, 10**9)).to_directed()
            return _normalize_graph_nodes(g), "fallback_proxy"
        g = nx.barabasi_albert_graph(n, 2, seed=rng.randint(0, 10**9))
        return _normalize_graph_nodes(g), "fallback_proxy"

    base = _largest_component_subgraph(rec.graph)
    if track == EXACT_TRACK:
        target_n = _randint(rng, 10, 35)
    else:
        target_n = _randint(rng, 100, 5000)

    if base.number_of_nodes() > target_n:
        sampled = _sample_subgraph(base, target_n=target_n, rng=rng)
    else:
        sampled = base.copy()

    sampled = _normalize_graph_nodes(sampled)
    return sampled, f"{rec.source}:{rec.name}"


def _generate_bucket(
    family: str,
    track: str,
    category: str,
    target: int,
    seed: int,
    force: bool,
    records: List[GraphRecord],
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
            g, source_tag = _build_real_slice(records, directed=False, track=track, rng=rng)
            directed = False
            stem = "real"
        else:
            g, source_tag = _build_real_slice(records, directed=True, track=track, rng=rng)
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
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    records = _collect_real_graphs()
    if not records:
        print("[WARN] No real datasets were loaded; falling back to synthetic proxies.")

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
            records=records,
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
