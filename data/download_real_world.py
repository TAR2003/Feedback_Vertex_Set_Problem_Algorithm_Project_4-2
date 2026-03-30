#!/usr/bin/env python3
"""
Download and prepare real-world graph datasets for FVS benchmarking.

This script is idempotent:
- If a target file already exists, it is skipped.
- Only missing datasets are downloaded/generated.

Output folders:
- data/raw_undirected/
- data/raw_directed/ (currently only optional directed downloads if configured)
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import urllib.request
from pathlib import Path
from typing import Dict, Hashable, Iterable, List, Set, Tuple

import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_UNDIRECTED_DIR = PROJECT_ROOT / "data" / "raw_undirected"
RAW_DIRECTED_DIR = PROJECT_ROOT / "data" / "raw_directed"

# Small built-in NetworkX datasets (undirected).
NETWORKX_DATASETS: Dict[str, callable] = {
    "karate_club": nx.karate_club_graph,
    "florentine_families": nx.florentine_families_graph,
    "les_miserables": nx.les_miserables_graph,
    "davis_southern_women": nx.davis_southern_women_graph,
}

# SNAP sources with explicit directed/undirected routing.
# Key: output file name, value: (source URL, is_directed)
SNAP_DATASETS: Dict[str, Tuple[str, bool]] = {
    # Social / collaboration (undirected)
    "facebook_combined.txt": ("https://snap.stanford.edu/data/facebook_combined.txt.gz", False),
    "ca-GrQc.txt": ("https://snap.stanford.edu/data/ca-GrQc.txt.gz", False),

    # Communication / web / routing (directed)
    "email-Enron.txt": ("https://snap.stanford.edu/data/email-Enron.txt.gz", True),
    "email-Eu-core.txt": ("https://snap.stanford.edu/data/email-Eu-core.txt.gz", True),
    "p2p-Gnutella08.txt": ("https://snap.stanford.edu/data/p2p-Gnutella08.txt.gz", True),
    "as-caida20071105.txt": ("https://snap.stanford.edu/data/as-caida20071105.txt.gz", True),
    "web-Stanford.txt": ("https://snap.stanford.edu/data/web-Stanford.txt.gz", True),
}


def ensure_dirs() -> None:
    RAW_UNDIRECTED_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIRECTED_DIR.mkdir(parents=True, exist_ok=True)


def coerce_or_remap_edges(edges: Iterable[Tuple[Hashable, Hashable]]) -> List[Tuple[int, int]]:
    """Convert endpoints to integers; if not possible, remap labels to integer IDs."""
    edge_list = list(edges)
    converted: List[Tuple[int, int]] = []
    try:
        for u_raw, v_raw in edge_list:
            converted.append((int(u_raw), int(v_raw)))
        return converted
    except (TypeError, ValueError):
        pass

    labels: Set[Hashable] = set()
    for u_raw, v_raw in edge_list:
        labels.add(u_raw)
        labels.add(v_raw)
    mapping = {label: idx for idx, label in enumerate(sorted(labels, key=lambda x: str(x)))}
    return [(mapping[u_raw], mapping[v_raw]) for u_raw, v_raw in edge_list]


def normalize_edges(edges: Iterable[Tuple[int, int]], directed: bool) -> List[Tuple[int, int]]:
    """Normalize and deduplicate edges to a deterministic two-column format."""
    seen: Set[Tuple[int, int]] = set()
    for u_raw, v_raw in edges:
        u = int(u_raw)
        v = int(v_raw)
        if directed:
            edge = (u, v)
        else:
            edge = (u, v) if u <= v else (v, u)
        seen.add(edge)
    return sorted(seen)


def remap_to_zero_index(edges: Iterable[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], int]:
    """Remap arbitrary node ids to compact 0..n-1 ids for parser stability."""
    node_ids: Set[int] = set()
    edge_list = list(edges)
    for u, v in edge_list:
        node_ids.add(u)
        node_ids.add(v)

    if not node_ids:
        return [], 0

    node_map = {old_id: new_id for new_id, old_id in enumerate(sorted(node_ids))}
    remapped = [(node_map[u], node_map[v]) for u, v in edge_list]
    return remapped, len(node_map)


def write_edge_list(edges: Iterable[Tuple[int, int]], out_file: Path) -> None:
    """Write canonical plain edge-list: one 'u v' per line."""
    with out_file.open("w", encoding="utf-8") as f:
        for u, v in edges:
            f.write(f"{u} {v}\n")


def is_canonical_two_column(path: Path) -> bool:
    """Return True if file already looks like canonical two-integer edge list."""
    saw_edge = False
    saw_zero = False
    with path.open("r", encoding="utf-8", errors="replace") as src:
        for line in src:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                return False
            if not parts[0].lstrip("-").isdigit() or not parts[1].lstrip("-").isdigit():
                return False
            u = int(parts[0])
            v = int(parts[1])
            if u == 0 or v == 0:
                saw_zero = True
            saw_edge = True
    return saw_edge and saw_zero


def parse_text_edgelist(file_path: Path) -> List[Tuple[int, int]]:
    """Parse text edge list while tolerating comments, headers, and weights."""
    edges: List[Tuple[int, int]] = []
    with file_path.open("r", encoding="utf-8", errors="replace") as src:
        for line in src:
            line = line.strip()
            if not line or line.startswith(("#", "%")):
                continue
            parts = line.split()

            if parts[0].lower() == "p":
                continue
            if not parts[0].lstrip("-").isdigit():
                continue
            if len(parts) < 2:
                continue

            try:
                u = int(parts[0])
                v = int(parts[1])
            except ValueError:
                continue
            edges.append((u, v))
    return edges


def save_graph_as_normalized_edgelist(graph: nx.Graph, out_file: Path, directed: bool) -> None:
    raw_edges = list(graph.edges())
    int_edges = coerce_or_remap_edges(raw_edges)
    normalized = normalize_edges(int_edges, directed=directed)
    remapped, _ = remap_to_zero_index(normalized)
    with out_file.open("w", encoding="utf-8") as f:
        for u, v in remapped:
            f.write(f"{u} {v}\n")


def download_gzip_to_text(url: str, out_file: Path) -> None:
    """Download a .gz text edge list and extract it to out_file."""
    tmp_gz = out_file.with_suffix(out_file.suffix + ".gz")
    try:
        urllib.request.urlretrieve(url, tmp_gz)
        with gzip.open(tmp_gz, "rt", encoding="utf-8", errors="replace") as src, out_file.open(
            "w", encoding="utf-8"
        ) as dst:
            shutil.copyfileobj(src, dst)
    finally:
        if tmp_gz.exists():
            tmp_gz.unlink()


def download_and_normalize_snap(url: str, out_file: Path, directed: bool) -> None:
    """Download SNAP .txt.gz then rewrite as normalized two-column edge list."""
    tmp_raw = out_file.with_suffix(out_file.suffix + ".tmp")
    download_gzip_to_text(url, tmp_raw)
    try:
        raw_edges = parse_text_edgelist(tmp_raw)
        normalized = normalize_edges(raw_edges, directed=directed)
        remapped, _ = remap_to_zero_index(normalized)
        with out_file.open("w", encoding="utf-8") as dst:
            for u, v in remapped:
                dst.write(f"{u} {v}\n")
    finally:
        if tmp_raw.exists():
            tmp_raw.unlink()


def normalize_existing_snap_file(path: Path, directed: bool) -> bool:
    """Normalize existing file in-place; returns True if rewritten."""
    if not path.exists() or is_canonical_two_column(path):
        return False

    raw_edges = parse_text_edgelist(path)
    normalized = normalize_edges(raw_edges, directed=directed)
    remapped, _ = remap_to_zero_index(normalized)

    tmp_path = path.with_suffix(path.suffix + ".normalized")
    try:
        with tmp_path.open("w", encoding="utf-8") as dst:
            for u, v in remapped:
                dst.write(f"{u} {v}\n")
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return True


def migrate_misplaced_snap_files() -> int:
    """Move known SNAP files to the folder matching their directedness."""
    moved = 0
    for out_name, (_url, is_directed) in SNAP_DATASETS.items():
        expected = RAW_DIRECTED_DIR if is_directed else RAW_UNDIRECTED_DIR
        wrong = RAW_UNDIRECTED_DIR if is_directed else RAW_DIRECTED_DIR

        wrong_path = wrong / out_name
        expected_path = expected / out_name
        if wrong_path.exists() and not expected_path.exists():
            shutil.move(str(wrong_path), str(expected_path))
            moved += 1
            print(f"[FIX] moved misplaced {out_name} -> {expected.name}/")
    return moved


def fetch_networkx_datasets(force: bool = False) -> Tuple[int, int]:
    """Return (downloaded_or_generated_count, skipped_count)."""
    created = 0
    skipped = 0

    for name, graph_builder in NETWORKX_DATASETS.items():
        out_file = RAW_UNDIRECTED_DIR / f"{name}.txt"
        if out_file.exists() and not force:
            if not is_canonical_two_column(out_file):
                graph = graph_builder()
                save_graph_as_normalized_edgelist(graph, out_file, directed=False)
                created += 1
                print(f"[FIX] normalized existing {out_file.name} using NetworkX source graph")
                continue
            skipped += 1
            print(f"[SKIP] {out_file.name} already exists")
            continue

        graph = graph_builder()
        save_graph_as_normalized_edgelist(graph, out_file, directed=False)
        created += 1
        print(f"[OK]   wrote {out_file.name} ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)")

    return created, skipped


def fetch_snap_datasets(force: bool = False) -> Tuple[int, int, List[str]]:
    """
    Return (downloaded_count, skipped_count, failed_files).

    Failures are reported but do not stop the whole pipeline.
    """
    downloaded = 0
    skipped = 0
    normalized_existing = 0
    failed: List[str] = []

    for out_name, (url, is_directed) in SNAP_DATASETS.items():
        out_dir = RAW_DIRECTED_DIR if is_directed else RAW_UNDIRECTED_DIR
        out_file = out_dir / out_name
        if out_file.exists() and not force:
            if normalize_existing_snap_file(out_file, directed=is_directed):
                normalized_existing += 1
                print(f"[FIX] normalized existing {out_file.name}")
            skipped += 1
            print(f"[SKIP] {out_file.name} already exists")
            continue

        try:
            graph_kind = "directed" if is_directed else "undirected"
            print(f"[DL ]  {out_name} from SNAP ({graph_kind})")
            download_and_normalize_snap(url, out_file, directed=is_directed)
            downloaded += 1
            print(f"[OK]   downloaded {out_file.name}")
        except Exception as ex:
            failed.append(out_name)
            print(f"[WARN] failed {out_name}: {ex}")

    if normalized_existing:
        print(f"[INFO] normalized {normalized_existing} existing SNAP file(s)")

    return downloaded, skipped, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/generate real-world benchmark graphs")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing target files instead of skipping",
    )
    args = parser.parse_args()

    ensure_dirs()
    moved = migrate_misplaced_snap_files()

    print("Preparing real-world undirected datasets...")
    nx_created, nx_skipped = fetch_networkx_datasets(force=args.force)
    snap_created, snap_skipped, snap_failed = fetch_snap_datasets(force=args.force)

    print("\nSummary")
    print("-------")
    print(f"Misplaced files moved: {moved}")
    print(f"NetworkX created: {nx_created}, skipped: {nx_skipped}")
    print(f"SNAP downloaded:  {snap_created}, skipped: {snap_skipped}")
    if snap_failed:
        print(f"SNAP failed:      {len(snap_failed)} -> {', '.join(snap_failed)}")


if __name__ == "__main__":
    main()
