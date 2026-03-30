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
from typing import Dict, List, Tuple

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

# SNAP sources are commonly distributed as .txt.gz edge lists.
# Key: output file name in raw_undirected/, value: source URL.
SNAP_DATASETS: Dict[str, str] = {
    # 1. Social Networks
    "facebook_combined.txt": "https://snap.stanford.edu/data/facebook_combined.txt.gz",
    "ca-GrQc.txt": "https://snap.stanford.edu/data/ca-GrQc.txt.gz",

    # 2. Email Communication Networks
    "email-Enron.txt": "https://snap.stanford.edu/data/email-Enron.txt.gz",
    "email-Eu-core.txt": "https://snap.stanford.edu/data/email-Eu-core.txt.gz",

    # 3. Peer-to-Peer / Technology Networks
    "p2p-Gnutella08.txt": "https://snap.stanford.edu/data/p2p-Gnutella08.txt.gz",
    "as-caida20071105.txt": "https://snap.stanford.edu/data/as-caida20071105.txt.gz",

    # 4. Web Graphs
    "web-Stanford.txt": "https://snap.stanford.edu/data/web-Stanford.txt.gz",
}


def ensure_dirs() -> None:
    RAW_UNDIRECTED_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIRECTED_DIR.mkdir(parents=True, exist_ok=True)


def save_undirected_graph_as_edgelist(graph: nx.Graph, out_file: Path) -> None:
    """Save a graph as a plain edge list: one 'u v' per line."""
    with out_file.open("w", encoding="utf-8") as f:
        for u, v in graph.edges():
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


def fetch_networkx_datasets(force: bool = False) -> Tuple[int, int]:
    """Return (downloaded_or_generated_count, skipped_count)."""
    created = 0
    skipped = 0

    for name, graph_builder in NETWORKX_DATASETS.items():
        out_file = RAW_UNDIRECTED_DIR / f"{name}.txt"
        if out_file.exists() and not force:
            skipped += 1
            print(f"[SKIP] {out_file.name} already exists")
            continue

        graph = graph_builder()
        save_undirected_graph_as_edgelist(graph, out_file)
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
    failed: List[str] = []

    for out_name, url in SNAP_DATASETS.items():
        out_file = RAW_UNDIRECTED_DIR / out_name
        if out_file.exists() and not force:
            skipped += 1
            print(f"[SKIP] {out_file.name} already exists")
            continue

        try:
            print(f"[DL ]  {out_name} from SNAP")
            download_gzip_to_text(url, out_file)
            downloaded += 1
            print(f"[OK]   downloaded {out_file.name}")
        except Exception as ex:
            failed.append(out_name)
            print(f"[WARN] failed {out_name}: {ex}")

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

    print("Preparing real-world undirected datasets...")
    nx_created, nx_skipped = fetch_networkx_datasets(force=args.force)
    snap_created, snap_skipped, snap_failed = fetch_snap_datasets(force=args.force)

    print("\nSummary")
    print("-------")
    print(f"NetworkX created: {nx_created}, skipped: {nx_skipped}")
    print(f"SNAP downloaded:  {snap_created}, skipped: {snap_skipped}")
    if snap_failed:
        print(f"SNAP failed:      {len(snap_failed)} -> {', '.join(snap_failed)}")


if __name__ == "__main__":
    main()
