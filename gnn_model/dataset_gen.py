"""
dataset_gen.py
==============
Two-stage dataset pipeline for GNN training.

Stage A: Generate synthetic graph INPUTS as .txt files.
    - Writes to: data/ML_synthetic/(undirected|directed)

Stage B: Convert prepared .txt inputs into PyG .pt training samples.
    - Reads all .txt recursively from: data/ML_synthetic (default)
    - Writes .pt files to: data/synthetic/(undirected|directed)

The converter supports mixed input text formats:
    - Plain edge lists (u v) with optional extra columns
    - DIMACS / PACE headers (e.g., p edge N M, p dfvs N M)
    - METIS-style adjacency lists (n m t + n adjacency lines)

Output format (.pt):
    data.x          Node features
    data.edge_index COO edge index tensor
    data.y          Binary labels: y[v] = 1 if v is in optimal FVS
    data.fvs_size   Integer optimal FVS size
"""

import argparse
import sys
import random
import math
from pathlib import Path
from typing import List, Optional, Tuple

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
for candidate in ("build-linux", "build-macos", "build-win", "build"):
    sys.path.insert(0, str(PROJECT_ROOT / "cpp_engine" / candidate))

try:
    import cpp_engine
    HAS_ENGINE = True
except ImportError:
    print("WARNING: cpp_engine not found. Using Python fallback solver (slow).")
    HAS_ENGINE = False

try:
    import torch
    from torch_geometric.data import Data
    HAS_TORCH = True
except ImportError:
    print("WARNING: torch or torch_geometric not found. Saving raw .txt files instead.")
    HAS_TORCH = False

import networkx as nx


ML_SYNTH_DIR   = PROJECT_ROOT / "data" / "ML_synthetic"
SYNTHETIC_DIR  = PROJECT_ROOT / "data" / "synthetic"


# ═══════════════════════════════════════════════════════════════════════════════
#  Graph Generators
# ═══════════════════════════════════════════════════════════════════════════════

def gen_erdos_renyi_undirected(n: int, p: float) -> list:
    """Generate Erdős–Rényi undirected graph G(n, p). Returns edge list."""
    G = nx.erdos_renyi_graph(n, p, seed=random.randint(0, 10**6))
    return list(G.edges())

def gen_barabasi_albert(n: int, m: int = 2) -> list:
    """Generate scale-free undirected graph using Barabási–Albert model."""
    G = nx.barabasi_albert_graph(n, m, seed=random.randint(0, 10**6))
    return list(G.edges())

def gen_watts_strogatz(n: int, k: int = 4, beta: float = 0.3) -> list:
    """Generate small-world undirected graph using Watts–Strogatz model."""
    G = nx.watts_strogatz_graph(n, k, beta, seed=random.randint(0, 10**6))
    return list(G.edges())

def gen_random_directed(n: int, p: float) -> list:
    """Generate a random directed graph G(n, p)."""
    G = nx.gnp_random_graph(n, p, directed=True, seed=random.randint(0, 10**6))
    return list(G.edges())

def gen_directed_cycle_union(n: int, num_cycles: int = 3) -> list:
    """Generate directed graph as a union of random directed cycles."""
    edges = set()
    for _ in range(num_cycles):
        perm = list(range(n))
        random.shuffle(perm)
        for i in range(n):
            edges.add((perm[i], perm[(i + 1) % n]))
    return list(edges)


def write_graph_txt(path: Path, n: int, edges: List[Tuple[int, int]], graph_type: str) -> None:
    """Write one graph as a simple edge-list TXT with metadata comments."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# type={graph_type}\n")
        f.write(f"# n={n} m={len(edges)}\n")
        for u, v in edges:
            f.write(f"{u} {v}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  Feature Extraction
# ═══════════════════════════════════════════════════════════════════════════════

def compute_node_features_undirected(n: int, edges: list) -> list:
    """
    Compute per-node features for undirected graph.
    Returns a list of feature vectors (one per vertex).

    Features:
      [0] degree (normalized by n-1)
      [1] clustering coefficient
      [2] log(degree + 1) — captures scale-free structure
    """
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)

    degrees  = dict(G.degree())
    clust    = nx.clustering(G)
    features = []
    for v in range(n):
        deg  = degrees.get(v, 0)
        cc   = clust.get(v, 0.0)
        features.append([
            deg / max(n - 1, 1),
            cc,
            math.log(deg + 1) / math.log(n + 1)
        ])
    return features

def compute_node_features_directed(n: int, edges: list) -> list:
    """
    Features for directed graphs:
      [0] in-degree (normalized)
      [1] out-degree (normalized)
      [2] min(in, out) / (n-1)  — proxy for cycle participation
    """
    in_deg  = [0] * n
    out_deg = [0] * n
    for u, v in edges:
        out_deg[u] += 1
        in_deg[v]  += 1
    features = []
    for v in range(n):
        ind  = in_deg[v]
        outd = out_deg[v]
        features.append([
            ind  / max(n - 1, 1),
            outd / max(n - 1, 1),
            min(ind, outd) / max(n - 1, 1)
        ])
    return features


# ═══════════════════════════════════════════════════════════════════════════════
#  Solvers (wrapper with fallback)
# ═══════════════════════════════════════════════════════════════════════════════

def solve_undirected(n: int, edges: list) -> list:
    """Solve undirected FVS.  Use IC (faster than BST for training data)."""
    if HAS_ENGINE:
        return cpp_engine.solve_undirected_IC(n, edges)
    # Python fallback: greedy by degree
    adj = {v: set() for v in range(n)}
    for u, v in edges:
        adj[u].add(v); adj[v].add(u)
    fvs = []
    removed = set()
    def has_cycle():
        visited = set()
        def dfs(s, par):
            visited.add(s)
            for nb in adj[s]:
                if nb in removed: continue
                if nb == par: continue
                if nb in visited: return True
                if dfs(nb, s): return True
            return False
        for v in range(n):
            if v not in removed and v not in visited:
                if dfs(v, -1): return True
        return False
    while has_cycle():
        best = max((v for v in range(n) if v not in removed),
                   key=lambda v: len(adj[v] - removed))
        fvs.append(best); removed.add(best)
    return fvs

def solve_directed(n: int, edges: list) -> list:
    """Solve directed FVS."""
    if HAS_ENGINE:
        return cpp_engine.solve_directed_IC(n, edges)
    # Python fallback: greedy
    out_adj = {v: set() for v in range(n)}
    for u, v in edges:
        out_adj[u].add(v)
    fvs = []
    removed = set()
    def has_dcycle():
        color = {}
        def dfs(u):
            color[u] = 1
            for nb in out_adj[u]:
                if nb in removed: continue
                c = color.get(nb, 0)
                if c == 1: return True
                if c == 0 and dfs(nb): return True
            color[u] = 2
            return False
        for v in range(n):
            if v not in removed and color.get(v, 0) == 0:
                if dfs(v): return True
        return False
    while has_dcycle():
        best = max((v for v in range(n) if v not in removed),
                   key=lambda v: len(out_adj[v] - removed))
        fvs.append(best); removed.add(best)
    return fvs


# ═══════════════════════════════════════════════════════════════════════════════
#  TXT Parsing (robust mixed-format ingestion)
# ═══════════════════════════════════════════════════════════════════════════════

def _is_int_token(tok: str) -> bool:
    return tok.lstrip("+-").isdigit()


def _read_non_comment_lines(filepath: Path) -> List[str]:
    lines: List[str] = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(("#", "%")):
                continue
            if line.lower().startswith("c "):
                continue
            lines.append(line)
    return lines


def _detect_metis(lines: List[str]) -> bool:
    """Heuristic detection for METIS-style adjacency list format."""
    if not lines:
        return False
    first = lines[0].split()
    if len(first) < 2 or len(first) > 4 or not all(_is_int_token(x) for x in first[:2]):
        return False

    # Typical METIS starts with: n m [t]
    n = int(first[0])
    m = int(first[1])
    if n <= 0:
        return False
    if m < 0:
        return False

    # METIS has exactly n adjacency lines after header.
    # Require strict shape to avoid misclassifying plain edge lists.
    if len(lines) != n + 1:
        return False

    # Sample a few following lines; they should be integer token lists.
    sample = lines[1:min(len(lines), 1 + min(n, 5))]
    if not sample:
        return False
    return all(all(_is_int_token(t) for t in ln.split()) for ln in sample)


def parse_graph_txt(filepath: Path, default_type: str = "undirected") -> Tuple[str, int, List[Tuple[int, int]]]:
    """
    Parse mixed-format graph text and normalize vertices to 0-indexing.

    Returns:
      (graph_type, n, edges)
      graph_type is 'undirected' or 'directed'
    """
    lines = _read_non_comment_lines(filepath)
    if not lines:
        raise ValueError(f"No parseable lines found in {filepath}")

    n_hint: Optional[int] = None
    graph_type = default_type
    edges: List[Tuple[int, int]] = []

    # PACE / DIMACS style header detection.
    for line in lines:
        parts = line.split()
        if parts and parts[0].lower() == "p":
            # Examples: p edge N M, p dfvs N M, p fvs N M
            if len(parts) >= 3 and _is_int_token(parts[2]):
                n_hint = int(parts[2])
            if len(parts) >= 2:
                kind = parts[1].lower()
                if kind in {"dfvs"}:
                    graph_type = "directed"
                elif kind in {"edge", "fvs"}:
                    graph_type = "undirected"
            break

    # Strong filename/folder hints when available.
    lowered_path = str(filepath).lower()
    if "directed" in lowered_path:
        graph_type = "directed"
    elif "undirected" in lowered_path:
        graph_type = "undirected"

    # Parse METIS adjacency if detected and no explicit header already declared format.
    if _detect_metis(lines) and not any(ln.split()[0].lower() == "p" for ln in lines if ln.split()):
        header = lines[0].split()
        n = int(header[0])
        data_lines = lines[1:1 + n]
        metis_edges: List[Tuple[int, int]] = []
        for u, ln in enumerate(data_lines):
            parts = ln.split()
            for tok in parts:
                if _is_int_token(tok):
                    v = int(tok) - 1
                    if 0 <= v < n:
                        metis_edges.append((u, v))
        return graph_type, n, metis_edges

    # Generic edge-list parsing.
    for line in lines:
        parts = line.split()
        if not parts:
            continue

        # Skip known headers/non-data lines
        if parts[0].lower() == "p":
            continue
        if not _is_int_token(parts[0]):
            continue
        if len(parts) < 2 or not _is_int_token(parts[1]):
            continue

        u = int(parts[0])
        v = int(parts[1])
        edges.append((u, v))

    if not edges:
        raise ValueError(f"No edges parsed from {filepath}")

    all_verts = set()
    for u, v in edges:
        all_verts.add(u)
        all_verts.add(v)
    min_v = min(all_verts)
    max_v = max(all_verts)

    # Normalize 1-indexed input to 0-indexed.
    if min_v == 1 and 0 not in all_verts:
        edges = [(u - 1, v - 1) for u, v in edges]
        max_v -= 1

    n = n_hint if n_hint is not None else (max_v + 1)
    n = max(n, max_v + 1)

    if graph_type == "undirected":
        # Canonicalize and deduplicate undirected edges.
        dedup = set()
        for u, v in edges:
            if u == v:
                dedup.add((u, v))
            else:
                dedup.add((u, v) if u < v else (v, u))
        edges = sorted(dedup)
    else:
        edges = sorted(set(edges))

    return graph_type, n, edges


def build_and_save_pt(graph_type: str, n: int, edges: List[Tuple[int, int]], out_path: Path) -> None:
    """Create one labeled PyG sample from parsed TXT graph and save it."""
    if not HAS_TORCH:
        raise RuntimeError("torch/torch_geometric are required for .pt generation.")

    if graph_type == "directed":
        feats = compute_node_features_directed(n, edges)
        fvs = solve_directed(n, edges)
    else:
        feats = compute_node_features_undirected(n, edges)
        fvs = solve_undirected(n, edges)

    x = torch.tensor(feats, dtype=torch.float)
    y = torch.zeros(n, dtype=torch.long)
    for v in fvs:
        if 0 <= v < n:
            y[v] = 1

    if edges:
        ei = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
        if graph_type == "undirected":
            ei = torch.cat([ei, ei.flip(0)], dim=1)
    else:
        ei = torch.zeros((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=ei, y=y)
    data.fvs_size = len(fvs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_path)


# ═══════════════════════════════════════════════════════════════════════════════
#  Dataset Generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_undirected_txt_inputs(n_graphs: int, max_n: int, out_dir: Path):
    """Generate undirected graph input TXT files (no labels)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating {n_graphs} undirected TXT inputs (max n={max_n}) -> {out_dir}")

    for i in range(n_graphs):
        n   = random.randint(5, max_n)
        p   = random.uniform(0.2, 0.6)
        gen = random.choice([
            lambda: gen_erdos_renyi_undirected(n, p),
            lambda: gen_barabasi_albert(n, random.randint(1, max(1, n // 4))),
            lambda: gen_watts_strogatz(n)
        ])
        edges = gen()
        write_graph_txt(out_dir / f"ugraph_{i:05d}.txt", n, edges, "undirected")

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n_graphs} generated")

    print("  TXT input generation done.")


def generate_directed_txt_inputs(n_graphs: int, max_n: int, out_dir: Path):
    """Generate directed graph input TXT files (no labels)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating {n_graphs} directed TXT inputs (max n={max_n}) -> {out_dir}")

    for i in range(n_graphs):
        n   = random.randint(5, max_n)
        p   = random.uniform(0.2, 0.5)
        gen = random.choice([
            lambda: gen_random_directed(n, p),
            lambda: gen_directed_cycle_union(n, random.randint(2, 5))
        ])
        edges = gen()
        write_graph_txt(out_dir / f"dgraph_{i:05d}.txt", n, edges, "directed")

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n_graphs} generated")

    print("  TXT input generation done.")


def convert_txt_folder_to_pt(
    input_dir: Path,
    output_dir: Path,
    mode_type: str,
    fail_fast: bool = False,
) -> None:
    """
    Convert all TXT files under input_dir recursively into labeled .pt graphs.

    mode_type controls conversion targets:
      - undirected: only undirected
      - directed: only directed
      - both: both graph types
    """
    if not input_dir.exists():
        print(f"TXT input folder does not exist: {input_dir}")
        return

    txt_files = sorted(input_dir.rglob("*.txt"))
    if not txt_files:
        print(f"No .txt files found under {input_dir}")
        return

    print(f"Converting {len(txt_files)} TXT files from {input_dir} -> {output_dir}")
    converted = 0
    skipped = 0

    for i, txt_path in enumerate(txt_files, start=1):
        try:
            parsed_type, n, edges = parse_graph_txt(
                txt_path,
                default_type="undirected" if mode_type == "undirected" else "directed" if mode_type == "directed" else "undirected",
            )

            if mode_type != "both" and parsed_type != mode_type:
                skipped += 1
                continue

            rel = txt_path.relative_to(input_dir)
            stem = rel.with_suffix("")
            out_name = str(stem).replace("\\", "__").replace("/", "__") + ".pt"
            out_path = output_dir / parsed_type / out_name

            build_and_save_pt(parsed_type, n, edges, out_path)
            converted += 1

            if i % 50 == 0:
                print(f"  Progress: {i}/{len(txt_files)} scanned, {converted} converted")

        except Exception as ex:
            print(f"  [WARN] Failed {txt_path}: {ex}")
            if fail_fast:
                raise

    print(f"Conversion done. Converted={converted}, Skipped={skipped}, Total={len(txt_files)}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate TXT graph inputs and convert TXT -> .pt for GNN")
    parser.add_argument("--mode",     default="all", choices=["generate_txt", "txt_to_pt", "all"])
    parser.add_argument("--type",     default="both", choices=["undirected", "directed", "both"])
    parser.add_argument("--n_graphs", type=int, default=1000)
    parser.add_argument("--max_n",    type=int, default=50)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--txt_out_dir",   type=str, default=str(ML_SYNTH_DIR))
    parser.add_argument("--txt_input_dir", type=str, default=None)
    parser.add_argument("--pt_out_dir",    type=str, default=str(SYNTHETIC_DIR))
    parser.add_argument("--fail_fast", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    txt_out_dir = Path(args.txt_out_dir)
    # If not provided, read from the same folder where TXT files are generated.
    txt_in_dir = Path(args.txt_input_dir) if args.txt_input_dir else txt_out_dir
    pt_out_dir = Path(args.pt_out_dir)

    if args.mode in ("generate_txt", "all"):
        if args.type in ("undirected", "both"):
            generate_undirected_txt_inputs(args.n_graphs, args.max_n, txt_out_dir / "undirected")
        if args.type in ("directed", "both"):
            generate_directed_txt_inputs(args.n_graphs, args.max_n, txt_out_dir / "directed")

    if args.mode in ("txt_to_pt", "all"):
        convert_txt_folder_to_pt(
            input_dir=txt_in_dir,
            output_dir=pt_out_dir,
            mode_type=args.type,
            fail_fast=args.fail_fast,
        )

if __name__ == "__main__":
    main()