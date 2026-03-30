"""
dataset_gen.py
==============
Generates training data for the GNN model.

For each synthetic graph:
  1. Generate the graph (Erdős–Rényi, scale-free, small-world)
  2. Solve it exactly using cpp_engine (BST for small, IC for medium)
  3. Save as a PyTorch Geometric Data object

Output format (saved as .pt files):
  data.x          — node features: [degree, clustering_coeff, betweenness (approx)]
  data.edge_index — edge index tensor (COO format)
  data.y          — binary labels: y[v] = 1 if v is in the optimal FVS, else 0
  data.fvs_size   — integer, size of optimal FVS

Usage:
  python gnn_model/dataset_gen.py --type undirected --n_graphs 1000 --max_n 50
  python gnn_model/dataset_gen.py --type directed   --n_graphs 500  --max_n 30
  python gnn_model/dataset_gen.py --type both       --n_graphs 2000 --max_n 80 --seed 42
"""

import argparse
import sys
import random
import math
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "cpp_engine" / "build"))

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
import os


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
#  Dataset Generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_undirected_dataset(n_graphs: int, max_n: int, out_dir: Path):
    """Generate undirected FVS dataset."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating {n_graphs} undirected graphs (max n={max_n}) → {out_dir}")

    for i in range(n_graphs):
        n   = random.randint(5, max_n)
        p   = random.uniform(0.2, 0.6)
        gen = random.choice([
            lambda: gen_erdos_renyi_undirected(n, p),
            lambda: gen_barabasi_albert(n, random.randint(1, max(1, n // 4))),
            lambda: gen_watts_strogatz(n)
        ])
        edges = gen()
        fvs   = solve_undirected(n, edges)

        if HAS_TORCH:
            feats = compute_node_features_undirected(n, edges)
            x     = torch.tensor(feats, dtype=torch.float)
            y     = torch.zeros(n, dtype=torch.long)
            for v in fvs: y[v] = 1

            if edges:
                ei = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
                # Add reverse edges (undirected → bidirected in edge_index)
                ei = torch.cat([ei, ei.flip(0)], dim=1)
            else:
                ei = torch.zeros((2, 0), dtype=torch.long)

            data = Data(x=x, edge_index=ei, y=y)
            data.fvs_size = len(fvs)
            torch.save(data, out_dir / f"graph_{i:05d}.pt")
        else:
            # Save as simple text
            with open(out_dir / f"graph_{i:05d}.txt", "w") as f:
                f.write(f"# n={n} fvs_size={len(fvs)}\n")
                f.write(f"# fvs={fvs}\n")
                for u, v in edges:
                    f.write(f"{u} {v}\n")

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n_graphs} generated")

    print("  Done.")

def generate_directed_dataset(n_graphs: int, max_n: int, out_dir: Path):
    """Generate directed FVS dataset."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating {n_graphs} directed graphs (max n={max_n}) → {out_dir}")

    for i in range(n_graphs):
        n   = random.randint(5, max_n)
        p   = random.uniform(0.2, 0.5)
        gen = random.choice([
            lambda: gen_random_directed(n, p),
            lambda: gen_directed_cycle_union(n, random.randint(2, 5))
        ])
        edges = gen()
        fvs   = solve_directed(n, edges)

        if HAS_TORCH:
            feats = compute_node_features_directed(n, edges)
            x     = torch.tensor(feats, dtype=torch.float)
            y     = torch.zeros(n, dtype=torch.long)
            for v in fvs: y[v] = 1

            if edges:
                ei = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
            else:
                ei = torch.zeros((2, 0), dtype=torch.long)

            data = Data(x=x, edge_index=ei, y=y)
            data.fvs_size = len(fvs)
            torch.save(data, out_dir / f"dgraph_{i:05d}.pt")
        else:
            with open(out_dir / f"dgraph_{i:05d}.txt", "w") as f:
                f.write(f"# n={n} fvs_size={len(fvs)}\n")
                f.write(f"# fvs={fvs}\n")
                for u, v in edges:
                    f.write(f"{u} {v}\n")

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n_graphs} generated")

    print("  Done.")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate FVS training datasets")
    parser.add_argument("--type",     default="both", choices=["undirected", "directed", "both"])
    parser.add_argument("--n_graphs", type=int, default=1000)
    parser.add_argument("--max_n",    type=int, default=50)
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    data_dir = PROJECT_ROOT / "data" / "synthetic"

    if args.type in ("undirected", "both"):
        generate_undirected_dataset(
            args.n_graphs, args.max_n,
            data_dir / "undirected"
        )
    if args.type in ("directed", "both"):
        generate_directed_dataset(
            args.n_graphs, args.max_n,
            data_dir / "directed"
        )

if __name__ == "__main__":
    main()