"""
data/generator.py
-----------------
Generate all synthetic graph instances and save them as GraphML files
in data/synthetic/.

Supports QUICK_MODE (n ≤ 200) via FVS_QUICK_MODE environment variable.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYNTHETIC_DIR = Path(__file__).parent / "synthetic"

# QUICK_MODE: use only small/medium instances (n ≤ 200)
QUICK_MODE: bool = os.environ.get("FVS_QUICK_MODE", "1") != "0"

# ER parameters
ER_N_VALUES = [10, 20, 50, 100, 200, 500, 1000]
ER_P_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]
ER_SEED     = 1

# BA parameters
BA_N_VALUES = [10, 20, 50, 100, 200, 500, 1000]
BA_M_VALUES = [2, 3, 5]
BA_SEED     = 1

# Grid sizes
GRID_SIZES  = [(3, 3), (5, 5), (10, 10), (20, 20), (30, 30)]

# WS parameters
WS_N_VALUES = [20, 50, 100, 200, 500]
WS_K_VALUES = [4, 6, 8]
WS_B_VALUES = [0.1, 0.3, 0.5]
WS_SEED     = 1

# Cycle-heavy: 4 density levels × 3 patterns
CH_DENSITIES = [0.10, 0.25, 0.50, 0.75]
CH_PATTERNS  = ["ring_cliques", "nested_cycles", "random_overlay"]

# Tree parameters
TREE_N_VALUES = [10, 20, 50, 100, 200]
TREE_TYPES    = ["random", "binary", "star"]


# ---------------------------------------------------------------------------
# Helper: save / skip logic
# ---------------------------------------------------------------------------

def _save_graph(graph: nx.Graph, path: Path) -> None:
    """Save graph as GraphML, skipping if file already exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        logger.debug("Skip (exists): %s", path.name)
        return
    nx.write_graphml(graph, str(path))
    logger.debug("Saved: %s", path.name)


def _quick_skip(n: int) -> bool:
    """Return True if QUICK_MODE is on and n > 200."""
    return QUICK_MODE and n > 200


# ---------------------------------------------------------------------------
# Graph generators
# ---------------------------------------------------------------------------

def _generate_er_graphs() -> list[Path]:
    """Generate Erdős–Rényi random graphs."""
    paths = []
    for n in ER_N_VALUES:
        if _quick_skip(n):
            continue
        for p in ER_P_VALUES:
            fname = SYNTHETIC_DIR / f"ER_n{n}_p{p}_seed{ER_SEED}.graphml"
            rng   = np.random.default_rng(ER_SEED)
            g     = nx.erdos_renyi_graph(n, p, seed=int(rng.integers(0, 2**31)))
            _save_graph(g, fname)
            paths.append(fname)
    logger.info("ER graphs: %d instances", len(paths))
    return paths


def _generate_ba_graphs() -> list[Path]:
    """Generate Barabási–Albert scale-free graphs."""
    paths = []
    for n in BA_N_VALUES:
        if _quick_skip(n):
            continue
        for m in BA_M_VALUES:
            if m >= n:
                continue
            fname = SYNTHETIC_DIR / f"BA_n{n}_m{m}_seed{BA_SEED}.graphml"
            g     = nx.barabasi_albert_graph(n, m, seed=BA_SEED)
            _save_graph(g, fname)
            paths.append(fname)
    logger.info("BA graphs: %d instances", len(paths))
    return paths


def _generate_grid_graphs() -> list[Path]:
    """Generate grid graphs of various sizes."""
    paths = []
    for rows, cols in GRID_SIZES:
        if _quick_skip(rows * cols):
            continue
        fname = SYNTHETIC_DIR / f"Grid_{rows}x{cols}.graphml"
        g     = nx.grid_2d_graph(rows, cols)
        # Relabel to integer nodes for compatibility
        g     = nx.convert_node_labels_to_integers(g)
        _save_graph(g, fname)
        paths.append(fname)
    logger.info("Grid graphs: %d instances", len(paths))
    return paths


def _generate_ws_graphs() -> list[Path]:
    """Generate Watts–Strogatz small-world graphs."""
    paths = []
    for n in WS_N_VALUES:
        if _quick_skip(n):
            continue
        for k in WS_K_VALUES:
            if k >= n:
                continue
            for b in WS_B_VALUES:
                b_str = str(b).replace(".", "")
                fname = SYNTHETIC_DIR / f"WS_n{n}_k{k}_b{b_str}_seed{WS_SEED}.graphml"
                g     = nx.watts_strogatz_graph(n, k, b, seed=WS_SEED)
                _save_graph(g, fname)
                paths.append(fname)
    logger.info("WS graphs: %d instances", len(paths))
    return paths


def _make_ring_of_cliques(n: int, extra_frac: float, seed: int) -> nx.Graph:
    """Ring of cliques: cliques connected in a ring, then extra edges added."""
    rng    = np.random.default_rng(seed)
    clique_size = max(3, n // 5)
    n_cliques   = max(3, n // clique_size)
    g = nx.ring_of_cliques(n_cliques, clique_size)
    g = nx.convert_node_labels_to_integers(g)
    _add_extra_edges(g, extra_frac, rng)
    return g


def _make_nested_cycles(n: int, extra_frac: float, seed: int) -> nx.Graph:
    """Nested cycles: an outer ring with inner triangles."""
    rng  = np.random.default_rng(seed)
    g    = nx.cycle_graph(max(4, n // 2))
    g    = nx.convert_node_labels_to_integers(g)
    cur  = g.number_of_nodes()
    # Add inner triangles
    for v in range(0, cur - 1, 3):
        w = cur + v // 3
        g.add_node(w)
        g.add_edge(v, w)
        g.add_edge(v + 1, w)
        if g.number_of_nodes() >= n:
            break
    g = nx.convert_node_labels_to_integers(g)
    _add_extra_edges(g, extra_frac, rng)
    return g


def _make_random_cycle_overlay(n: int, extra_frac: float, seed: int) -> nx.Graph:
    """Random spanning tree with additional edges to plant cycles."""
    rng = np.random.default_rng(seed)
    g   = nx.random_tree(n, seed=int(rng.integers(0, 2**31)))
    _add_extra_edges(g, extra_frac, rng)
    return g


def _add_extra_edges(g: nx.Graph, frac: float, rng: np.random.Generator) -> None:
    """Add frac * |V| extra random edges to the graph."""
    nodes  = list(g.nodes())
    n_add  = max(1, int(frac * len(nodes)))
    for _ in range(n_add * 3):  # Attempt 3× to account for existing edges
        u, v = rng.choice(nodes, 2, replace=False)
        if not g.has_edge(u, v) and u != v:
            g.add_edge(u, v)
            n_add -= 1
        if n_add <= 0:
            break


def _generate_cycle_heavy_graphs() -> list[Path]:
    """Generate cycle-heavy graphs with planted cycles."""
    paths   = []
    n_nodes = 50  # Base size (small enough for exact algorithms)
    for density in CH_DENSITIES:
        for pattern in CH_PATTERNS:
            d_str = str(int(density * 100))
            fname = SYNTHETIC_DIR / f"CycleHeavy_n{n_nodes}_density{d_str}_{pattern}.graphml"
            seed  = hash((density, pattern)) % (2**20)
            if pattern == "ring_cliques":
                g = _make_ring_of_cliques(n_nodes, density, seed)
            elif pattern == "nested_cycles":
                g = _make_nested_cycles(n_nodes, density, seed)
            else:
                g = _make_random_cycle_overlay(n_nodes, density, seed)
            _save_graph(g, fname)
            paths.append(fname)
    logger.info("CycleHeavy graphs: %d instances", len(paths))
    return paths


def _generate_tree_graphs() -> list[Path]:
    """Generate tree graphs (FVS should be empty for all trees)."""
    paths = []
    for n in TREE_N_VALUES:
        for ttype in TREE_TYPES:
            fname = SYNTHETIC_DIR / f"Tree_{ttype}_n{n}.graphml"
            if ttype == "random":
                g = nx.random_tree(n, seed=42)
            elif ttype == "binary":
                g = nx.balanced_tree(2, max(1, int(np.log2(n + 1))))
                g = nx.convert_node_labels_to_integers(g)
            else:  # star
                g = nx.star_graph(n - 1)
                g = nx.convert_node_labels_to_integers(g)
            _save_graph(g, fname)
            paths.append(fname)
    logger.info("Tree graphs: %d instances", len(paths))
    return paths


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_all(quick_mode: Optional[bool] = None) -> list[Path]:
    """
    Generate all synthetic graph instances.

    Args:
        quick_mode: Override QUICK_MODE env var if provided.

    Returns:
        List of paths to generated GraphML files.
    """
    global QUICK_MODE
    if quick_mode is not None:
        QUICK_MODE = quick_mode

    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Generating synthetic datasets (QUICK_MODE=%s)...", QUICK_MODE)

    all_paths: list[Path] = []
    all_paths.extend(_generate_er_graphs())
    all_paths.extend(_generate_ba_graphs())
    all_paths.extend(_generate_grid_graphs())
    all_paths.extend(_generate_ws_graphs())
    all_paths.extend(_generate_cycle_heavy_graphs())
    all_paths.extend(_generate_tree_graphs())

    logger.info("Total synthetic instances: %d", len(all_paths))
    return all_paths


def load_all_graphs(quick_mode: Optional[bool] = None) -> list[tuple[str, nx.Graph]]:
    """
    Load all generated synthetic graphs.

    Returns:
        List of (instance_id, graph) pairs. instance_id = file stem.
    """
    global QUICK_MODE
    if quick_mode is not None:
        QUICK_MODE = quick_mode

    graphs = []
    if not SYNTHETIC_DIR.exists():
        logger.warning("Synthetic directory not found: %s", SYNTHETIC_DIR)
        return graphs

    for fpath in sorted(SYNTHETIC_DIR.glob("*.graphml")):
        try:
            g = nx.read_graphml(str(fpath))
            # Ensure integer node IDs for consistency
            g = nx.convert_node_labels_to_integers(g)
            instance_id = fpath.stem
            # Skip oversized instances in QUICK_MODE
            if _quick_skip(g.number_of_nodes()):
                continue
            graphs.append((instance_id, g))
        except Exception as exc:
            logger.warning("Failed to load %s: %s", fpath.name, exc)

    logger.info("Loaded %d synthetic graphs", len(graphs))
    return graphs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    paths = generate_all()
    print(f"Generated {len(paths)} synthetic instances.")
