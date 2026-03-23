"""
data/validator.py
-----------------
Graph validation utilities: cycle detection, FVS validity checking,
and graph statistics computation.
"""

import logging
from typing import Optional

import networkx as nx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------

def has_cycle(graph: nx.Graph) -> bool:
    """
    DFS-based cycle detection on an undirected graph. O(V+E).

    Returns True if the graph contains at least one cycle, False otherwise.
    """
    visited: set = set()
    # parent dict maps node -> node that discovered it (to avoid trivial back-edges)
    parent: dict = {}

    def _dfs(node, par):
        visited.add(node)
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                parent[neighbor] = node
                if _dfs(neighbor, node):
                    return True
            elif neighbor != par:
                # Found a back-edge → cycle exists
                return True
        return False

    for v in graph.nodes():
        if v not in visited:
            parent[v] = None
            if _dfs(v, None):
                return True
    return False


# ---------------------------------------------------------------------------
# FVS validity check
# ---------------------------------------------------------------------------

def is_valid_fvs(graph: nx.Graph, fvs_set) -> bool:
    """
    Verify that *fvs_set* is a valid Feedback Vertex Set for *graph*.

    Removes the nodes in fvs_set from a copy of the graph and checks
    that the resulting graph is acyclic (a forest).

    Args:
        graph:   Undirected NetworkX graph (not mutated).
        fvs_set: Iterable of node IDs claimed to form an FVS.

    Returns:
        True if graph − fvs_set is acyclic, False otherwise.
    """
    if fvs_set is None:
        return False

    fvs = set(fvs_set)
    # Nodes in fvs_set that don't exist in the graph are silently ignored
    remaining_nodes = [v for v in graph.nodes() if v not in fvs]
    subgraph = graph.subgraph(remaining_nodes)
    return not has_cycle(subgraph)


# ---------------------------------------------------------------------------
# Graph statistics
# ---------------------------------------------------------------------------

def _estimate_fvs_lower_bound(graph: nx.Graph) -> int:
    """
    Lower bound on FVS size: |E| - |V| + c   where c = number of components.
    This equals the minimum number of edges that must be removed to make the
    graph a forest, which lower-bounds the number of vertices to remove.
    """
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    if n == 0:
        return 0
    c = nx.number_connected_components(graph)
    return max(0, m - n + c)


def graph_stats(graph: nx.Graph) -> dict:
    """
    Compute descriptive statistics for a graph.

    Returns a dict with keys:
        n_vertices, n_edges, density, avg_degree, max_degree,
        n_components, is_connected, estimated_fvs_lower_bound
    """
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    degrees = [d for _, d in graph.degree()]

    stats = {
        "n_vertices": n,
        "n_edges": m,
        "density": nx.density(graph),
        "avg_degree": (sum(degrees) / n) if n > 0 else 0.0,
        "max_degree": max(degrees) if degrees else 0,
        "n_components": nx.number_connected_components(graph),
        "is_connected": nx.is_connected(graph) if n > 0 else True,
        "estimated_fvs_lower_bound": _estimate_fvs_lower_bound(graph),
    }
    return stats
