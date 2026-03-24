"""
algorithms/iterative_compression.py
------------------------------------
Algorithm 1: Iterative Compression (IC) — exact FPT algorithm.

Time complexity: O(5^k * k * n^2)
Guarantees optimal (minimum) FVS.
"""

import logging
from typing import Optional

import networkx as nx

from algorithms.base import FVSSolver
from data.validator import is_valid_fvs, has_cycle, _estimate_fvs_lower_bound

logger = logging.getLogger(__name__)


class IterativeCompression(FVSSolver):
    """
    Exact FPT algorithm for FVS using iterative compression.

    The algorithm processes vertices one at a time, maintaining a small
    'compression set' F.  When |F| exceeds k, it calls the compression
    subroutine to shrink F back to size ≤ k.
    """

    def name(self) -> str:
        return "IterativeCompression"

    def short_name(self) -> str:
        return "IC"

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def solve(self, graph: nx.Graph, k: Optional[int] = None) -> tuple[set, dict]:
        """
        Find minimum FVS using iterative compression.

        Args:
            graph: Input undirected graph (not mutated).
            k:     Upper bound on FVS size.  If None, find minimum by linear search.

        Returns:
            (fvs_set, info_dict)
        """
        g = graph.copy()
        iterations = [0]  # mutable counter shared across helpers

        # Determine search range for k
        lb = _estimate_fvs_lower_bound(g)
        k_start = lb if k is None else k
        k_max   = len(g.nodes())  # worst case: entire vertex set

        result_fvs: Optional[set] = None
        k_used = k_start

        for k_try in range(k_start, k_max + 1):
            fvs = self._ic_main(g, k_try, iterations)
            if fvs is not None:
                result_fvs = fvs
                k_used = k_try
                break
            if k is not None:
                # Caller specified k; don't try larger values
                break

        if result_fvs is None:
            result_fvs = set(graph.nodes())  # fallback: remove everything

        return result_fvs, {
            "iterations": iterations[0],
            "time_sec": 0.0,
            "memory_mb": 0.0,
            "notes": f"IC solution size={len(result_fvs)}, k={k_used}",
            "convergence": [],
        }

    # ------------------------------------------------------------------
    # Step A — Main iterative compression loop
    # ------------------------------------------------------------------

    def _ic_main(self, graph: nx.Graph, k: int, iterations: list) -> Optional[set]:
        """
        Process vertices one by one, compressing F whenever |F| == k+1.

        Returns a valid FVS of size ≤ k, or None if none exists.
        """
        nodes = list(graph.nodes())
        if not nodes:
            return set()

        # Start with the first node (not empty!)
        F: set = {nodes[0]}

        for idx, v in enumerate(nodes[1:], 1):
            F.add(v)
            iterations[0] += 1

            # If F exceeds k, try to compress it
            if len(F) > k + 1:
                return None  # Impossible to compress further

            if len(F) == k + 1:
                # Work with the subgraph induced by first idx+1 nodes
                # Must use .copy() because subgraph() returns a read-only view
                subgraph = graph.subgraph(nodes[:idx+1]).copy()
                compressed = self._compress(subgraph, F, k, iterations)
                if compressed is None:
                    return None
                F = compressed

        return F if len(F) <= k else None

    # ------------------------------------------------------------------
    # Step B — Compression subroutine
    # ------------------------------------------------------------------

    def _compress(self, graph: nx.Graph, F: set, k: int,
                  iterations: list) -> Optional[set]:
        """
        Given |F| == k+1, find an FVS of size ≤ k.

        Enumerates all 2^(k+1) partitions (F1, F2) of F:
          F1 = vertices we keep in the FVS
          F2 = vertices we try to move outside the FVS
        """
        F_list = list(F)
        n_F = len(F_list)

        # Enumerate all subsets of F to form (F1, F2) pairs
        for mask in range(1 << n_F):
            F1 = {F_list[i] for i in range(n_F) if mask & (1 << i)}
            F2 = F - F1
            iterations[0] += 1

            # F2 must induce a forest in G (independent forest check)
            if not self._is_independent_forest(graph, F2):
                continue

            # Remaining budget after keeping F1 in the FVS
            k_remaining = k - len(F1)
            if k_remaining < 0:
                continue

            # Find FVS of size ≤ k_remaining in G − F2, disjoint from F2
            g_minus_F2 = graph.copy()
            g_minus_F2.remove_nodes_from(F2)

            Y = self._disjoint_fvs(g_minus_F2, F2, k_remaining, iterations)
            if Y is not None:
                candidate = F1 | Y
                # Validate: ensure candidate is a valid FVS for the full graph
                if is_valid_fvs(graph, candidate):
                    return candidate

        return None  # All partitions failed

    # ------------------------------------------------------------------
    # Step C — Disjoint bounded search tree
    # ------------------------------------------------------------------

    def _disjoint_fvs(self, graph: nx.Graph, forbidden: set, k: int,
                      iterations: list) -> Optional[set]:
        """
        Find an FVS of size ≤ k in *graph* that is disjoint from *forbidden*.

        Uses a bounded search tree: pick a cycle, branch on each vertex in it.
        Applies degree-0/1 reduction rules before branching.
        """
        # Apply reduction: remove nodes that cannot be in any cycle
        g, forced = self._apply_reductions(graph, forbidden, k)
        if forced is None:
            return None  # Reduction consumed more than k vertices
        k -= len(forced)

        if k < 0:
            return None

        if not has_cycle(g):
            return forced  # Acyclic — reductions suffice

        # Find a cycle and branch on one of its vertices
        cycle = self._find_cycle(g)
        if cycle is None:
            return forced

        # Try branching on each cycle vertex that is not forbidden
        branchable = [v for v in cycle if v not in forbidden]
        if not branchable:
            # All cycle vertices are forbidden — cannot solve this subproblem
            return None

        for v in branchable:
            iterations[0] += 1
            g2 = g.copy()
            g2.remove_node(v)
            sub_result = self._disjoint_fvs(g2, forbidden, k - 1, iterations)
            if sub_result is not None:
                return forced | {v} | sub_result

        return None  # All branches failed

    # ------------------------------------------------------------------
    # Step D — Cycle finder
    # ------------------------------------------------------------------

    @staticmethod
    def _find_cycle(graph: nx.Graph) -> Optional[list]:
        """
        DFS-based cycle finder. Returns list of vertex IDs in a cycle.
        Returns None if graph is acyclic. O(V+E).
        """
        visited: dict = {}  # node -> parent
        cycle_nodes: list = []

        def dfs(v, parent):
            visited[v] = parent
            for w in graph.neighbors(v):
                if w == parent:
                    continue
                if w in visited:
                    # Reconstruct cycle: from w back to v via parent chain
                    path = [w, v]
                    cur = v
                    max_steps = len(graph)  # Safety limit
                    steps = 0
                    while cur != w and steps < max_steps:
                        if cur not in visited:
                            break  # Malformed parent chain; stop
                        cur = visited[cur]
                        path.append(cur)
                        steps += 1
                    cycle_nodes.extend(path)
                    return True
                if dfs(w, v):
                    return True
            return False

        for node in graph.nodes():
            if node not in visited:
                if dfs(node, None):
                    return list(set(cycle_nodes))
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_independent_forest(graph: nx.Graph, vertex_set: set) -> bool:
        """Return True if the subgraph induced by vertex_set is a forest."""
        if not vertex_set:
            return True
        subg = graph.subgraph(vertex_set)
        return not has_cycle(subg)

    @staticmethod
    def _apply_reductions(graph: nx.Graph, forbidden: set,
                          k: int) -> tuple[nx.Graph, Optional[set]]:
        """
        Remove degree-0 and degree-1 vertices (they cannot be in any cycle).
        Force self-loop vertices into the FVS.

        Returns (reduced_graph, forced_set) or (graph, None) if k exhausted.
        """
        g = graph.copy()
        forced: set = set()
        changed = True
        while changed:
            changed = False
            to_remove_free = []
            to_force = []
            for v in list(g.nodes()):
                # Self-loop: must be in FVS
                if g.has_edge(v, v):
                    to_force.append(v)
                    changed = True
                # Degree 0 or 1: cannot be in any cycle
                elif g.degree(v) <= 1:
                    to_remove_free.append(v)
                    changed = True
            for v in to_remove_free:
                if g.has_node(v):
                    g.remove_node(v)
            for v in to_force:
                if g.has_node(v):
                    forced.add(v)
                    g.remove_node(v)
                    k -= 1
                    if k < 0:
                        return g, None
        return g, forced
