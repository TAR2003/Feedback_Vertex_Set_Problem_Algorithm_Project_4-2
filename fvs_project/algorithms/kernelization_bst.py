"""
algorithms/kernelization_bst.py
--------------------------------
Algorithm 2: Kernelization + Bounded Search Tree (KBST) — exact algorithm.

Time complexity: O(4^k * n^2) after kernelization reduces graph to ≤ k^2 + k vertices.
"""

import logging
from collections import deque
from typing import Optional

import networkx as nx

from algorithms.base import FVSSolver
from data.validator import is_valid_fvs, has_cycle, _estimate_fvs_lower_bound

logger = logging.getLogger(__name__)


class KernelizationBST(FVSSolver):
    """
    Exact FVS solver combining preprocessing (kernelization) with a bounded search tree.

    Reduction rules dramatically shrink the graph before branching,
    yielding fast practical performance even for moderate n.
    """

    def name(self) -> str:
        return "KernelizationBST"

    def short_name(self) -> str:
        return "KBST"

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def solve(self, graph: nx.Graph, k: Optional[int] = None) -> tuple[set, dict]:
        """
        Find minimum FVS using kernelization + bounded search tree.

        Args:
            graph: Undirected NetworkX graph (not mutated).
            k:     Target FVS size.  If None, search from lower bound.

        Returns:
            (fvs_set, info_dict)
        """
        iterations = [0]

        lb = _estimate_fvs_lower_bound(graph)
        k_start = lb if k is None else k
        k_max   = graph.number_of_nodes()

        result_fvs: Optional[set] = None
        k_used = k_start

        for k_try in range(k_start, k_max + 1):
            fvs = self._solve_for_k(graph.copy(), k_try, set(), iterations)
            if fvs is not None:
                result_fvs = fvs
                k_used = k_try
                break
            if k is not None:
                break

        if result_fvs is None:
            result_fvs = set(graph.nodes())

        return result_fvs, {
            "iterations": iterations[0],
            "time_sec": 0.0,
            "memory_mb": 0.0,
            "notes": f"KBST solution size={len(result_fvs)}, k={k_used}",
            "convergence": [],
        }

    # ------------------------------------------------------------------
    # Step A — Reduction rules (applied exhaustively)
    # ------------------------------------------------------------------

    def _apply_reductions(self, graph: nx.Graph, partial_sol: set,
                          k: int) -> tuple[nx.Graph, set, int]:
        """
        Apply FVS reduction rules to fixed-point.

        Rules (in efficiency order):
          1. Remove degree-0 vertices
          2. Remove degree-1 vertices
          3. Self-loop → must be in FVS
          5. High-degree (> 2k) → must be in FVS
          4. Degree-2 contraction (applied last)

        Returns (reduced_graph, updated_partial_sol, updated_k).
        Returns (graph, partial_sol, -1) if k becomes negative.
        """
        changed = True
        sol = set(partial_sol)
        g = graph

        while changed:
            changed = False

            # Rules 1, 2, 3, 5 — single pass over vertices
            to_delete_free: list = []
            to_force: list = []

            for v in list(g.nodes()):
                if not g.has_node(v):
                    continue
                deg = g.degree(v)
                # Rule 3: self-loop
                if g.has_edge(v, v):
                    to_force.append(v)
                    changed = True
                # Rule 1 & 2: degree ≤ 1
                elif deg <= 1:
                    to_delete_free.append(v)
                    changed = True
                # Rule 5: high degree > 2k
                elif k > 0 and deg > 2 * k:
                    to_force.append(v)
                    changed = True

            for v in to_delete_free:
                if g.has_node(v):
                    g = g.copy()
                    g.remove_node(v)
            for v in to_force:
                if g.has_node(v):
                    g = g.copy()
                    sol.add(v)
                    g.remove_node(v)
                    k -= 1
                    if k < 0:
                        return g, sol, -1  # Over-budget

            # Rule 4: degree-2 contraction
            for v in list(g.nodes()):
                if not g.has_node(v) or g.degree(v) != 2:
                    continue
                neighbors = list(g.neighbors(v))
                if len(neighbors) < 2:
                    continue
                u, w = neighbors[0], neighbors[1]
                if u == w:
                    continue  # Multi-edge case; handled by Rule 3/branching
                g = g.copy()
                # If (u, w) edge already exists → triangle; branch in BST
                if not g.has_edge(u, w):
                    # Safe to contract: remove v, add (u,w)
                    g.remove_node(v)
                    g.add_edge(u, w)
                    changed = True
                    break  # Restart inner loop after mutation

        return g, sol, k

    # ------------------------------------------------------------------
    # Step B — Feasibility check after kernelization
    # ------------------------------------------------------------------

    @staticmethod
    def _is_feasible(graph: nx.Graph, k: int) -> bool:
        """Return False if the kernel bound |V| > k^2 + k is violated."""
        return graph.number_of_nodes() <= k * k + k

    # ------------------------------------------------------------------
    # Step C — Bounded search tree on the kernel
    # ------------------------------------------------------------------

    def _solve_for_k(self, graph: nx.Graph, k: int, partial: set,
                     iterations: list) -> Optional[set]:
        """
        Recursively solve FVS on *graph* using BST, after kernelization.

        Returns a valid FVS of size ≤ k combined with *partial*, or None.
        """
        # Apply reductions
        g, sol, k_rem = self._apply_reductions(graph, partial, k)

        if k_rem < 0:
            return None  # Budget exceeded

        if not has_cycle(g):
            return sol  # Acyclic — done

        if not self._is_feasible(g, k_rem):
            return None  # Kernel too large for this k

        # Find shortest cycle (via BFS) to branch on
        cycle = self._find_shortest_cycle(g)
        if cycle is None:
            return sol  # No cycle found (shouldn't reach here normally)

        # Branch on each vertex in the cycle
        for v in cycle:
            iterations[0] += 1
            g2 = g.copy()
            g2.remove_node(v)
            result = self._solve_for_k(g2, k_rem - 1, sol | {v}, iterations)
            if result is not None:
                return result

        return None  # All branches exhausted

    # ------------------------------------------------------------------
    # Step D — Shortest cycle finder using BFS
    # ------------------------------------------------------------------

    @staticmethod
    def _find_shortest_cycle(graph: nx.Graph) -> Optional[list]:
        """
        Find a shortest cycle in the graph using BFS from each vertex.

        Returns list of vertices in the cycle, or None if acyclic.
        O(V * (V + E)).
        """
        best_cycle: Optional[list] = None

        for start in graph.nodes():
            # BFS to detect shortest cycle through *start*
            parent: dict = {start: None}
            dist: dict   = {start: 0}
            queue: deque = deque([start])

            while queue:
                v = queue.popleft()
                for w in graph.neighbors(v):
                    if w not in dist:
                        dist[w] = dist[v] + 1
                        parent[w] = v
                        queue.append(w)
                    elif parent[v] != w:  # Back-edge ≠ tree edge → cycle
                        # Reconstruct cycle
                        cycle: list = [v, w]
                        cur = v
                        while cur != start and parent[cur] is not None:
                            cur = parent[cur]
                            cycle.append(cur)
                        cycle_set = list(set(cycle))
                        if best_cycle is None or len(cycle_set) < len(best_cycle):
                            best_cycle = cycle_set
                        break  # One cycle per starting vertex is enough

        return best_cycle
