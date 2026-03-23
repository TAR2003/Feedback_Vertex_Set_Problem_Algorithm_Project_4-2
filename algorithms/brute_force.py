"""
algorithms/brute_force.py
-------------------------
Exact brute-force FVS solver by exhaustive enumeration.
Only usable for n ≤ 20 due to exponential complexity.

Time complexity: O(2^n * (n + m))
"""

import logging
from itertools import combinations
from typing import Optional

import networkx as nx

from algorithms.base import FVSSolver
from data.validator import is_valid_fvs

logger = logging.getLogger(__name__)

MAX_VERTICES = 20  # Hard upper limit for brute-force applicability


class BruteForce(FVSSolver):
    """
    Ground-truth solver: enumerates all subsets of V in increasing size order
    and returns the first valid FVS found (guaranteed minimum).
    """

    def name(self) -> str:
        return "BruteForce"

    def short_name(self) -> str:
        return "BRUTE_FORCE"

    def solve(self, graph: nx.Graph, k: Optional[int] = None) -> tuple[set, dict]:
        """
        Solve FVS by exhaustive subset enumeration.

        Raises ValueError if n > MAX_VERTICES.

        Returns:
            (fvs_set, info_dict) where fvs_set is the minimum FVS found.
            Returns (None, info_dict) if the graph has no cycles (FVS = empty set
            would be valid — returned as empty set to signal true optimality).
        """
        n = graph.number_of_nodes()
        if n > MAX_VERTICES:
            raise ValueError(
                f"BruteForce refuses to run on n={n} > {MAX_VERTICES}. "
                "Use IterativeCompression or KernelizationBST for larger instances."
            )

        nodes = list(graph.nodes())
        iterations = 0

        # --- Fast path: graph is already acyclic (FVS = empty set) ---
        if not self._has_any_cycle(graph):
            logger.debug("Graph is already acyclic; FVS = {}")
            return set(), {"iterations": 0, "time_sec": 0.0, "memory_mb": 0.0,
                           "notes": "Graph already acyclic", "convergence": []}

        # --- Enumerate subsets by increasing size ---
        for size in range(1, n + 1):
            if k is not None and size > k:
                # Caller asked for FVS of size ≤ k; proved impossible
                break
            for subset in combinations(nodes, size):
                iterations += 1
                candidate = set(subset)
                if is_valid_fvs(graph, candidate):
                    logger.debug("BruteForce: found FVS of size %d after %d iterations",
                                 size, iterations)
                    return candidate, {
                        "iterations": iterations,
                        "time_sec": 0.0,
                        "memory_mb": 0.0,
                        "notes": f"Optimal; checked {iterations} subsets",
                        "convergence": [],
                    }

        # No FVS of size ≤ k found (or no FVS at all, which can't happen for connected graph)
        logger.warning("BruteForce: no FVS found after %d iterations", iterations)
        return set(nodes), {
            "iterations": iterations,
            "time_sec": 0.0,
            "memory_mb": 0.0,
            "notes": "No FVS found within size limit",
            "convergence": [],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _has_any_cycle(graph: nx.Graph) -> bool:
        """Return True if the graph has at least one cycle (lightweight check)."""
        try:
            nx.find_cycle(graph)
            return True
        except nx.NetworkXNoCycle:
            return False
