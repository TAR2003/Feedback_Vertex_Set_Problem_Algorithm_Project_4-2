"""
algorithms/base.py
------------------
Abstract base class for all FVS solver implementations.
"""

from abc import ABC, abstractmethod
from typing import Optional

import networkx as nx


class FVSSolver(ABC):
    """
    Abstract base class every FVS algorithm must inherit from.

    solve() returns a (fvs_set, info_dict) pair where:
      fvs_set  — set of node IDs that form the Feedback Vertex Set
      info_dict — dict with performance metadata
    """

    @abstractmethod
    def solve(self, graph: nx.Graph, k: Optional[int] = None) -> tuple[set, dict]:
        """
        Find a minimum (or near-minimum) Feedback Vertex Set.

        Args:
            graph: Undirected NetworkX graph. Must NOT be mutated.
            k:     Target FVS size hint. If None, solver discovers minimum k.

        Returns:
            (fvs_set, info_dict)
            fvs_set:  set of node IDs
            info_dict: {
                'iterations': int,
                'time_sec':   float,
                'memory_mb':  float,
                'notes':      str,
                'convergence': list[tuple[int, int]]  # (generation, fvs_size) for GA
            }
        """

    def name(self) -> str:
        """Human-readable algorithm name."""
        return self.__class__.__name__

    def short_name(self) -> str:
        """Short identifier used in performance.csv and report.csv."""
        return self.__class__.__name__.upper()
