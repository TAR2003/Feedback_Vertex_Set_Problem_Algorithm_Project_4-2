"""
algorithms/memetic_ga.py
------------------------
Algorithm 3: Memetic Genetic Algorithm for FVS.

Combines evolutionary search (GA) with local hill-climbing (memetic component).
Handles large instances (n > 1000) where exact algorithms are intractable.

Encoding: permutation of vertices.
Decoding: greedy left-to-right scan — include vertex in FVS only if graph
          still has cycles without it.
"""

import logging
import random
from typing import Optional

import networkx as nx

from algorithms.base import FVSSolver
from data.validator import is_valid_fvs, has_cycle

logger = logging.getLogger(__name__)


class MemeticGA(FVSSolver):
    """
    Memetic Algorithm for FVS: Genetic Algorithm with local search refinement.

    Parameters are configurable via __init__ to support hyperparameter studies.
    """

    def __init__(
        self,
        population_size: int = 100,
        max_generations: int = 200,
        mutation_rate: float = 0.05,
        crossover_rate: float = 0.8,
        tournament_size: int = 3,
        local_search_iterations: int = 50,
        random_seed: int = 42,
    ):
        self.population_size        = population_size
        self.max_generations        = max_generations
        self.mutation_rate          = mutation_rate
        self.crossover_rate         = crossover_rate
        self.tournament_size        = tournament_size
        self.local_search_iterations = local_search_iterations
        self.random_seed            = random_seed

    def name(self) -> str:
        return "MemeticGA"

    def short_name(self) -> str:
        return "MEMETIC"

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def solve(self, graph: nx.Graph, k: Optional[int] = None) -> tuple[set, dict]:
        """
        Find a (near-optimal) FVS using the memetic genetic algorithm.

        Args:
            graph: Undirected NetworkX graph (not mutated).
            k:     Ignored for GA (heuristic; cannot guarantee ≤ k).

        Returns:
            (fvs_set, info_dict) where info_dict includes convergence history.
        """
        rng = random.Random(self.random_seed)
        nodes = list(graph.nodes())
        n = len(nodes)

        if n == 0:
            return set(), {"iterations": 0, "time_sec": 0.0, "memory_mb": 0.0,
                           "notes": "Empty graph", "convergence": []}

        # --- Step A: Initialize population ---
        population = self._initialize(nodes, rng, graph)

        best_fvs      = self._decode(population[0], graph)
        best_size     = len(best_fvs)
        convergence   = []  # list of (generation, best_fvs_size)

        # --- Step G: Main GA loop ---
        for gen in range(self.max_generations):
            # Evaluate fitness for all individuals
            fitnesses = [self._fitness(ind, graph) for ind in population]

            # Identify best individual index
            best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
            current_best_fvs = self._decode(population[best_idx], graph)

            # Apply local search to the best individual (memetic step)
            improved_perm = self._local_search(population[best_idx], graph, rng)
            improved_fvs  = self._decode(improved_perm, graph)
            if len(improved_fvs) < len(current_best_fvs):
                population[best_idx] = improved_perm
                current_best_fvs     = improved_fvs

            # Update global best
            if len(current_best_fvs) < best_size:
                best_fvs  = current_best_fvs
                best_size = len(best_fvs)

            convergence.append((gen, best_size))

            # Create new generation
            new_pop = [improved_perm]  # Elitism: keep best individual
            while len(new_pop) < self.population_size:
                p1 = self._tournament_select(population, fitnesses, rng)
                p2 = self._tournament_select(population, fitnesses, rng)
                child = self._crossover_ox(p1, p2, rng)
                child = self._mutate(child, graph, rng)
                new_pop.append(child)
            population = new_pop

        # Final validation & fallback
        if not is_valid_fvs(graph, best_fvs):
            logger.warning("GA produced invalid FVS; falling back to full vertex set")
            best_fvs = set(graph.nodes())

        return best_fvs, {
            "iterations": self.max_generations * self.population_size,
            "time_sec": 0.0,
            "memory_mb": 0.0,
            "notes": f"GA fvs_size={len(best_fvs)}, gens={self.max_generations}",
            "convergence": convergence,
        }

    # ------------------------------------------------------------------
    # Step A — Population initialization
    # ------------------------------------------------------------------

    def _initialize(self, nodes: list, rng: random.Random,
                    graph: nx.Graph) -> list:
        """
        Create initial population.

        Half random permutations, half greedy-biased (high-degree first
        with slight shuffle).
        """
        pop = []
        half = self.population_size // 2

        # Random half
        for _ in range(half):
            perm = nodes[:]
            rng.shuffle(perm)
            pop.append(perm)

        # Greedy-biased half: sort by degree desc, then lightly shuffle
        degree_order = sorted(nodes, key=lambda v: graph.degree(v), reverse=True)
        for _ in range(self.population_size - half):
            perm = degree_order[:]
            # Swap ~10% of adjacent pairs for diversity
            for i in range(max(1, len(perm) // 10)):
                a = rng.randint(0, len(perm) - 1)
                b = rng.randint(0, len(perm) - 1)
                perm[a], perm[b] = perm[b], perm[a]
            pop.append(perm)

        return pop

    # ------------------------------------------------------------------
    # Step B — Decode + Fitness
    # ------------------------------------------------------------------

    def _decode(self, perm: list, graph: nx.Graph) -> set:
        """
        Greedy decoder: process vertices in permutation order.
        For each vertex, check if the processed subgraph (excluding current FVS)
        has cycles. If yes, add the vertex to FVS.

        Always produces a valid FVS.
        """
        fvs: set = set()
        processed = set()

        for v in perm:
            processed.add(v)
            # Check: does the processed subgraph still have cycles?
            # (considering vertices not in fvs)
            test_vertices = [u for u in processed if u not in fvs]
            test_graph = graph.subgraph(test_vertices)
            if has_cycle(test_graph):
                # Need v in FVS to help break cycles
                fvs.add(v)

        return fvs

    def _fitness(self, perm: list, graph: nx.Graph) -> float:
        """Fitness = negative FVS size (maximize → minimize size)."""
        return -len(self._decode(perm, graph))

    # ------------------------------------------------------------------
    # Step C — Tournament selection
    # ------------------------------------------------------------------

    def _tournament_select(self, population: list, fitnesses: list,
                           rng: random.Random) -> list:
        """Return the best individual from a random tournament."""
        candidates = rng.sample(range(len(population)),
                                min(self.tournament_size, len(population)))
        winner = max(candidates, key=lambda i: fitnesses[i])
        return population[winner][:]

    # ------------------------------------------------------------------
    # Step D — Order Crossover (OX)
    # ------------------------------------------------------------------

    def _crossover_ox(self, p1: list, p2: list, rng: random.Random) -> list:
        """
        Order Crossover (OX) for permutations.

        Copies a random segment from p1 into child, then fills remaining
        positions with elements from p2 in their p2 order.
        """
        if rng.random() > self.crossover_rate:
            return p1[:]  # No crossover

        n = len(p1)
        if n < 2:
            return p1[:]

        a, b = sorted(rng.sample(range(n), 2))
        child = [None] * n
        segment = set(p1[a:b + 1])
        child[a:b + 1] = p1[a:b + 1]

        # Fill remaining slots from p2 in order
        p2_iter = (x for x in p2 if x not in segment)
        for i in list(range(0, a)) + list(range(b + 1, n)):
            child[i] = next(p2_iter)

        return child

    # ------------------------------------------------------------------
    # Step E — Mutation
    # ------------------------------------------------------------------

    def _mutate(self, perm: list, graph: nx.Graph, rng: random.Random) -> list:
        """
        Swap mutation: swap two random positions with probability mutation_rate.
        Bonus mutation: move a high-degree vertex toward the front.
        """
        perm = perm[:]
        n = len(perm)

        # Standard swap mutation
        if rng.random() < self.mutation_rate and n >= 2:
            i, j = rng.sample(range(n), 2)
            perm[i], perm[j] = perm[j], perm[i]

        # Bias: move a high-degree node to the front
        if rng.random() < self.mutation_rate / 2 and n >= 2:
            # Pick highest-degree remaining vertex not already at front
            top_v = max(perm[n // 2:], key=lambda v: graph.degree(v),
                        default=None)
            if top_v is not None:
                idx = perm.index(top_v)
                target = rng.randint(0, n // 4)
                perm[idx], perm[target] = perm[target], perm[idx]

        return perm

    # ------------------------------------------------------------------
    # Step F — Local search (hill climbing)
    # ------------------------------------------------------------------

    def _local_search(self, perm: list, graph: nx.Graph,
                      rng: random.Random) -> list:
        """
        Hill-climbing local search on the best individual.

        Tries vertex removals and swaps; accepts any improvement.
        This is the 'memetic' refinement step.
        """
        best_perm = perm[:]
        best_fvs  = self._decode(best_perm, graph)

        for _ in range(self.local_search_iterations):
            fvs = self._decode(best_perm, graph)
            fvs_list = list(fvs)
            non_fvs  = [v for v in best_perm if v not in fvs]
            improved = False

            # Try removing each FVS vertex
            for v in fvs_list:
                candidate = fvs - {v}
                if is_valid_fvs(graph, candidate):
                    # Reorder perm to push v to the back
                    best_perm.remove(v)
                    best_perm.append(v)
                    best_fvs = candidate
                    improved = True
                    break

            # Try swapping an FVS vertex with a non-FVS vertex
            if not improved and fvs_list and non_fvs:
                v = rng.choice(fvs_list)
                u = rng.choice(non_fvs)
                candidate = (fvs - {v}) | {u}
                if is_valid_fvs(graph, candidate) and len(candidate) <= len(best_fvs):
                    idx_v = best_perm.index(v)
                    idx_u = best_perm.index(u)
                    best_perm[idx_v], best_perm[idx_u] = (
                        best_perm[idx_u], best_perm[idx_v])
                    best_fvs = candidate

        return best_perm
