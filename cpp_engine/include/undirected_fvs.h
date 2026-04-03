#pragma once
/**
 * @file undirected_fvs.h
 * @brief Undirected graph class used by all FVS algorithms.
 *
 * Represents an undirected graph using adjacency sets.
 * Supports O(log n) edge insertion/removal and O(1) vertex activation.
 *
 * Solver functions exposed to Python via pybind11:
 *   - solve_undirected_BST(n, edges) -> vector<int>  (Bounded Search Tree)
 *   - solve_undirected_IC(n, edges)  -> vector<int>  (Iterative Compression)
 *   - solve_undirected_MA(n, edges, pop, gens) -> vector<int> (Memetic)
 */

#include "graph_base.h"
#include <set>
#include <vector>
#include <utility>

struct UndirectedGraph;

bool kernelize_undirected(UndirectedGraph &g,
                          std::vector<int> &forced,
                          int &k);

// ─── Graph class ─────────────────────────────────────────────────────────────

struct UndirectedGraph : public GraphBase
{
    std::vector<std::set<int>> adj; ///< adj[v] = set of active neighbors of v

    /// Construct an empty undirected graph with n vertices
    explicit UndirectedGraph(int n);

    /// Add undirected edge {u, v}. Does nothing if already present.
    void add_edge(int u, int v) override;

    /// Remove undirected edge {u, v}. Does nothing if absent.
    void remove_edge(int u, int v) override;

    /**
     * Deactivate vertex v AND remove it from every neighbor's adjacency set.
     * Stores removed edges in `removed_edges` for later restoration.
     * @param removed_edges  Output vector; appended with (v, neighbor) pairs.
     */
    void deactivate_full(int v, std::vector<std::pair<int, int>> &removed_edges);

    /**
     * Undo a deactivate_full: restore v and re-add all edges in removed_edges.
     * Caller must pass the same vector produced by deactivate_full.
     */
    void reactivate_full(int v, const std::vector<std::pair<int, int>> &removed_edges);

    /// @return degree of v in the current active subgraph
    int degree(int v) const override;

    /**
     * O(n + m) DFS-based cycle check.
     * @return true if the active subgraph contains at least one cycle
     */
    bool has_cycle() const;

    /**
     * Find one cycle in the active subgraph.
     * Uses DFS with parent tracking to extract the back-edge cycle.
     * @return vertices of a cycle in order, or empty vector if graph is acyclic
     */
    std::vector<int> find_cycle() const;

    /**
     * Find a shortest cycle (girth witness) in the active subgraph.
     * Uses BFS from each active source vertex.
     * @return vertices of a shortest cycle in order, or empty if acyclic
     */
    std::vector<int> find_shortest_cycle() const;

    /// @return a deep copy of this graph (respects `active[]`)
    UndirectedGraph copy() const;
};

// ─── Solver declarations (implemented in exact_solver_u.cpp / memetic_u.cpp) ─

/**
 * Bounded Search Tree exact solver for undirected FVS.
 *
 * Strategy:
 *   1. Apply kernelization rules (degree-0/1 pruning, self-loop forced inclusion).
 *   2. Find a cycle C.  One vertex of C must be in the FVS.
 *   3. Branch on every vertex in C, recurse with budget k-1.
 *   4. Use iterative deepening to find minimum k.
 *
 * @param n     Number of vertices
 * @param edges List of (u, v) edge pairs (0-indexed)
 * @return      Minimum FVS as a vector of vertex indices
 */
std::vector<int> solve_undirected_BST(int n,
                                      const std::vector<std::pair<int, int>> &edges);

/**
 * Iterative Compression exact solver for undirected FVS.
 *
 * Strategy:
 *   1. Start with all vertices as a trivial FVS.
 *   2. Iteratively try to remove one vertex at a time while maintaining validity.
 *   3. After each removal attempt, call BST on the remaining problem to verify/fix.
 *   4. Compression step: given FVS of size k+1, enumerate k-subsets and validate.
 *
 * @param n     Number of vertices
 * @param edges List of (u, v) edge pairs (0-indexed)
 * @return      Minimum FVS as a vector of vertex indices
 */
std::vector<int> solve_undirected_IC(int n,
                                     const std::vector<std::pair<int, int>> &edges);

/**
 * Memetic Algorithm (Genetic Algorithm + Local Search) for undirected FVS.
 *
 * @param n          Number of vertices
 * @param edges      Edge list
 * @param pop_size   Population size (default 50)
 * @param max_gens   Maximum generations (default 200)
 * @return           Best FVS found
 */
std::vector<int> solve_undirected_MA(int n,
                                     const std::vector<std::pair<int, int>> &edges,
                                     int pop_size = 50, int max_gens = 200,
                                     int patience = 50);

/**
 * Kernelized Memetic Algorithm (KMA) for undirected FVS.
 *
 * Pipeline:
 *   1. Apply kernelization rules to collect forced FVS vertices.
 *   2. Solve the reduced kernel graph with MA.
 *   3. Map kernel solution back and merge with forced vertices.
 */
std::vector<int> solve_undirected_KME(int n,
                                      const std::vector<std::pair<int, int>> &edges,
                                      int pop_size = 50, int max_gens = 200,
                                      int patience = 30);

// Preferred KMA entry point (KME kept as backward-compatible alias).
std::vector<int> solve_undirected_KMA(int n,
                                      const std::vector<std::pair<int, int>> &edges,
                                      int pop_size = 50, int max_gens = 200,
                                      int patience = 30);