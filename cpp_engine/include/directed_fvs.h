#pragma once
/**
 * @file directed_fvs.h
 * @brief Directed graph class and solver declarations for DFVS.
 *
 * Key insight for directed FVS:
 *   A directed graph G has a cycle iff it is NOT a DAG.
 *   Cycles in a digraph only pass through vertices that lie in non-trivial
 *   Strongly Connected Components (SCCs).  Any vertex whose SCC is trivial
 *   (size 1, no self-loop) can NEVER be part of a cycle → safe to ignore.
 *
 * Tarjan's SCC algorithm (O(n + m)) is used to decompose the graph and
 * focus the exact solvers only on the cyclic parts.
 *
 * Solver functions exposed to Python:
 *   - solve_directed_BST(n, edges) -> vector<int>
 *   - solve_directed_IC(n, edges)  -> vector<int>
 *   - solve_directed_MA(n, edges, pop, gens) -> vector<int>
 */

#include "graph_base.h"
#include <set>
#include <vector>
#include <utility>

struct DirectedGraph;

bool kernelize_directed(DirectedGraph &g,
                        std::vector<int> &forced,
                        int &k);

// ─── Graph class ─────────────────────────────────────────────────────────────

struct DirectedGraph : public GraphBase
{
    std::vector<std::set<int>> out_adj; ///< out_adj[v] = successors of v
    std::vector<std::set<int>> in_adj;  ///< in_adj[v]  = predecessors of v

    /// Construct empty directed graph with n vertices
    explicit DirectedGraph(int n);

    /// Add directed edge u → v
    void add_edge(int u, int v) override;

    /// Remove directed edge u → v
    void remove_edge(int u, int v) override;

    /**
     * Deactivate vertex v: mark inactive, remove from all adjacency sets.
     * Fills `removed` with (u, v) pairs for each removed edge.
     */
    void deactivate_full(int v, std::vector<std::pair<int, int>> &removed);

    /**
     * Undo deactivate_full by restoring v and re-inserting all edges in `removed`.
     */
    void reactivate_full(int v, const std::vector<std::pair<int, int>> &removed);

    /// In-degree of v in the active subgraph
    int in_degree(int v) const;

    /// Out-degree of v in the active subgraph
    int out_degree(int v) const;

    /// For GraphBase::degree(); returns in_degree + out_degree
    int degree(int v) const override;

    /**
     * Check for directed cycles using DFS coloring.
     * Colors: WHITE=0, GRAY=1 (on stack), BLACK=2 (done)
     * @return true if there exists a directed cycle in the active subgraph
     */
    bool has_directed_cycle() const;

    /**
     * Find one directed cycle using DFS with ancestor tracking.
     * @return vertices of a directed cycle, or empty if the graph is a DAG
     */
    std::vector<int> find_directed_cycle() const;

    /**
     * Find a shortest directed cycle in the active subgraph.
     * Uses BFS from each active source vertex.
     * @return vertices of a shortest directed cycle, or empty if the graph is a DAG
     */
    std::vector<int> find_shortest_directed_cycle() const;

    /**
     * Tarjan's SCC algorithm. O(n + m).
     * @return list of SCCs; each SCC is a list of vertex indices.
     *         Only non-trivial SCCs (size > 1, or size 1 with self-loop)
     *         need to be considered for FVS.
     */
    std::vector<std::vector<int>> find_SCCs() const;

    /// @return a deep copy of this graph
    DirectedGraph copy() const;
};

// ─── Solver declarations ──────────────────────────────────────────────────────

/**
 * Bounded Search Tree exact solver for directed FVS.
 * Uses SCC decomposition + directed kernelization + directed BST branching.
 */
std::vector<int> solve_directed_BST(int n,
                                    const std::vector<std::pair<int, int>> &edges);

/**
 * Iterative Compression exact solver for directed FVS.
 * Compression uses k-subset enumeration over the compressed FVS.
 */
std::vector<int> solve_directed_IC(int n,
                                   const std::vector<std::pair<int, int>> &edges);

/**
 * Memetic Algorithm for directed FVS.
 * Population of binary vectors; fitness = FVS size (feasible) or + penalty.
 */
std::vector<int> solve_directed_MA(int n,
                                   const std::vector<std::pair<int, int>> &edges,
                                   int pop_size = 50, int max_gens = 200);

/**
 * Kernelized Memetic Algorithm (KMA) for directed FVS.
 *
 * Pipeline:
 *   1. Apply directed kernelization rules to collect forced DFVS vertices.
 *   2. Solve the reduced kernel graph with MA.
 *   3. Map kernel solution back and merge with forced vertices.
 */
std::vector<int> solve_directed_KME(int n,
                                    const std::vector<std::pair<int, int>> &edges,
                                    int pop_size = 50, int max_gens = 200);

// Preferred KMA entry point (KME kept as backward-compatible alias).
std::vector<int> solve_directed_KMA(int n,
                                    const std::vector<std::pair<int, int>> &edges,
                                    int pop_size = 50, int max_gens = 200);