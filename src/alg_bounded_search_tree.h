#pragma once
#include "graph.h"
#include "alg_kernelization.h"
#include <vector>

/**
 * @file alg_bounded_search_tree.h
 * @brief Bounded Search Tree with Kernelization for FVS
 * 
 * Combines polynomial-time kernelization with smart branching search
 * Time Complexity: O(4^k * n^2) after kernelization
 * 
 * Algorithm Strategy:
 * 1. Apply kernelization to reduce graph size
 * 2. If reduced parameter k' < 0, return NO
 * 3. If kernel is acyclic, return forced vertices as FVS
 * 4. Otherwise, find a cycle and branch on its vertices
 * 5. For each vertex v in cycle, recursively solve (G-v, k'-1)
 * 6. Use smart vertex selection (highest degree, centrality)
 * 
 * Key Innovation:
 * - Small kernel (after reduction) leads to shallow search tree
 * - Most real-world graphs reduce by 50-90%
 * - Branching only on cycle vertices (bounded by k)
 * 
 * Reference: Chen et al. (2006), Cygan et al. (2015)
 */

/**
 * @brief Bounded Search Tree algorithm with kernelization
 * 
 * @param G Input graph
 * @param k Maximum FVS size
 * @param fvs_out Output FVS (if found)
 * @return true if FVS of size ≤ k exists
 * 
 * @details
 * First applies kernelization, then uses bounded depth search tree.
 * Much faster than pure branching due to preprocessing.
 */
bool bounded_search_tree_fvs(const Graph &G, int k, std::vector<int> &fvs_out);

/**
 * @brief Internal recursive search function
 * 
 * @param G Current graph
 * @param k Remaining budget
 * @param current_fvs Vertices already in FVS
 * @param fvs_out Final FVS
 * @return true if solution found
 */
bool bst_search_recursive(const Graph &G, int k, 
                         std::vector<int> current_fvs,
                         std::vector<int> &fvs_out);

/**
 * @brief Find a cycle in the graph using DFS
 * 
 * @param G Graph to search
 * @param cycle Output cycle vertices
 * @return true if cycle found
 */
bool find_cycle(const Graph &G, std::vector<int> &cycle);

/**
 * @brief Select best vertex from cycle to branch on
 * 
 * Uses heuristics: highest degree, most cycles involved
 * 
 * @param G Graph
 * @param cycle Cycle vertices
 * @return Best vertex to branch on
 */
int select_branch_vertex(const Graph &G, const std::vector<int> &cycle);

/**
 * @brief Remove a vertex from graph (create subgraph)
 * 
 * @param G Original graph
 * @param removed_vertex Vertex to remove
 * @param vertex_map Maps new IDs to original IDs
 * @return New graph with vertex removed
 */
Graph remove_vertex(const Graph &G, int removed_vertex, 
                   std::vector<int> &vertex_map);
