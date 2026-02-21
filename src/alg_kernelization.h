#pragma once
#include "graph.h"
#include <vector>
#include <set>

/**
 * @file alg_kernelization.h
 * @brief Kernelization and reduction rules for Feedback Vertex Set
 * 
 * Implements polynomial-time preprocessing that reduces graph size
 * while preserving FVS structure. Often reduces problem size by 50-90%.
 * 
 * Reduction Rules:
 * 1. Degree 0: Remove isolated vertices (never in FVS)
 * 2. Degree 1: Remove leaves (never in FVS)
 * 3. Degree 2: Bypass and contract degree-2 vertices
 * 4. Self-loops: Must include vertex in FVS
 * 5. Multiple edges: Keep only one edge between each pair
 * 6. Chains: Contract long paths between high-degree vertices
 * 
 * Reference: Comprehensive survey by Fomin & Villanger (2012)
 */

/**
 * @brief Result of kernelization process
 */
struct KernelResult {
    Graph reduced_graph;           // Reduced kernel graph
    std::vector<int> forced_in_fvs; // Vertices that must be in FVS
    std::vector<int> removed_safe;  // Vertices safely removed (not in FVS)
    std::vector<int> original_mapping; // Maps kernel vertices to original IDs
    int original_n;                // Original graph size
    int k_reduced;                 // Reduced parameter k' = k - |forced_in_fvs|
    
    KernelResult() : original_n(0), k_reduced(0) {}
};

/**
 * @brief Apply all kernelization rules to reduce graph
 * 
 * @param G Input graph
 * @param k FVS size parameter
 * @return KernelResult containing reduced graph and mappings
 * 
 * @details
 * Repeatedly applies reduction rules until no more rules applicable.
 * Rules are applied in order of efficiency: degree 0/1, self-loops, 
 * degree 2, then more complex structural rules.
 */
KernelResult kernelize_graph(const Graph &G, int k);

/**
 * @brief Reconstruct original FVS from kernel solution
 * 
 * @param kernel_fvs FVS computed on kernel
 * @param result Kernelization result with mappings
 * @return FVS in original graph
 */
std::vector<int> reconstruct_fvs(const std::vector<int> &kernel_fvs, 
                                 const KernelResult &result);

/**
 * @brief Rule 1: Remove degree 0 and 1 vertices
 * 
 * @param G Graph to reduce
 * @param removed Output set of removed vertices
 * @return Number of vertices removed
 */
int apply_degree_01_rule(Graph &G, std::set<int> &removed);

/**
 * @brief Rule 2: Contract degree-2 vertices
 * 
 * If v has degree 2 with neighbors u, w:
 * - If u = w (self-loop created), include u in FVS
 * - Otherwise, bypass v and connect u-w directly
 * 
 * @param G Graph to reduce
 * @param removed Output set of bypassed vertices
 * @param forced Output set forced into FVS
 * @return Number of vertices contracted
 */
int apply_degree_2_rule(Graph &G, std::set<int> &removed, std::set<int> &forced);

/**
 * @brief Rule 3: Identify and include self-loop vertices in FVS
 * 
 * @param G Graph to check
 * @param forced Output set of vertices forced into FVS
 * @return Number of self-loops found
 */
int apply_self_loop_rule(const Graph &G, std::set<int> &forced);

/**
 * @brief Remove duplicate edges (keep only one edge per pair)
 * 
 * @param G Graph to clean
 * @return Number of duplicate edges removed
 */
int remove_duplicate_edges(Graph &G);

/**
 * @brief Apply all reduction rules exhaustively
 * 
 * @param G Graph to reduce (modified in place)
 * @param forced Vertices forced into FVS
 * @param removed Vertices safely removed
 * @return true if any rule was applied
 */
bool apply_all_rules(Graph &G, std::set<int> &forced, std::set<int> &removed);

/**
 * @brief Build reduced graph from original after applying rules
 * 
 * @param G Original graph
 * @param active Active vertices in reduced graph
 * @param vertex_map Maps new vertex IDs to original IDs
 * @return Reduced graph
 */
Graph build_reduced_graph(const Graph &G, const std::set<int> &active,
                         std::vector<int> &vertex_map);
