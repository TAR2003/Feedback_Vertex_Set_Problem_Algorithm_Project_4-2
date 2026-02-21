#pragma once
#include "graph.h"
#include <string>

/**
 * @file graph_generators.h
 * @brief Graph generation utilities for FVS experiments
 * 
 * Generates various types of graphs for comprehensive testing:
 * - Random (Erdős-Rényi): Baseline performance
 * - Scale-Free (Barabási-Albert): Real-world networks
 * - Small-World (Watts-Strogatz): Social networks
 * - Grid: Structured topology
 * - Cycle-Heavy: Worst-case scenarios
 * - Trees: Sanity checks (FVS should be 0)
 * 
 * Reference: Newman (2010), Barabási & Albert (1999)
 */

/**
 * @brief Generate Erdős-Rényi random graph G(n, p)
 * 
 * @param n Number of vertices
 * @param p Edge probability (0 < p < 1)
 * @param seed Random seed
 * @return Random graph
 * 
 * @details
 * Each edge appears independently with probability p
 * Expected edges: p * n * (n-1) / 2
 * Expected degree: p * (n-1)
 */
Graph generate_erdos_renyi(int n, double p, unsigned seed = 42);

/**
 * @brief Generate Barabási-Albert scale-free graph
 * 
 * @param n Final number of vertices
 * @param m Number of edges to attach from new vertex (m ≥ 1)
 * @param seed Random seed
 * @return Scale-free graph
 * 
 * @details
 * Uses preferential attachment: new vertices connect to existing
 * vertices with probability proportional to their degree.
 * Produces power-law degree distribution (common in real networks)
 */
Graph generate_barabasi_albert(int n, int m, unsigned seed = 42);

/**
 * @brief Generate Watts-Strogatz small-world graph
 * 
 * @param n Number of vertices
 * @param k Each vertex connected to k nearest neighbors (even k)
 * @param beta Rewiring probability (0 ≤ β ≤ 1)
 * @param seed Random seed
 * @return Small-world graph
 * 
 * @details
 * Start with ring lattice, rewire each edge with probability β
 * β=0: regular lattice, β=1: random graph
 * Small β gives high clustering + short path length (small-world)
 */
Graph generate_watts_strogatz(int n, int k, double beta, unsigned seed = 42);

/**
 * @brief Generate 2D grid graph
 * 
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Grid graph
 * 
 * @details
 * Each vertex connected to 4-neighbors (up, down, left, right)
 * Total vertices: rows * cols
 * Total edges: 2 * rows * cols - rows - cols
 */
Graph generate_grid(int rows, int cols);

/**
 * @brief Generate random tree (acyclic graph)
 * 
 * @param n Number of vertices
 * @param seed Random seed
 * @return Random tree (FVS should be 0)
 * 
 * @details
 * Generated using random edges that don't create cycles
 * Sanity check: any valid FVS algorithm should return empty set
 */
Graph generate_random_tree(int n, unsigned seed = 42);

/**
 * @brief Generate cycle-heavy graph (stress test)
 * 
 * @param n Number of vertices
 * @param cycle_density Controls number of cycles (0.0 to 1.0)
 * @param seed Random seed
 * @return Graph with many cycles
 * 
 * @details
 * Creates multiple overlapping cycles to maximize FVS size
 * Used for worst-case performance testing
 */
Graph generate_cycle_heavy(int n, double cycle_density, unsigned seed = 42);

/**
 * @brief Generate complete graph K_n
 * 
 * @param n Number of vertices
 * @return Complete graph
 * 
 * @details
 * Every vertex connected to every other vertex
 * FVS size = n - 1 (remove all but 1 vertex)
 */
Graph generate_complete(int n);

/**
 * @brief Generate complete bipartite graph K_{m,n}
 * 
 * @param m Vertices in first partition
 * @param n Vertices in second partition
 * @return Complete bipartite graph
 * 
 * @details
 * No cycles if m=1 or n=1
 * Cycles exist for m,n ≥ 2
 */
Graph generate_complete_bipartite(int m, int n);

/**
 * @brief Save graph to edge list file
 * 
 * @param G Graph to save
 * @param filename Output filename
 * @param comment Optional comment line
 */
void save_graph_to_file(const Graph &G, const std::string &filename,
                       const std::string &comment = "");

/**
 * @brief Generate benchmark suite for experiments
 * 
 * Creates comprehensive test set with various graph types and sizes
 * 
 * @param output_dir Directory to save generated graphs
 * @param seed Base random seed
 */
void generate_benchmark_suite(const std::string &output_dir, unsigned seed = 42);
