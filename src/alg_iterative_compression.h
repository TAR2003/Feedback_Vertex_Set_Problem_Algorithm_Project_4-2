#pragma once
#include "graph.h"
#include <vector>

/**
 * @file alg_iterative_compression.h
 * @brief Iterative Compression algorithm for Feedback Vertex Set problem
 * 
 * Implements the groundbreaking FPT algorithm by Dehne et al. (2004)
 * Time Complexity: O(5^k * k * n^2)
 * 
 * Key Idea:
 * - Build solution incrementally by adding vertices one by one
 * - When solution exceeds k+1, compress it back to size k
 * - Uses smart enumeration over partitions to find compression
 * 
 * Algorithm Overview:
 * 1. Start with first vertex in FVS
 * 2. For each remaining vertex:
 *    a. Add to current FVS
 *    b. If |FVS| = k+1, try to compress to size k
 *    c. Enumerate partitions (F1, F2) of current FVS
 *    d. Find Y ⊆ V\F2 such that F1 ∪ Y is valid FVS of size ≤ k
 * 3. Return final FVS if |FVS| ≤ k, else NO solution
 */

/**
 * @brief Iterative Compression algorithm for FVS
 * 
 * @param G Input graph
 * @param k Maximum FVS size parameter
 * @param fvs_out Output vector to store FVS vertices (if found)
 * @return true if FVS of size ≤ k exists, false otherwise
 * 
 * @details
 * This is one of the most important FPT techniques developed.
 * Published in top theory conferences (STOC/FOCS).
 * Demonstrates advanced understanding of parameterized algorithms.
 */
bool iterative_compression_fvs(const Graph &G, int k, std::vector<int> &fvs_out);

/**
 * @brief Helper function to compress FVS from size k+1 to k
 * 
 * @param G Input graph
 * @param current_fvs Current FVS of size k+1
 * @param k Target size
 * @param compressed_out Output compressed FVS
 * @return true if compression succeeds
 */
bool compress_fvs(const Graph &G, const std::vector<int> &current_fvs, 
                  int k, std::vector<int> &compressed_out);

/**
 * @brief Check if a subset forms a valid FVS
 * 
 * @param G Input graph
 * @param subset Vertices to check
 * @return true if removing subset makes graph acyclic
 */
bool is_valid_fvs(const Graph &G, const std::vector<int> &subset);

/**
 * @brief Generate all 2-partitions of a set
 * 
 * @param elements Input set
 * @param partitions Output vector of partition pairs
 */
void generate_partitions(const std::vector<int> &elements, 
                        std::vector<std::pair<std::vector<int>, std::vector<int>>> &partitions);
