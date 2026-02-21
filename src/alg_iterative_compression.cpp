#include "alg_iterative_compression.h"
#include "utils.h"
#include <algorithm>
#include <queue>
#include <set>

/**
 * @brief Check if removing a subset makes the graph acyclic using DFS
 */
bool is_valid_fvs(const Graph &G, const std::vector<int> &subset) {
    std::vector<char> removed(G.n, 0);
    for (int v : subset) {
        if (v >= 0 && v < G.n) removed[v] = 1;
    }
    
    int remaining_nodes = 0;
    return is_acyclic_after_removal(G, removed, remaining_nodes);
}

/**
 * @brief Generate all 2-partitions of a set using bit manipulation
 * 
 * For a set of size n, there are 2^n possible subsets
 * Each partition (F1, F2) where F1 ∪ F2 = elements and F1 ∩ F2 = ∅
 */
void generate_partitions(const std::vector<int> &elements,
                        std::vector<std::pair<std::vector<int>, std::vector<int>>> &partitions) {
    partitions.clear();
    int n = elements.size();
    
    // Generate all 2^n subsets for F1
    for (int mask = 0; mask < (1 << n); ++mask) {
        std::vector<int> F1, F2;
        
        for (int i = 0; i < n; ++i) {
            if (mask & (1 << i)) {
                F1.push_back(elements[i]);
            } else {
                F2.push_back(elements[i]);
            }
        }
        
        partitions.push_back({F1, F2});
    }
}

/**
 * @brief Try to find a valid FVS of size ≤ k that extends F1 without using F2
 * 
 * @param G Input graph
 * @param F1 Vertices we must keep in FVS
 * @param F2 Vertices we cannot use in FVS
 * @param k Maximum total size
 * @param result_out Output FVS
 * @return true if such FVS found
 * 
 * Strategy: Greedily add high-degree vertices from V \ (F1 ∪ F2) until acyclic
 */
bool find_extension(const Graph &G, const std::vector<int> &F1, 
                    const std::vector<int> &F2, int k, std::vector<int> &result_out) {
    std::set<int> forbidden(F2.begin(), F2.end());
    std::set<int> mandatory(F1.begin(), F1.end());
    
    // Start with mandatory vertices
    std::vector<int> current_fvs = F1;
    
    if ((int)current_fvs.size() > k) return false;
    
    // Check if F1 already forms valid FVS
    if (is_valid_fvs(G, current_fvs)) {
        result_out = current_fvs;
        return true;
    }
    
    // Build list of candidates (V \ (F1 ∪ F2))
    std::vector<std::pair<int, int>> candidates; // (degree, vertex)
    for (int v = 0; v < G.n; ++v) {
        if (forbidden.count(v) || mandatory.count(v)) continue;
        // Compute effective degree (edges to non-FVS vertices)
        candidates.push_back({(int)G.adj[v].size(), v});
    }
    
    // Sort by degree descending (greedy heuristic)
    std::sort(candidates.begin(), candidates.end(), std::greater<std::pair<int, int>>());
    
    // Greedily add vertices until acyclic or budget exhausted
    for (auto [deg, v] : candidates) {
        if ((int)current_fvs.size() >= k) break;
        
        current_fvs.push_back(v);
        if (is_valid_fvs(G, current_fvs)) {
            result_out = current_fvs;
            return true;
        }
    }
    
    // Last check with current set
    if ((int)current_fvs.size() <= k && is_valid_fvs(G, current_fvs)) {
        result_out = current_fvs;
        return true;
    }
    
    return false;
}

/**
 * @brief Compress FVS from size k+1 to k using partition enumeration
 * 
 * Core of the iterative compression technique:
 * - Given FVS F of size k+1
 * - Enumerate all 2-partitions (F1, F2) of F
 * - For each partition, try to find Y ⊆ V\F2 such that F1 ∪ Y is valid FVS
 * - If |F1 ∪ Y| ≤ k, we successfully compressed
 */
bool compress_fvs(const Graph &G, const std::vector<int> &current_fvs,
                  int k, std::vector<int> &compressed_out) {
    if ((int)current_fvs.size() <= k) {
        compressed_out = current_fvs;
        return true;
    }
    
    // Generate all 2-partitions of current_fvs
    std::vector<std::pair<std::vector<int>, std::vector<int>>> partitions;
    generate_partitions(current_fvs, partitions);
    
    // Try each partition
    for (auto &[F1, F2] : partitions) {
        std::vector<int> candidate;
        
        // Try to extend F1 to a valid FVS without using F2
        if (find_extension(G, F1, F2, k, candidate)) {
            if ((int)candidate.size() <= k) {
                compressed_out = candidate;
                return true;
            }
        }
    }
    
    return false;
}

/**
 * @brief Main Iterative Compression algorithm
 * 
 * Algorithm steps:
 * 1. Order vertices v_1, ..., v_n (arbitrary order is fine)
 * 2. Initialize F = {v_1}
 * 3. For i = 2 to n:
 *    a. F = F ∪ {v_i}
 *    b. If |F| = k+1:
 *       - Try to compress F to size k
 *       - If compression fails, return NO (no FVS of size ≤ k exists)
 * 4. Return F if |F| ≤ k
 */
bool iterative_compression_fvs(const Graph &G, int k, std::vector<int> &fvs_out) {
    fvs_out.clear();
    
    if (G.n == 0) return true;
    if (k < 0) return false;
    
    // Check if graph is already acyclic
    if (is_valid_fvs(G, {})) {
        return true;
    }
    
    // Initialize with first vertex
    std::vector<int> current_fvs = {0};
    
    // Incrementally add vertices
    for (int i = 1; i < G.n; ++i) {
        current_fvs.push_back(i);
        
        // When size exceeds k, try to compress
        if ((int)current_fvs.size() > k) {
            std::vector<int> compressed;
            
            if (!compress_fvs(G, current_fvs, k, compressed)) {
                // Compression failed - no FVS of size ≤ k exists
                return false;
            }
            
            current_fvs = compressed;
        }
    }
    
    // Final validation
    if ((int)current_fvs.size() <= k && is_valid_fvs(G, current_fvs)) {
        fvs_out = current_fvs;
        return true;
    }
    
    return false;
}
