#include "alg_kernelization.h"
#include "utils.h"
#include <algorithm>
#include <queue>
#include <map>

/**
 * @brief Remove isolated vertices and leaves (degree 0 and 1)
 * 
 * These vertices can never be part of any cycle, so safe to remove
 */
int apply_degree_01_rule(Graph &G, std::set<int> &removed) {
    int count = 0;
    bool changed = true;
    
    while (changed) {
        changed = false;
        for (int v = 0; v < G.n; ++v) {
            if (removed.count(v)) continue;
            
            // Count non-removed neighbors
            int active_degree = 0;
            for (int u : G.adj[v]) {
                if (!removed.count(u) && u != v) active_degree++;
            }
            
            if (active_degree <= 1) {
                removed.insert(v);
                count++;
                changed = true;
            }
        }
    }
    
    return count;
}

/**
 * @brief Identify self-loops (must be in FVS)
 */
int apply_self_loop_rule(const Graph &G, std::set<int> &forced) {
    int count = 0;
    
    for (int v = 0; v < G.n; ++v) {
        if (forced.count(v)) continue;
        
        // Check if v has self-loop
        for (int u : G.adj[v]) {
            if (u == v) {
                forced.insert(v);
                count++;
                break;
            }
        }
    }
    
    return count;
}

/**
 * @brief Contract degree-2 vertices
 * 
 * For vertex v with exactly 2 neighbors u and w:
 * - If u = w, this creates a triangle, include one vertex in FVS
 * - Otherwise, bypass v: remove v and add edge (u,w)
 */
int apply_degree_2_rule(Graph &G, std::set<int> &removed, std::set<int> &forced) {
    int count = 0;
    bool changed = true;
    
    while (changed) {
        changed = false;
        
        for (int v = 0; v < G.n; ++v) {
            if (removed.count(v) || forced.count(v)) continue;
            
            // Get active neighbors
            std::vector<int> neighbors;
            for (int u : G.adj[v]) {
                if (!removed.count(u) && !forced.count(u) && u != v) {
                    neighbors.push_back(u);
                }
            }
            
            // Remove duplicates
            std::sort(neighbors.begin(), neighbors.end());
            neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
            
            if (neighbors.size() == 2) {
                int u = neighbors[0];
                int w = neighbors[1];
                
                // Check if u and w are already connected
                bool already_connected = false;
                for (int neighbor : G.adj[u]) {
                    if (neighbor == w) {
                        already_connected = true;
                        break;
                    }
                }
                
                // Bypass v: remove v and connect u-w
                removed.insert(v);
                
                // Add edge u-w if not exists
                if (!already_connected) {
                    G.add_edge(u, w);
                }
                
                count++;
                changed = true;
                break; // Restart to handle cascading effects
            }
        }
    }
    
    return count;
}

/**
 * @brief Remove duplicate edges
 */
int remove_duplicate_edges(Graph &G) {
    int count = 0;
    
    for (int v = 0; v < G.n; ++v) {
        std::vector<int> &adj = G.adj[v];
        if (adj.empty()) continue;
        
        std::sort(adj.begin(), adj.end());
        auto last = std::unique(adj.begin(), adj.end());
        count += adj.end() - last;
        adj.erase(last, adj.end());
    }
    
    return count;
}

/**
 * @brief Apply all reduction rules exhaustively
 */
bool apply_all_rules(Graph &G, std::set<int> &forced, std::set<int> &removed) {
    int total_changes = 0;
    
    // Apply self-loop rule first (vertices must be in FVS)
    total_changes += apply_self_loop_rule(G, forced);
    
    // Remove vertices forced into FVS from graph
    for (int v : forced) {
        if (!removed.count(v)) {
            removed.insert(v);
        }
    }
    
    // Apply degree 0/1 rule
    total_changes += apply_degree_01_rule(G, removed);
    
    // Apply degree 2 rule (contraction)
    total_changes += apply_degree_2_rule(G, removed, forced);
    
    // Clean up duplicate edges
    total_changes += remove_duplicate_edges(G);
    
    return total_changes > 0;
}

/**
 * @brief Build reduced graph with only active vertices
 */
Graph build_reduced_graph(const Graph &G, const std::set<int> &active,
                         std::vector<int> &vertex_map) {
    vertex_map.clear();
    std::map<int, int> old_to_new;
    
    // Build vertex mapping
    int new_id = 0;
    for (int v : active) {
        old_to_new[v] = new_id;
        vertex_map.push_back(v);
        new_id++;
    }
    
    // Create reduced graph
    Graph reduced(new_id);
    
    for (int old_v : active) {
        int new_v = old_to_new[old_v];
        
        for (int old_u : G.adj[old_v]) {
            if (active.count(old_u)) {
                int new_u = old_to_new[old_u];
                // Only add if not duplicate (handle in final graph construction)
                bool found = false;
                for (int neighbor : reduced.adj[new_v]) {
                    if (neighbor == new_u) {
                        found = true;
                        break;
                    }
                }
                if (!found && new_v < new_u) {
                    // Add edge only once for undirected graph
                    reduced.add_edge(new_v, new_u);
                }
            }
        }
    }
    
    return reduced;
}

/**
 * @brief Main kernelization algorithm
 */
KernelResult kernelize_graph(const Graph &G, int k) {
    KernelResult result;
    result.original_n = G.n;
    
    // Make a working copy
    Graph work_graph = G;
    
    std::set<int> forced;  // Vertices forced into FVS
    std::set<int> removed; // Vertices removed from consideration
    
    // Apply rules exhaustively
    int iterations = 0;
    const int max_iterations = 100; // Prevent infinite loops
    
    while (iterations < max_iterations) {
        bool changed = apply_all_rules(work_graph, forced, removed);
        if (!changed) break;
        iterations++;
    }
    
    // Build sets of vertices
    std::set<int> active;
    for (int v = 0; v < G.n; ++v) {
        if (!removed.count(v) && !forced.count(v)) {
            active.insert(v);
        }
    }
    
    // Store forced and safely removed vertices
    result.forced_in_fvs = std::vector<int>(forced.begin(), forced.end());
    result.removed_safe = std::vector<int>(removed.begin(), removed.end());
    
    // Separate actually removed from forced
    std::set<int> actually_removed;
    for (int v : removed) {
        if (!forced.count(v)) {
            actually_removed.insert(v);
        }
    }
    result.removed_safe = std::vector<int>(actually_removed.begin(), actually_removed.end());
    
    // Build reduced graph
    result.reduced_graph = build_reduced_graph(G, active, result.original_mapping);
    
    // Update k parameter
    result.k_reduced = k - (int)result.forced_in_fvs.size();
    
    return result;
}

/**
 * @brief Reconstruct original FVS from kernel solution
 */
std::vector<int> reconstruct_fvs(const std::vector<int> &kernel_fvs,
                                 const KernelResult &result) {
    std::vector<int> original_fvs;
    
    // Add forced vertices
    for (int v : result.forced_in_fvs) {
        original_fvs.push_back(v);
    }
    
    // Map kernel FVS back to original vertex IDs
    for (int v : kernel_fvs) {
        if (v >= 0 && v < (int)result.original_mapping.size()) {
            original_fvs.push_back(result.original_mapping[v]);
        }
    }
    
    // Remove duplicates and sort
    std::sort(original_fvs.begin(), original_fvs.end());
    original_fvs.erase(std::unique(original_fvs.begin(), original_fvs.end()), 
                      original_fvs.end());
    
    return original_fvs;
}
