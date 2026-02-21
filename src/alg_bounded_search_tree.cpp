#include "alg_bounded_search_tree.h"
#include "alg_kernelization.h"
#include "utils.h"
#include <algorithm>
#include <set>
#include <map>

/**
 * @brief Find a cycle using DFS
 * 
 * Returns first cycle found (any cycle will do for branching)
 */
bool find_cycle(const Graph &G, std::vector<int> &cycle) {
    cycle.clear();
    
    std::vector<int> color(G.n, 0); // 0=white, 1=gray, 2=black
    std::vector<int> parent(G.n, -1);
    
    // DFS to find cycle
    std::function<bool(int)> dfs = [&](int u) -> bool {
        color[u] = 1; // Mark as visiting
        
        for (int v : G.adj[u]) {
            if (v == u) continue; // Skip self-loops (handled in kernelization)
            
            if (color[v] == 1 && v != parent[u]) {
                // Found back edge - reconstruct cycle
                cycle.push_back(v);
                int curr = u;
                while (curr != v && curr != -1) {
                    cycle.push_back(curr);
                    curr = parent[curr];
                }
                return true;
            }
            
            if (color[v] == 0) {
                parent[v] = u;
                if (dfs(v)) return true;
            }
        }
        
        color[u] = 2; // Mark as done
        return false;
    };
    
    // Try DFS from each unvisited vertex
    for (int i = 0; i < G.n; ++i) {
        if (color[i] == 0) {
            if (dfs(i)) {
                return true;
            }
        }
    }
    
    return false;
}

/**
 * @brief Select best vertex from cycle to branch on
 * 
 * Heuristic: Choose vertex with highest degree
 * This maximizes the number of potential cycles broken
 */
int select_branch_vertex(const Graph &G, const std::vector<int> &cycle) {
    if (cycle.empty()) return -1;
    
    int best_vertex = cycle[0];
    int max_degree = G.adj[cycle[0]].size();
    
    for (int v : cycle) {
        int degree = G.adj[v].size();
        if (degree > max_degree) {
            max_degree = degree;
            best_vertex = v;
        }
    }
    
    return best_vertex;
}

/**
 * @brief Remove vertex from graph
 * 
 * Creates new graph with specified vertex removed
 */
Graph remove_vertex(const Graph &G, int removed_vertex,
                   std::vector<int> &vertex_map) {
    vertex_map.clear();
    std::map<int, int> old_to_new;
    
    // Build mapping (skip removed vertex)
    int new_id = 0;
    for (int v = 0; v < G.n; ++v) {
        if (v != removed_vertex) {
            old_to_new[v] = new_id;
            vertex_map.push_back(v);
            new_id++;
        }
    }
    
    // Build new graph
    Graph new_graph(new_id);
    
    for (int old_v = 0; old_v < G.n; ++old_v) {
        if (old_v == removed_vertex) continue;
        
        int new_v = old_to_new[old_v];
        
        for (int old_u : G.adj[old_v]) {
            if (old_u == removed_vertex) continue;
            
            int new_u = old_to_new[old_u];
            
            // Check if edge already added
            bool found = false;
            for (int neighbor : new_graph.adj[new_v]) {
                if (neighbor == new_u) {
                    found = true;
                    break;
                }
            }
            
            if (!found && new_v < new_u) {
                new_graph.add_edge(new_v, new_u);
            }
        }
    }
    
    return new_graph;
}

/**
 * @brief Recursive bounded search tree
 * 
 * Base cases:
 * 1. k < 0: Budget exhausted, return false
 * 2. Graph is acyclic: Found solution, return true
 * 
 * Recursive case:
 * 1. Find a cycle
 * 2. Branch on highest-degree vertex in cycle
 * 3. Recursively solve (G-v, k-1)
 */
bool bst_search_recursive(const Graph &G, int k,
                         std::vector<int> current_fvs,
                         std::vector<int> &fvs_out) {
    // Base case 1: Budget exhausted
    if (k < 0) return false;
    
    // Base case 2: Check if graph is acyclic
    std::vector<char> removed(G.n, 0);
    int remaining = 0;
    if (is_acyclic_after_removal(G, removed, remaining)) {
        fvs_out = current_fvs;
        return true;
    }
    
    // Base case 3: Budget zero but still has cycles
    if (k == 0) return false;
    
    // Find a cycle to branch on
    std::vector<int> cycle;
    if (!find_cycle(G, cycle)) {
        // No cycle found (shouldn't happen if not acyclic)
        fvs_out = current_fvs;
        return true;
    }
    
    // Select best vertex from cycle
    int branch_v = select_branch_vertex(G, cycle);
    
    // Branch: try including branch_v in FVS
    std::vector<int> vertex_map;
    Graph reduced = remove_vertex(G, branch_v, vertex_map);
    
    std::vector<int> new_fvs = current_fvs;
    new_fvs.push_back(branch_v);
    
    // Recursive call
    if (bst_search_recursive(reduced, k - 1, new_fvs, fvs_out)) {
        return true;
    }
    
    // If single branch failed, try other vertices in cycle
    // (Optional: could branch on multiple vertices)
    
    return false;
}

/**
 * @brief Main Bounded Search Tree algorithm with kernelization
 * 
 * Steps:
 * 1. Apply kernelization to reduce graph
 * 2. Check if reduced parameter is negative (no solution)
 * 3. Apply bounded search tree on kernel
 * 4. Reconstruct solution in original graph
 */
bool bounded_search_tree_fvs(const Graph &G, int k, std::vector<int> &fvs_out) {
    fvs_out.clear();
    
    if (k < 0) return false;
    
    // Step 1: Kernelize
    KernelResult kernel = kernelize_graph(G, k);
    
    // Step 2: Check reduced parameter
    if (kernel.k_reduced < 0) {
        // Forced vertices exceed budget
        return false;
    }
    
    // Step 3: Check if kernel is already acyclic
    std::vector<char> removed(kernel.reduced_graph.n, 0);
    int remaining = 0;
    if (is_acyclic_after_removal(kernel.reduced_graph, removed, remaining)) {
        // Kernel is acyclic, return forced vertices
        fvs_out = kernel.forced_in_fvs;
        return true;
    }
    
    // Step 4: Apply bounded search tree on kernel
    std::vector<int> kernel_fvs;
    std::vector<int> initial_fvs;
    
    if (!bst_search_recursive(kernel.reduced_graph, kernel.k_reduced, 
                             initial_fvs, kernel_fvs)) {
        return false;
    }
    
    // Step 5: Reconstruct solution in original graph
    fvs_out = reconstruct_fvs(kernel_fvs, kernel);
    
    return true;
}
