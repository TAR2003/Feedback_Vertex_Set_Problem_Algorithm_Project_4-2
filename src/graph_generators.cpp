#include "graph_generators.h"
#include <random>
#include <algorithm>
#include <set>
#include <fstream>
#include <queue>
#include <cmath>

/**
 * @brief Generate Erdős-Rényi random graph
 */
Graph generate_erdos_renyi(int n, double p, unsigned seed) {
    Graph G(n);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    
    for (int u = 0; u < n; ++u) {
        for (int v = u + 1; v < n; ++v) {
            if (uni(rng) < p) {
                G.add_edge(u, v);
            }
        }
    }
    
    return G;
}

/**
 * @brief Generate Barabási-Albert scale-free graph
 */
Graph generate_barabasi_albert(int n, int m, unsigned seed) {
    if (m < 1) m = 1;
    if (m >= n) m = n - 1;
    
    Graph G(n);
    std::mt19937 rng(seed);
    
    // Start with small complete graph
    int initial = std::max(m, 2);
    for (int u = 0; u < initial; ++u) {
        for (int v = u + 1; v < initial; ++v) {
            G.add_edge(u, v);
        }
    }
    
    // Track degree for preferential attachment
    std::vector<int> degrees(n, 0);
    for (int u = 0; u < initial; ++u) {
        degrees[u] = initial - 1;
    }
    
    // Add remaining vertices with preferential attachment
    for (int new_v = initial; new_v < n; ++new_v) {
        // Compute cumulative probabilities
        std::vector<double> cumulative(new_v);
        int total_degree = 0;
        for (int u = 0; u < new_v; ++u) {
            total_degree += degrees[u];
        }
        
        if (total_degree == 0) total_degree = 1; // Prevent division by zero
        
        cumulative[0] = (double)degrees[0] / total_degree;
        for (int u = 1; u < new_v; ++u) {
            cumulative[u] = cumulative[u-1] + (double)degrees[u] / total_degree;
        }
        
        // Select m vertices to connect to
        std::set<int> targets;
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        
        while ((int)targets.size() < m && (int)targets.size() < new_v) {
            double r = uni(rng);
            int target = std::lower_bound(cumulative.begin(), cumulative.begin() + new_v, r) 
                        - cumulative.begin();
            target = std::min(target, new_v - 1);
            targets.insert(target);
        }
        
        // Add edges
        for (int target : targets) {
            G.add_edge(new_v, target);
            degrees[new_v]++;
            degrees[target]++;
        }
    }
    
    return G;
}

/**
 * @brief Generate Watts-Strogatz small-world graph
 */
Graph generate_watts_strogatz(int n, int k, double beta, unsigned seed) {
    if (k >= n) k = n - 1;
    if (k % 2 == 1) k--; // Make k even
    if (k < 2) k = 2;
    
    Graph G(n);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::uniform_int_distribution<int> vertex_dist(0, n - 1);
    
    // Create ring lattice
    for (int u = 0; u < n; ++u) {
        for (int j = 1; j <= k / 2; ++j) {
            int v = (u + j) % n;
            G.add_edge(u, v);
        }
    }
    
    // Rewire edges with probability beta
    std::set<std::pair<int, int>> edges_set;
    for (int u = 0; u < n; ++u) {
        for (int v : G.adj[u]) {
            if (u < v) edges_set.insert({u, v});
        }
    }
    
    std::vector<std::pair<int, int>> edges(edges_set.begin(), edges_set.end());
    
    for (auto [u, v] : edges) {
        if (uni(rng) < beta) {
            // Rewire edge (u, v) to (u, w) for random w
            int w = vertex_dist(rng);
            int attempts = 0;
            while ((w == u || w == v || edges_set.count({std::min(u, w), std::max(u, w)})) 
                   && attempts < n) {
                w = vertex_dist(rng);
                attempts++;
            }
            
            if (w != u && w != v) {
                // Note: In undirected graph, rewiring is complex
                // We simply add new edge (may increase edge count slightly)
                G.add_edge(u, w);
            }
        }
    }
    
    return G;
}

/**
 * @brief Generate 2D grid graph
 */
Graph generate_grid(int rows, int cols) {
    int n = rows * cols;
    Graph G(n);
    
    auto index = [=](int r, int c) { return r * cols + c; };
    
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int u = index(r, c);
            
            // Connect to right neighbor
            if (c + 1 < cols) {
                int v = index(r, c + 1);
                G.add_edge(u, v);
            }
            
            // Connect to bottom neighbor
            if (r + 1 < rows) {
                int v = index(r + 1, c);
                G.add_edge(u, v);
            }
        }
    }
    
    return G;
}

/**
 * @brief Generate random tree
 */
Graph generate_random_tree(int n, unsigned seed) {
    Graph G(n);
    if (n <= 1) return G;
    
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, n - 2);
    
    // Generate random tree using Prüfer sequence
    std::vector<int> degree(n, 1);
    std::vector<int> prufer(n - 2);
    
    for (int i = 0; i < n - 2; ++i) {
        prufer[i] = dist(rng) % (n - 1);
        degree[prufer[i]]++;
    }
    
    // Convert Prüfer sequence to tree
    for (int i = 0; i < n - 2; ++i) {
        for (int v = 0; v < n; ++v) {
            if (degree[v] == 1) {
                G.add_edge(v, prufer[i]);
                degree[v]--;
                degree[prufer[i]]--;
                break;
            }
        }
    }
    
    // Connect last two vertices with degree 1
    int u = -1, v = -1;
    for (int i = 0; i < n; ++i) {
        if (degree[i] == 1) {
            if (u == -1) u = i;
            else { v = i; break; }
        }
    }
    if (u != -1 && v != -1) {
        G.add_edge(u, v);
    }
    
    return G;
}

/**
 * @brief Generate cycle-heavy graph
 */
Graph generate_cycle_heavy(int n, double cycle_density, unsigned seed) {
    Graph G(n);
    if (n < 3) return G;
    
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> vertex_dist(0, n - 1);
    std::uniform_int_distribution<int> cycle_length_dist(3, std::min(10, n));
    
    int num_cycles = (int)(cycle_density * n);
    
    for (int c = 0; c < num_cycles; ++c) {
        int length = cycle_length_dist(rng);
        std::set<int> cycle_vertices;
        
        // Select random vertices for cycle
        while ((int)cycle_vertices.size() < length) {
            cycle_vertices.insert(vertex_dist(rng));
        }
        
        std::vector<int> cycle(cycle_vertices.begin(), cycle_vertices.end());
        
        // Create cycle
        for (size_t i = 0; i < cycle.size(); ++i) {
            int u = cycle[i];
            int v = cycle[(i + 1) % cycle.size()];
            G.add_edge(u, v);
        }
    }
    
    return G;
}

/**
 * @brief Generate complete graph
 */
Graph generate_complete(int n) {
    Graph G(n);
    
    for (int u = 0; u < n; ++u) {
        for (int v = u + 1; v < n; ++v) {
            G.add_edge(u, v);
        }
    }
    
    return G;
}

/**
 * @brief Generate complete bipartite graph
 */
Graph generate_complete_bipartite(int m, int n) {
    Graph G(m + n);
    
    for (int u = 0; u < m; ++u) {
        for (int v = 0; v < n; ++v) {
            G.add_edge(u, m + v);
        }
    }
    
    return G;
}

/**
 * @brief Save graph to file
 */
void save_graph_to_file(const Graph &G, const std::string &filename,
                       const std::string &comment) {
    std::ofstream out(filename);
    
    if (!comment.empty()) {
        out << "# " << comment << "\n";
    }
    
    out << "# n=" << G.n << "\n";
    
    // Write edges
    std::set<std::pair<int, int>> written;
    for (int u = 0; u < G.n; ++u) {
        for (int v : G.adj[u]) {
            if (u < v && !written.count({u, v})) {
                out << u << " " << v << "\n";
                written.insert({u, v});
            }
        }
    }
    
    out.close();
}

/**
 * @brief Generate comprehensive benchmark suite
 */
void generate_benchmark_suite(const std::string &output_dir, unsigned seed) {
    std::vector<int> sizes = {10, 20, 50, 100, 200};
    std::vector<double> densities = {0.1, 0.3, 0.5, 0.7, 0.9};
    
    int graph_id = 0;
    
    // Erdős-Rényi graphs
    for (int n : sizes) {
        for (double p : densities) {
            Graph G = generate_erdos_renyi(n, p, seed + graph_id);
            std::string filename = output_dir + "/er_n" + std::to_string(n) 
                                 + "_p" + std::to_string((int)(p * 100)) + ".txt";
            save_graph_to_file(G, filename, "Erdos-Renyi n=" + std::to_string(n) 
                             + " p=" + std::to_string(p));
            graph_id++;
        }
    }
    
    // Barabási-Albert graphs
    std::vector<int> m_values = {2, 3, 5};
    for (int n : sizes) {
        for (int m : m_values) {
            if (m < n) {
                Graph G = generate_barabasi_albert(n, m, seed + graph_id);
                std::string filename = output_dir + "/ba_n" + std::to_string(n) 
                                     + "_m" + std::to_string(m) + ".txt";
                save_graph_to_file(G, filename, "Barabasi-Albert n=" + std::to_string(n) 
                                 + " m=" + std::to_string(m));
                graph_id++;
            }
        }
    }
    
    // Grid graphs
    std::vector<std::pair<int, int>> grids = {{3, 3}, {5, 5}, {10, 10}, {15, 15}};
    for (auto [rows, cols] : grids) {
        Graph G = generate_grid(rows, cols);
        std::string filename = output_dir + "/grid_" + std::to_string(rows) 
                             + "x" + std::to_string(cols) + ".txt";
        save_graph_to_file(G, filename, "Grid " + std::to_string(rows) 
                         + "x" + std::to_string(cols));
        graph_id++;
    }
    
    // Trees (sanity check)
    for (int n : sizes) {
        Graph G = generate_random_tree(n, seed + graph_id);
        std::string filename = output_dir + "/tree_n" + std::to_string(n) + ".txt";
        save_graph_to_file(G, filename, "Random Tree n=" + std::to_string(n));
        graph_id++;
    }
    
    // Cycle-heavy graphs
    for (int n : sizes) {
        if (n >= 10) {
            Graph G = generate_cycle_heavy(n, 0.5, seed + graph_id);
            std::string filename = output_dir + "/cycle_heavy_n" + std::to_string(n) + ".txt";
            save_graph_to_file(G, filename, "Cycle-heavy n=" + std::to_string(n));
            graph_id++;
        }
    }
}
