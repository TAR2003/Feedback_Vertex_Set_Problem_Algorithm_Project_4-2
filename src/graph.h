#pragma once
#include <vector>
#include <tuple>
#include <string>

// Simple undirected graph representation (0-based vertices)
struct Graph {
    int n;
    std::vector<std::vector<int>> adj;

    Graph() : n(0) {}
    Graph(int n_) : n(n_), adj(n_) {}

    void add_edge(int u, int v) {
        if (u < 0 || v < 0) return;
        if (u >= n || v >= n) return;
        adj[u].push_back(v);
        if (u != v) adj[v].push_back(u);
    }

    static Graph from_edge_list_file(const std::string &path);
    std::pair<int,int> edge_count() const;
};
