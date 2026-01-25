#include "graph.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <set>
#include <stdexcept>

Graph Graph::from_edge_list_file(const std::string &path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open file: " + path);
    int u,v;
    std::vector<std::pair<int,int>> edges;
    int maxv = -1;
    std::string line;
    while (std::getline(in, line)){
        if (line.empty() || line[0]=='#') continue;
        std::istringstream ss(line);
        if (!(ss >> u >> v)) continue;
        edges.emplace_back(u,v);
        maxv = std::max(maxv, std::max(u,v));
    }
    Graph G(maxv+1);
    for (auto &e: edges) G.add_edge(e.first, e.second);
    return G;
}

std::pair<int,int> Graph::edge_count() const {
    long long m = 0;
    for (int i = 0; i < n; ++i) m += adj[i].size();
    // each edge counted twice for undirected except self-loops
    m = m/2;
    return {n, (int)m};
}
