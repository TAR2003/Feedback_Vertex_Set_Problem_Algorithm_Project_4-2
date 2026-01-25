#include "alg_approx.h"
#include "utils.h"
#include <stack>
#include <algorithm>
#include <set>

static bool find_any_cycle_nodes(const Graph &G, const std::vector<char> &removed, std::vector<int> &cycle) {
    // reuse find_cycle style
    int n = G.n;
    std::vector<int> state(n,0), parent(n,-1);
    std::function<bool(int)> dfs = [&](int u)->bool{
        state[u]=1;
        for (int v: G.adj[u]){
            if (removed[v]) continue;
            if (state[v]==0){ parent[v]=u; if (dfs(v)) return true; }
            else if (state[v]==1 && v!=parent[u]){
                cycle.clear(); int cur=u; cycle.push_back(v);
                while (cur!=v && cur!=-1){ cycle.push_back(cur); cur=parent[cur]; }
                return true;
            }
        }
        state[u]=2; return false;
    };
    for (int i=0;i<n;++i){ if (removed[i]) continue; if (state[i]==0){ if (dfs(i)) return true; } }
    return false;
}

std::vector<int> two_approximation(const Graph &G) {
    std::vector<char> removed(G.n, 0);
    std::vector<int> S;
    std::vector<int> cycle;
    while (find_any_cycle_nodes(G, removed, cycle)){
        // pick an edge (cycle[0], cycle[1]) and add both endpoints
        if (cycle.size()>=2){
            int u = cycle[0], v = cycle[1];
            if (!removed[u]) { removed[u]=1; S.push_back(u); }
            if (!removed[v]) { removed[v]=1; S.push_back(v); }
        } else {
            // self loop? remove that vertex
            int u = cycle[0]; if (!removed[u]){ removed[u]=1; S.push_back(u); }
        }
    }
    return S;
}

std::vector<int> greedy_max_degree(const Graph &G) {
    std::vector<char> removed(G.n, 0);
    std::vector<int> deg(G.n,0);
    for (int i=0;i<G.n;++i) deg[i] = (int)G.adj[i].size();
    std::vector<int> S;
    while (true){
        int remaining_nodes = 0; for (int i=0;i<G.n;++i) if (!removed[i]) remaining_nodes++;
        int rem; if (is_acyclic_after_removal(G, removed, rem)) break;
        // pick highest degree vertex
        int best=-1; int bestd=-1;
        for (int i=0;i<G.n;++i) if (!removed[i] && deg[i]>bestd){ best=i; bestd=deg[i]; }
        if (best==-1) break;
        removed[best]=1; S.push_back(best);
        // reduce degrees
        for (int v: G.adj[best]) if (!removed[v]) deg[v]--;
    }
    return S;
}
