#include "alg_exact.h"
#include "utils.h"
#include <algorithm>
#include <stack>
#include <set>

// Helper: find a cycle in G avoiding removed vertices; returns vertices in cycle order
static bool find_cycle(const Graph &G, const std::vector<char> &removed, std::vector<int> &cycle) {
    int n = G.n;
    std::vector<int> state(n,0), parent(n,-1);
    std::function<bool(int)> dfs = [&](int u)->bool{
        state[u]=1;
        for (int v: G.adj[u]){
            if (removed[v]) continue;
            if (state[v]==0){ parent[v]=u; if (dfs(v)) return true; }
            else if (state[v]==1 && v!=parent[u]){
                // found cycle: reconstruct u->...->v
                cycle.clear();
                int cur=u; cycle.push_back(v);
                while(cur!=v && cur!=-1){ cycle.push_back(cur); cur=parent[cur]; }
                // cycle now v,u,...,next back to v
                return true;
            }
        }
        state[u]=2; return false;
    };
    for (int i=0;i<n;++i){
        if (removed[i] || state[i]) continue;
        if (dfs(i)) return true;
    }
    return false;
}

static bool branch_search(const Graph &G, std::vector<char> &removed, int k, std::vector<int> &solution) {
    int rem_k = k;
    std::vector<int> cycle;
    if (!find_cycle(G, removed, cycle)){
        // no cycles => removed set is a valid FVS
        solution.clear();
        for (int i=0;i<G.n;++i) if (removed[i]) solution.push_back(i);
        return true;
    }
    if (k<=0) return false;
    // Try removing each vertex in the cycle (branching)
    // Prefer smaller cycles
    for (int v: cycle){
        removed[v]=1;
        if (branch_search(G, removed, k-1, solution)) return true;
        removed[v]=0;
    }
    return false;
}

bool exact_fvs_bounded(const Graph &G, int k, std::vector<int> &fvs_out) {
    std::vector<char> removed(G.n,0);
    return branch_search(G, removed, k, fvs_out);
}

int exact_fvs_min(const Graph &G, int k_limit, std::vector<int> &fvs_out) {
    for (int k = 0; k <= k_limit; ++k){
        if (exact_fvs_bounded(G, k, fvs_out)) return (int)fvs_out.size();
    }
    return -1; // not found within limit
}
