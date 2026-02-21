#include "utils.h"
#include <chrono>
#include <functional>
#include <fstream>
#include <stdexcept>
#include <queue>
#include <stack>

MeasureResult measure_function_runtime(std::function<void()> f) {
    long rss_kb = 0;
    
#ifdef _WIN32
    // Windows: Use GetProcessMemoryInfo
    PROCESS_MEMORY_COUNTERS_EX pmc;
    auto t1 = std::chrono::high_resolution_clock::now();
    f();
    auto t2 = std::chrono::high_resolution_clock::now();
    
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        rss_kb = pmc.WorkingSetSize / 1024; // Convert bytes to KB
    }
#else
    // Unix/Linux: Use getrusage
    struct rusage usage_before, usage_after;
    getrusage(RUSAGE_SELF, &usage_before);
    auto t1 = std::chrono::high_resolution_clock::now();
    f();
    auto t2 = std::chrono::high_resolution_clock::now();
    getrusage(RUSAGE_SELF, &usage_after);
    rss_kb = usage_after.ru_maxrss; // typically in kilobytes on Linux
#endif
    
    double elapsed = std::chrono::duration<double, std::milli>(t2 - t1).count();
    return {elapsed, rss_kb};
}

bool is_acyclic_after_removal(const Graph &G, const std::vector<char> &removed, int &remaining_nodes) {
    int n = G.n;
    std::vector<int> vis(n,0), parent(n,-1);
    remaining_nodes = 0;
    for (int i=0;i<n;++i) if (!removed[i]) remaining_nodes++;
    std::function<bool(int)> dfs = [&](int s)->bool{
        std::stack<int> st;
        st.push(s);
        parent[s] = -1;
        while(!st.empty()){
            int u = st.top(); st.pop();
            if (vis[u]==0){
                vis[u]=1;
                for(int v: G.adj[u]){
                    if (removed[v]) continue;
                    if (vis[v]==0){ parent[v]=u; st.push(v); }
                    else if (v!=parent[u]) return false; // found cycle
                }
            }
        }
        return true;
    };
    for (int i=0;i<n;++i){
        if (removed[i] || vis[i]) continue;
        if (!dfs(i)) return false;
    }
    return true;
}

std::vector<int> read_subset_from_file(const std::string &path){
    std::vector<int> res;
    std::ifstream in(path);
    int x;
    while(in>>x) res.push_back(x);
    return res;
}
