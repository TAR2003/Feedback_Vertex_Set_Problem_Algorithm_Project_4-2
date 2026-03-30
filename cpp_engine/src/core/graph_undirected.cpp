/**
 * @file graph_undirected.cpp
 * @brief Implements the UndirectedGraph data structure.
 *
 * Uses adjacency SETS (std::set<int>) so that:
 *   - Edge insertion/removal is O(log n)
 *   - Degree query is O(1) via set.size()
 *   - Iterating over active neighbors is straightforward
 *
 * "Active" neighbors means the set may contain a vertex j even after j is
 * deactivated via deactivate(); it is the caller's responsibility to filter
 * by active[] when that matters.  However, deactivate_full() removes the
 * vertex from ALL neighbor sets, making subsequent traversals cleaner.
 */

#include "undirected_fvs.h"
#include <stack>
#include <functional>
#include <unordered_map>
#include <numeric>
#include <stdexcept>

// ─── Constructor ─────────────────────────────────────────────────────────────

UndirectedGraph::UndirectedGraph(int n)
    : GraphBase(n), adj(n) {}

// ─── Edge operations ─────────────────────────────────────────────────────────

void UndirectedGraph::add_edge(int u, int v)
{
    if (!is_active(u) || !is_active(v))
        return;
    adj[u].insert(v);
    adj[v].insert(u);
}

void UndirectedGraph::remove_edge(int u, int v)
{
    adj[u].erase(v);
    adj[v].erase(u);
}

// ─── Vertex removal / restoration ────────────────────────────────────────────

/**
 * Full removal: mark v inactive AND detach it from every neighbor's adj list.
 * Stores every removed edge in `removed_edges` so restoration is O(deg(v)).
 */
void UndirectedGraph::deactivate_full(int v,
                                      std::vector<std::pair<int, int>> &removed_edges)
{
    active[v] = false;
    // Collect and erase all edges incident to v
    for (int nb : adj[v])
    {
        removed_edges.push_back({v, nb});
        adj[nb].erase(v);
    }
    adj[v].clear();
}

/**
 * Inverse of deactivate_full: put v back and re-insert all its edges.
 */
void UndirectedGraph::reactivate_full(int v,
                                      const std::vector<std::pair<int, int>> &removed_edges)
{
    active[v] = true;
    for (auto &[a, b] : removed_edges)
    {
        // Only restore edges where v is one endpoint
        if (a == v)
        {
            adj[v].insert(b);
            adj[b].insert(v);
        }
        else if (b == v)
        {
            adj[v].insert(a);
            adj[a].insert(v);
        }
    }
}

int UndirectedGraph::degree(int v) const
{
    if (!is_active(v))
        return 0;
    // Count only active neighbors
    int cnt = 0;
    for (int nb : adj[v])
    {
        if (active[nb])
            ++cnt;
    }
    return cnt;
}

// ─── Cycle detection and extraction ──────────────────────────────────────────

/**
 * DFS-based cycle detection.
 * Marks vertices with a parent to detect back-edges.
 * @return true if the active subgraph contains a cycle.
 */
bool UndirectedGraph::has_cycle() const
{
    // Union-Find cycle detection (correct for undirected graphs).
    // A cycle exists iff any edge connects two vertices already in the same
    // connected component. Avoids the parent-tracking bugs of iterative DFS.
    std::vector<int> parent(n), rnk(n, 0);
    for (int i = 0; i < n; ++i)
        parent[i] = i;

    std::function<int(int)> find = [&](int x) -> int
    {
        if (parent[x] != x)
            parent[x] = find(parent[x]);
        return parent[x];
    };

    for (int u = 0; u < n; ++u)
    {
        if (!active[u])
            continue;
        for (int nb : adj[u])
        {
            if (!active[nb] || nb <= u)
                continue; // each undirected edge once
            int ru = find(u), rnb = find(nb);
            if (ru == rnb)
                return true; // same component -> cycle!
            // Union by rank
            if (rnk[ru] < rnk[rnb])
                std::swap(ru, rnb);
            parent[rnb] = ru;
            if (rnk[ru] == rnk[rnb])
                ++rnk[ru];
        }
    }
    return false;
}

/**
 * Find one cycle in the active subgraph.
 * Uses recursive-style DFS (iterative with parent map) to extract the cycle.
 *
 * Algorithm:
 *   1. DFS; when a back-edge (u → ancestor) is found, the cycle is the path
 *      from `ancestor` to `u` in the DFS tree.
 *   2. Extract the path by walking up the parent array.
 *
 * @return list of vertex indices forming the cycle, or {} if acyclic.
 */
std::vector<int> UndirectedGraph::find_cycle() const
{
    std::vector<int> color(n, 0);
    std::vector<int> parent(n, -1);
    std::vector<bool> on_stack(n, false);

    // We need a recursive DFS here to properly track the ancestor path.
    // Use a manual stack that tracks "entry" vs "exit" of each vertex.
    // Stack item: (vertex, parent, iterator_position_as_index)
    // For simplicity, use recursive lambda via std::function.

    std::vector<int> cycle;

    std::function<bool(int, int)> dfs = [&](int u, int par) -> bool
    {
        color[u] = 1;
        on_stack[u] = true;

        for (int nb : adj[u])
        {
            if (!active[nb])
                continue;
            if (nb == par)
                continue;
            if (color[nb] == 1 && on_stack[nb])
            {
                // Found back-edge u → nb; extract cycle nb … u
                cycle.push_back(u);
                int cur = u;
                while (cur != nb)
                {
                    cur = parent[cur];
                    cycle.push_back(cur);
                }
                return true;
            }
            if (color[nb] == 0)
            {
                parent[nb] = u;
                if (dfs(nb, u))
                    return true;
            }
        }

        color[u] = 2;
        on_stack[u] = false;
        return false;
    };

    for (int s = 0; s < n; ++s)
    {
        if (active[s] && color[s] == 0)
        {
            parent[s] = -1;
            if (dfs(s, -1))
                break;
        }
    }
    return cycle;
}

// ─── Deep copy ───────────────────────────────────────────────────────────────

UndirectedGraph UndirectedGraph::copy() const
{
    UndirectedGraph g(n);
    g.active = active;
    g.adj = adj;
    return g;
}