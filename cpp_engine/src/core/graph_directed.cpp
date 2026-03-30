/**
 * @file graph_directed.cpp
 * @brief Implements the DirectedGraph data structure and Tarjan's SCC algorithm.
 *
 * Tarjan's Algorithm (O(n + m)):
 *   Performs a single DFS, maintaining a stack of "potentially cyclic" vertices.
 *   Each vertex gets a discovery time (disc) and a low-link value (low).
 *   low[v] = min discovery time reachable from v's subtree via one back-edge.
 *   When low[v] == disc[v], v is the root of an SCC → pop the stack to get it.
 *
 * Why SCC matters for directed FVS:
 *   Every directed cycle is contained entirely within a single SCC.
 *   Vertices not in any SCC of size > 1 (and no self-loops) are NEVER on a cycle,
 *   so they don't need to be in the FVS — this gives us free reductions.
 */

#include "directed_fvs.h"
#include <stack>
#include <functional>
#include <stdexcept>
#include <queue>
#include <climits>
#include <algorithm>

// ─── Constructor ─────────────────────────────────────────────────────────────

DirectedGraph::DirectedGraph(int n)
    : GraphBase(n), out_adj(n), in_adj(n) {}

// ─── Edge operations ─────────────────────────────────────────────────────────

void DirectedGraph::add_edge(int u, int v)
{
    if (!is_active(u) || !is_active(v))
        return;
    out_adj[u].insert(v);
    in_adj[v].insert(u);
}

void DirectedGraph::remove_edge(int u, int v)
{
    out_adj[u].erase(v);
    in_adj[v].erase(u);
}

// ─── Vertex removal / restoration ────────────────────────────────────────────

/**
 * Remove vertex v and all edges incident to it (both incoming and outgoing).
 * Saves removed edges into `removed` for restoration.
 */
void DirectedGraph::deactivate_full(int v,
                                    std::vector<std::pair<int, int>> &removed)
{
    active[v] = false;

    // Remove outgoing edges v → nb
    for (int nb : out_adj[v])
    {
        removed.push_back({v, nb});
        in_adj[nb].erase(v);
    }
    out_adj[v].clear();

    // Remove incoming edges nb → v
    for (int nb : in_adj[v])
    {
        removed.push_back({nb, v});
        out_adj[nb].erase(v);
    }
    in_adj[v].clear();
}

/**
 * Restore vertex v and all its removed edges.
 */
void DirectedGraph::reactivate_full(int v,
                                    const std::vector<std::pair<int, int>> &removed)
{
    active[v] = true;
    for (auto &[u, w] : removed)
    {
        out_adj[u].insert(w);
        in_adj[w].insert(u);
    }
}

// ─── Degree queries ───────────────────────────────────────────────────────────

int DirectedGraph::in_degree(int v) const
{
    if (!is_active(v))
        return 0;
    int cnt = 0;
    for (int nb : in_adj[v])
        if (active[nb])
            ++cnt;
    return cnt;
}

int DirectedGraph::out_degree(int v) const
{
    if (!is_active(v))
        return 0;
    int cnt = 0;
    for (int nb : out_adj[v])
        if (active[nb])
            ++cnt;
    return cnt;
}

int DirectedGraph::degree(int v) const
{
    return in_degree(v) + out_degree(v);
}

// ─── Cycle detection ─────────────────────────────────────────────────────────

/**
 * DFS with 3-coloring:
 *   WHITE (0) = not visited
 *   GRAY  (1) = on the current DFS recursion stack
 *   BLACK (2) = fully processed
 *
 * A GRAY vertex reachable from the current DFS path means a directed back-edge
 * → directed cycle exists.
 */
bool DirectedGraph::has_directed_cycle() const
{
    std::vector<int> color(n, 0);

    std::function<bool(int)> dfs = [&](int u) -> bool
    {
        color[u] = 1; // GRAY
        for (int nb : out_adj[u])
        {
            if (!active[nb])
                continue;
            if (color[nb] == 1)
                return true; // back edge → cycle
            if (color[nb] == 0 && dfs(nb))
                return true;
        }
        color[u] = 2; // BLACK
        return false;
    };

    for (int v = 0; v < n; ++v)
    {
        if (active[v] && color[v] == 0)
        {
            if (dfs(v))
                return true;
        }
    }
    return false;
}

/**
 * Find one directed cycle.
 * Uses DFS with parent tracking to extract the cycle path.
 *
 * @return Directed cycle as a list of vertices, or {} if the graph is a DAG.
 */
std::vector<int> DirectedGraph::find_directed_cycle() const
{
    std::vector<int> color(n, 0);
    std::vector<int> parent(n, -1);
    std::vector<int> cycle;

    std::function<bool(int)> dfs = [&](int u) -> bool
    {
        color[u] = 1;
        for (int nb : out_adj[u])
        {
            if (!active[nb])
                continue;
            if (color[nb] == 1)
            {
                // Back edge u → nb. Extract cycle: nb … u → nb
                cycle.push_back(nb);
                int cur = u;
                while (cur != nb)
                {
                    cycle.push_back(cur);
                    cur = parent[cur];
                }
                // cycle is now reversed; reverse it for correct order
                std::reverse(cycle.begin(), cycle.end());
                return true;
            }
            if (color[nb] == 0)
            {
                parent[nb] = u;
                if (dfs(nb))
                    return true;
            }
        }
        color[u] = 2;
        return false;
    };

    for (int v = 0; v < n; ++v)
    {
        if (active[v] && color[v] == 0)
        {
            if (dfs(v))
                break;
        }
    }
    return cycle;
}

// ─── Tarjan's SCC Algorithm ───────────────────────────────────────────────────

/**
 * Tarjan's Strongly Connected Component algorithm.
 *
 * Implementation details:
 *   - disc[v]  = discovery time (DFS timestamp)
 *   - low[v]   = lowest disc reachable from v's subtree
 *   - on_stack[v] = true while v is on the SCC candidate stack
 *
 * When DFS finishes processing v:
 *   If low[v] == disc[v], v is the root of an SCC.
 *   Pop the stack until we pop v — those are the SCC members.
 *
 * @return All SCCs as a list of vertex lists (only active vertices included).
 */
std::vector<std::vector<int>> DirectedGraph::find_SCCs() const
{
    std::vector<int> disc(n, -1);
    std::vector<int> low(n, -1);
    std::vector<bool> on_stack(n, false);
    std::stack<int> stk;
    int timer = 0;

    std::vector<std::vector<int>> sccs;

    std::function<void(int)> dfs = [&](int u)
    {
        disc[u] = low[u] = timer++;
        stk.push(u);
        on_stack[u] = true;

        for (int nb : out_adj[u])
        {
            if (!active[nb])
                continue;
            if (disc[nb] == -1)
            {
                dfs(nb);
                low[u] = std::min(low[u], low[nb]);
            }
            else if (on_stack[nb])
            {
                // nb is an ancestor on the current path
                low[u] = std::min(low[u], disc[nb]);
            }
        }

        // u is the root of an SCC
        if (low[u] == disc[u])
        {
            std::vector<int> scc;
            while (true)
            {
                int w = stk.top();
                stk.pop();
                on_stack[w] = false;
                scc.push_back(w);
                if (w == u)
                    break;
            }
            sccs.push_back(std::move(scc));
        }
    };

    for (int v = 0; v < n; ++v)
    {
        if (active[v] && disc[v] == -1)
            dfs(v);
    }
    return sccs;
}

// ─── Deep copy ───────────────────────────────────────────────────────────────

DirectedGraph DirectedGraph::copy() const
{
    DirectedGraph g(n);
    g.active = active;
    g.out_adj = out_adj;
    g.in_adj = in_adj;
    return g;
}

std::vector<int> DirectedGraph::find_shortest_directed_cycle() const
{
    int best_len = INT_MAX;
    std::vector<int> best_cycle;

    for (int s = 0; s < n; ++s)
    {
        if (!active[s])
            continue;

        std::vector<int> dist(n, -1);
        std::vector<int> parent(n, -1);
        std::queue<int> q;

        dist[s] = 0;
        q.push(s);

        while (!q.empty())
        {
            int u = q.front();
            q.pop();

            if (dist[u] + 1 >= best_len)
                continue;

            for (int nb : out_adj[u])
            {
                if (!active[nb])
                    continue;

                if (nb == s)
                {
                    int len = dist[u] + 1;
                    if (len < best_len)
                    {
                        std::vector<int> path_rev;
                        int cur = u;
                        while (cur != s && cur != -1)
                        {
                            path_rev.push_back(cur);
                            cur = parent[cur];
                        }
                        if (cur == s)
                        {
                            std::reverse(path_rev.begin(), path_rev.end());
                            std::vector<int> cycle;
                            cycle.push_back(s);
                            cycle.insert(cycle.end(), path_rev.begin(), path_rev.end());
                            if (static_cast<int>(cycle.size()) == len)
                            {
                                best_len = len;
                                best_cycle = std::move(cycle);
                            }
                        }
                    }
                    continue;
                }

                if (dist[nb] == -1)
                {
                    dist[nb] = dist[u] + 1;
                    parent[nb] = u;
                    q.push(nb);
                }
            }
        }
    }

    return best_cycle;
}