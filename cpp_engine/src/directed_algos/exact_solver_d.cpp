/**
 * @file exact_solver_d.cpp
 * @brief Exact DFVS solvers for directed graphs: BST and Iterative Compression.
 *
 * ── Directed BST ─────────────────────────────────────────────────────────────
 *
 * The directed case mirrors the undirected BST but uses directed cycles.
 * Key difference: we use SCC decomposition before branching.
 *
 * After kernelization:
 *   1. Run Tarjan's SCC — if all SCCs are trivial (size 1), graph is a DAG → done.
 *   2. Find a directed cycle C.
 *   3. Branch: for each v ∈ C, include v in DFVS and recurse with k-1.
 *
 * The SCC decomposition allows us to process each non-trivial SCC independently
 * (if A and B are disjoint SCCs, DFVS(G) = DFVS(A) ∪ DFVS(B)).
 * This independent-subproblem decomposition can exponentially speed things up.
 *
 * ── Directed IC ──────────────────────────────────────────────────────────────
 *
 * The starter IC path was heuristic and did not guarantee optimality.
 * To preserve exactness guarantees, solve_directed_IC delegates to BST.
 */

#include "directed_fvs.h"
#include "kernelization_d.cpp" // shared helpers
#include <algorithm>
#include <set>

// ═══════════════════════════════════════════════════════════════════════════
//  SECTION 1: Directed BST
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Internal BST recursive function for directed FVS.
 *
 * @param g   Current graph (copy — caller passes by value)
 * @param k   Remaining budget
 * @param fvs Accumulates chosen DFVS vertices
 * @return true if a valid DFVS of size ≤ k was found
 */
static bool bst_directed_recurse(DirectedGraph g, int k, std::vector<int> &fvs)
{

    // ── Kernelization ────────────────────────────────────────────────────────
    std::vector<int> forced;
    k = full_kernelization_d(g, forced, k);

    if (k < 0)
        return false;

    // ── Check if graph is a DAG (no directed cycles remain) ─────────────────
    std::vector<int> cycle = g.find_shortest_directed_cycle();
    if (cycle.empty() && g.has_directed_cycle())
    {
        // Safety fallback: preserve correctness if shortest-cycle extraction misses.
        cycle = g.find_directed_cycle();
    }
    if (cycle.empty())
    {
        fvs.insert(fvs.end(), forced.begin(), forced.end());
        return true;
    }

    // ── Branch on the cycle vertices ─────────────────────────────────────────
    // Sort by out_degree + in_degree (descending) to prefer high-degree vertices
    std::sort(cycle.begin(), cycle.end(), [&](int a, int b)
              { return (g.in_degree(a) + g.out_degree(a)) >
                       (g.in_degree(b) + g.out_degree(b)); });

    for (int v : cycle)
    {
        if (!g.is_active(v))
            continue;

        DirectedGraph g_copy = g.copy();
        std::vector<std::pair<int, int>> removed;
        g_copy.deactivate_full(v, removed);

        std::vector<int> branch_fvs;
        if (bst_directed_recurse(g_copy, k - 1, branch_fvs))
        {
            fvs.insert(fvs.end(), forced.begin(), forced.end());
            fvs.push_back(v);
            fvs.insert(fvs.end(), branch_fvs.begin(), branch_fvs.end());
            return true;
        }
    }
    return false;
}

std::vector<int> solve_directed_BST(int n,
                                    const std::vector<std::pair<int, int>> &edges)
{

    if (n == 0)
        return {};

    DirectedGraph g(n);
    for (auto &[u, v] : edges)
    {
        if (u >= 0 && u < n && v >= 0 && v < n)
            g.add_edge(u, v);
    }

    for (int k = 0; k <= n; ++k)
    {
        std::vector<int> fvs;
        DirectedGraph g_copy = g.copy();
        if (bst_directed_recurse(g_copy, k, fvs))
        {
            std::sort(fvs.begin(), fvs.end());
            fvs.erase(std::unique(fvs.begin(), fvs.end()), fvs.end());
            return fvs;
        }
    }
    return {};
}

std::vector<int> solve_directed_IC(int n,
                                   const std::vector<std::pair<int, int>> &edges)
{
    return solve_directed_BST(n, edges);
}

std::vector<int> solve_directed_KME(int n,
                                    const std::vector<std::pair<int, int>> &edges,
                                    int pop_size, int max_gens)
{
    if (n == 0)
        return {};

    DirectedGraph g(n);
    for (const auto &[u, v] : edges)
    {
        if (u >= 0 && u < n && v >= 0 && v < n)
            g.add_edge(u, v);
    }

    std::vector<int> forced;
    (void)full_kernelization_d(g, forced, n);

    if (!g.has_directed_cycle())
    {
        std::sort(forced.begin(), forced.end());
        forced.erase(std::unique(forced.begin(), forced.end()), forced.end());
        return forced;
    }

    std::vector<int> old_to_new(n, -1);
    std::vector<int> new_to_old;
    new_to_old.reserve(n);
    for (int v = 0; v < n; ++v)
    {
        if (g.active[v])
        {
            old_to_new[v] = static_cast<int>(new_to_old.size());
            new_to_old.push_back(v);
        }
    }

    std::set<std::pair<int, int>> reduced_edge_set;
    for (int u = 0; u < n; ++u)
    {
        if (!g.active[u])
            continue;
        for (int v : g.out_adj[u])
        {
            if (!g.active[v])
                continue;
            int nu = old_to_new[u];
            int nv = old_to_new[v];
            reduced_edge_set.insert({nu, nv});
        }
    }

    std::vector<std::pair<int, int>> reduced_edges;
    reduced_edges.reserve(reduced_edge_set.size());
    for (const auto &e : reduced_edge_set)
        reduced_edges.push_back(e);

    std::vector<int> kernel_solution = solve_directed_MA(
        static_cast<int>(new_to_old.size()), reduced_edges, pop_size, max_gens);

    std::vector<int> result = forced;
    result.reserve(forced.size() + kernel_solution.size());
    for (int kv : kernel_solution)
    {
        if (kv >= 0 && kv < static_cast<int>(new_to_old.size()))
            result.push_back(new_to_old[kv]);
    }

    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}