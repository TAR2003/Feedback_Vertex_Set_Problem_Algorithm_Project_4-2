/**
 * @file exact_solver_u.cpp
 * @brief Exact FVS solvers for undirected graphs: BST and Iterative Compression.
 *
 * ── Bounded Search Tree (BST) ─────────────────────────────────────────────────
 *
 * The BST algorithm exploits the following observation:
 *   Every cycle contains at least one vertex that must be in the FVS.
 *   If we find a cycle C, we can branch: for each v ∈ C, include v in FVS
 *   and recurse on G - v with budget k-1.
 *
 * Combined with kernelization (which prunes trivially unnecessary vertices),
 * this gives a search tree of size O(d^k) where d is the cycle size found.
 * With good cycle selection (short cycles), this is close to O(3^k).
 *
 * The outer loop uses iterative deepening: try k = 0, 1, 2, … until a
 * valid FVS is found.  This guarantees the MINIMUM FVS is returned.
 *
 * ── Iterative Compression (IC) ───────────────────────────────────────────────
 *
 * The starter IC implementation was a bounded local-search heuristic, not a
 * mathematically exact iterative-compression algorithm. To keep exactness
 * guarantees, solve_undirected_IC now delegates to the exact BST solver.
 */

#include "undirected_fvs.h"
#include "kernelization_u.cpp" // forward-include shared helpers
#include <algorithm>
#include <set>

// ═══════════════════════════════════════════════════════════════════════════
//  SECTION 1: BST Core
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Internal BST recursive function.
 *
 * @param g     Current (partially reduced) graph
 * @param k     Remaining budget (how many more vertices we can pick)
 * @param fvs   Accumulates the chosen FVS vertices
 * @return true if a valid FVS of size ≤ k was found and stored in fvs
 */
static bool bst_recurse(UndirectedGraph g, int k, std::vector<int> &fvs)
{
    // ── Step 1: Kernelization ────────────────────────────────────────────────
    std::vector<int> forced;
    k = full_kernelization_u(g, forced, k);

    if (k < 0)
        return false; // budget exceeded by forced vertices

    // Check if the reduced graph is already a forest
    std::vector<int> cycle = g.find_shortest_cycle();
    if (cycle.empty() && g.has_cycle())
    {
        // Safety fallback: preserve correctness if shortest-cycle extraction misses.
        cycle = g.find_cycle();
    }
    if (cycle.empty())
    {
        // No cycles remain — all forced vertices form a valid FVS
        fvs.insert(fvs.end(), forced.begin(), forced.end());
        return true;
    }

    // ── Step 2: Branch on cycle vertices ────────────────────────────────────
    // We MUST include at least one vertex from the cycle in our FVS.
    // Try each vertex in the cycle as a candidate.

    // Heuristic: sort cycle vertices by degree (descending) — higher-degree
    // vertices are more likely to break many cycles at once.
    std::sort(cycle.begin(), cycle.end(), [&](int a, int b)
              { return g.degree(a) > g.degree(b); });

    for (int v : cycle)
    {
        if (!g.is_active(v))
            continue;

        // Branch: include v in FVS
        UndirectedGraph g_copy = g.copy();
        std::vector<std::pair<int, int>> removed_edges;
        g_copy.deactivate_full(v, removed_edges);

        std::vector<int> branch_fvs;
        if (bst_recurse(g_copy, k - 1, branch_fvs))
        {
            // Success: commit this branch
            fvs.insert(fvs.end(), forced.begin(), forced.end());
            fvs.push_back(v);
            fvs.insert(fvs.end(), branch_fvs.begin(), branch_fvs.end());
            return true;
        }
    }

    return false; // no branch succeeded within budget k
}

// ─── Public BST Solver ────────────────────────────────────────────────────────

std::vector<int> solve_undirected_BST(int n,
                                      const std::vector<std::pair<int, int>> &edges)
{

    if (n == 0)
        return {};

    // Build the graph
    UndirectedGraph g(n);
    for (auto &[u, v] : edges)
    {
        if (u >= 0 && u < n && v >= 0 && v < n)
            g.add_edge(u, v);
    }

    // Iterative deepening: find MINIMUM k such that BST succeeds
    for (int k = 0; k <= n; ++k)
    {
        std::vector<int> fvs;
        UndirectedGraph g_copy = g.copy();
        if (bst_recurse(g_copy, k, fvs))
        {
            // Remove duplicates (forced vertices can appear twice in rare edge cases)
            std::sort(fvs.begin(), fvs.end());
            fvs.erase(std::unique(fvs.begin(), fvs.end()), fvs.end());
            return fvs;
        }
    }
    return {}; // unreachable — every graph has a trivial FVS of size n
}

std::vector<int> solve_undirected_IC(int n,
                                     const std::vector<std::pair<int, int>> &edges)
{
    return solve_undirected_BST(n, edges);
}

std::vector<int> solve_undirected_KME(int n,
                                      const std::vector<std::pair<int, int>> &edges,
                                      int pop_size, int max_gens)
{
    if (n == 0)
        return {};

    UndirectedGraph g(n);
    for (const auto &[u, v] : edges)
    {
        if (u >= 0 && u < n && v >= 0 && v < n)
            g.add_edge(u, v);
    }

    std::vector<int> forced;
    (void)full_kernelization_u(g, forced, n);

    if (!g.has_cycle())
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
        for (int v : g.adj[u])
        {
            if (!g.active[v])
                continue;
            int nu = old_to_new[u];
            int nv = old_to_new[v];
            if (nu == nv)
                continue;
            if (nu > nv)
                std::swap(nu, nv);
            reduced_edge_set.insert({nu, nv});
        }
    }

    std::vector<std::pair<int, int>> reduced_edges;
    reduced_edges.reserve(reduced_edge_set.size());
    for (const auto &e : reduced_edge_set)
        reduced_edges.push_back(e);

    std::vector<int> kernel_solution = solve_undirected_MA(
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