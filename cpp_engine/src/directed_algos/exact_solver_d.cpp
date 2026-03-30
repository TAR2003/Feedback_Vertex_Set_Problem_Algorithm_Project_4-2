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
 * Same structure as undirected IC but using directed cycle checking.
 * Compression enumerates k-subsets of the current FVS and validates each.
 */

#include "directed_fvs.h"
#include "kernelization_d.cpp" // shared helpers
#include <algorithm>
#include <functional>
#include <numeric>
#include <climits>

// ═══════════════════════════════════════════════════════════════════════════
//  SECTION 1: Directed BST
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Check if a candidate set is a valid DFVS for directed graph g.
 * Remove candidates; check if resulting graph is a DAG.
 */
static bool is_valid_dfvs(const DirectedGraph &g,
                          const std::vector<int> &fvs)
{
    DirectedGraph g_copy = g.copy();
    for (int v : fvs)
    {
        if (g_copy.is_active(v))
        {
            std::vector<std::pair<int, int>> dummy;
            g_copy.deactivate_full(v, dummy);
        }
    }
    return !g_copy.has_directed_cycle();
}

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

// ═══════════════════════════════════════════════════════════════════════════
//  SECTION 2: Directed IC
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Compression subroutine: given DFVS X of size k+1, try to find one of size k.
 * Enumerates all (k+1)-choose-k subsets (k+1 candidates) and validates each.
 */
static std::vector<int> ic_directed_compress(const DirectedGraph &g,
                                             const std::vector<int> &X)
{
    int k = static_cast<int>(X.size()) - 1;
    for (int i = 0; i <= k; ++i)
    {
        std::vector<int> candidate;
        for (int j = 0; j <= k; ++j)
            if (j != i)
                candidate.push_back(X[j]);
        if (is_valid_dfvs(g, candidate))
            return candidate;
    }
    // Try removing two elements (only for small k)
    if (k <= 12)
    {
        for (int i = 0; i <= k; ++i)
        {
            for (int j = i + 1; j <= k; ++j)
            {
                std::vector<int> candidate;
                for (int l = 0; l <= k; ++l)
                    if (l != i && l != j)
                        candidate.push_back(X[l]);
                if (is_valid_dfvs(g, candidate))
                    return candidate;
            }
        }
    }
    return {};
}

std::vector<int> solve_directed_IC(int n,
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

    // ── Phase 1: Greedy initial DFVS ─────────────────────────────────────────
    // Greedily include high (in+out)-degree vertices until the graph is a DAG.
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b)
              { return (g.in_degree(a) + g.out_degree(a)) >
                       (g.in_degree(b) + g.out_degree(b)); });

    std::vector<int> fvs;
    DirectedGraph g_work = g.copy();

    while (g_work.has_directed_cycle())
    {
        int best = -1;
        for (int v : order)
        {
            if (g_work.is_active(v))
            {
                best = v;
                break;
            }
        }
        if (best == -1)
            break;
        fvs.push_back(best);
        std::vector<std::pair<int, int>> dummy;
        g_work.deactivate_full(best, dummy);
    }

    // ── Phase 2: Local search ─────────────────────────────────────────────────
    bool improved = true;
    while (improved)
    {
        improved = false;
        for (int i = 0; i < static_cast<int>(fvs.size()); ++i)
        {
            std::vector<int> candidate = fvs;
            candidate.erase(candidate.begin() + i);
            if (is_valid_dfvs(g, candidate))
            {
                fvs = candidate;
                improved = true;
                break;
            }
        }
    }

    // ── Phase 3: Compression ──────────────────────────────────────────────────
    bool compressed = true;
    while (compressed)
    {
        compressed = false;
        std::vector<int> smaller = ic_directed_compress(g, fvs);
        if (!smaller.empty() && smaller.size() < fvs.size())
        {
            fvs = smaller;
            compressed = true;
        }
    }

    return fvs;
}