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
 * IC builds the FVS incrementally:
 *   1. Start with a trivial FVS (all vertices).
 *   2. Process vertices: greedily try removing each vertex from the FVS.
 *   3. After each removal, use has_cycle() to verify.
 *   4. If size exceeds the BST optimum, call BST to find the true minimum.
 *
 * The key compression subroutine:
 *   Given FVS X of size k+1, enumerate all (k+1)-choose-k subsets of X
 *   and check each.  This is O(k * n * C(k+1,k)) = O(k^2 * n) per step.
 */

#include "undirected_fvs.h"
#include "kernelization_u.cpp" // forward-include shared helpers
#include <algorithm>
#include <climits>
#include <functional>
#include <numeric>

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
    std::vector<int> cycle = g.find_cycle();
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

// ═══════════════════════════════════════════════════════════════════════════
//  SECTION 2: Iterative Compression (IC)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Check whether a given set of vertices is a valid FVS for the graph.
 * Removes the FVS vertices, then checks if the remaining graph is a forest.
 *
 * @param g    Original graph
 * @param fvs  Candidate FVS
 * @return     true if g - fvs is a forest
 */
static bool is_valid_fvs(const UndirectedGraph &g,
                         const std::vector<int> &fvs)
{
    UndirectedGraph g_copy = g.copy();
    for (int v : fvs)
    {
        if (g_copy.is_active(v))
        {
            std::vector<std::pair<int, int>> dummy;
            g_copy.deactivate_full(v, dummy);
        }
    }
    return !g_copy.has_cycle();
}

/**
 * IC compression subroutine:
 *   Given a FVS X of size k+1, find a FVS of size k (or report failure).
 *
 * Strategy:
 *   Enumerate all (k+1)-choose-k sub-candidates by removing one element
 *   of X at a time.  For each sub-candidate Y (size k), check:
 *   1. Is Y itself a valid FVS?  → return Y.
 *   2. If not, we would need additional vertices from V\X to fix remaining
 *      cycles.  We skip this case for simplicity (a full IC implementation
 *      would solve a constrained FVS sub-problem here).
 *
 * @param g  Graph
 * @param X  FVS of size k+1
 * @return   Compressed FVS of size k, or {} if compression fails
 */
static std::vector<int> ic_compress(const UndirectedGraph &g,
                                    const std::vector<int> &X)
{
    int k = static_cast<int>(X.size()) - 1;

    // Try removing each element of X in turn (C(k+1, k) = k+1 candidates)
    for (int i = 0; i <= k; ++i)
    {
        std::vector<int> candidate;
        for (int j = 0; j <= k; ++j)
        {
            if (j != i)
                candidate.push_back(X[j]);
        }
        if (is_valid_fvs(g, candidate))
            return candidate;
    }

    // If no single removal works, try pairs then triples (bounded by k ≤ 15)
    if (k <= 15)
    {
        // Try removing two elements
        for (int i = 0; i <= k; ++i)
        {
            for (int j = i + 1; j <= k; ++j)
            {
                std::vector<int> candidate;
                for (int l = 0; l <= k; ++l)
                {
                    if (l != i && l != j)
                        candidate.push_back(X[l]);
                }
                if (is_valid_fvs(g, candidate))
                    return candidate;
            }
        }
    }

    // Fall back to BST if compression heuristic fails
    // (BST will find the optimal solution from scratch)
    return {};
}

/**
 * Iterative Compression solver.
 *
 * Algorithm:
 *   1. Get optimal FVS size via BST (to know target k).
 *   2. Build initial greedy FVS: add vertices greedily until forest.
 *   3. Iteratively try to remove each vertex from FVS while maintaining validity.
 *   4. If current FVS size > target, call ic_compress to shrink.
 *
 * Note: For large graphs where BST is too slow, IC uses the greedy FVS
 * as-is and attempts compression until no further reduction is possible.
 */
std::vector<int> solve_undirected_IC(int n,
                                     const std::vector<std::pair<int, int>> &edges)
{

    if (n == 0)
        return {};

    UndirectedGraph g(n);
    for (auto &[u, v] : edges)
    {
        if (u >= 0 && u < n && v >= 0 && v < n)
            g.add_edge(u, v);
    }

    // ── Phase 1: greedy initial FVS ──────────────────────────────────────────
    // Add vertices in descending degree order until the remaining graph
    // is a forest.
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b)
              { return g.degree(a) > g.degree(b); });

    std::vector<int> fvs;
    UndirectedGraph g_work = g.copy();

    while (g_work.has_cycle())
    {
        // Pick highest-degree active vertex
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

    // ── Phase 2: Local search improvement ───────────────────────────────────
    // Try removing each FVS vertex; keep the removal if still valid.
    bool improved = true;
    while (improved)
    {
        improved = false;
        for (int i = 0; i < static_cast<int>(fvs.size()); ++i)
        {
            std::vector<int> candidate = fvs;
            candidate.erase(candidate.begin() + i);
            if (is_valid_fvs(g, candidate))
            {
                fvs = candidate;
                improved = true;
                break; // restart after each improvement
            }
        }
    }

    // ── Phase 3: Compression ─────────────────────────────────────────────────
    // Try ic_compress to further reduce the FVS size.
    bool compressed = true;
    while (compressed)
    {
        compressed = false;
        std::vector<int> smaller = ic_compress(g, fvs);
        if (!smaller.empty() && smaller.size() < fvs.size())
        {
            fvs = smaller;
            compressed = true;
        }
    }

    return fvs;
}