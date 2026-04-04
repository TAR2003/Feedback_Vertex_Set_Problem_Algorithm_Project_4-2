/**
 * @file exact_solver_d.cpp
 * @brief Exact DFVS solvers: directed BST and Iterative Compression.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 *  ALGORITHM 1: Directed Bounded Search Tree (BST)
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *  Identical structure to undirected BST, but:
 *  (a) Cycles are DIRECTED cycles (found by DFS with GRAY/BLACK coloring).
 *  (b) Kernelization uses directed rules D0/D1 (self-loop, source/sink).
 *  (c) SCC pre-processing: any vertex whose SCC is trivial (size 1, no
 *      self-loop) is guaranteed NOT to be on any directed cycle, so we
 *      delete it before the main search.
 *
 *  Core observation (same as undirected):
 *    Pick any directed cycle C.  At least one vertex of C must be in the DFVS.
 *    Branch on each v ∈ C: "include v, recurse on G–{v} with k–1".
 *
 *  SCC decomposition pre-pass:
 *    Run Tarjan's algorithm once.  Vertices in SCCs of size 1 (no self-loop)
 *    cannot be on any directed cycle → safe to delete.
 *    For each non-trivial SCC, solve the DFVS problem independently and
 *    union the results.  This is correct because:
 *      - A directed cycle is entirely within one SCC.
 *      - Removing a DFVS from one SCC does not affect other SCCs.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 *  ALGORITHM 2: Directed Iterative Compression (IC)
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *  Same structure as undirected IC:
 *  1. Add vertices v_0, …, v_{n-1} one by one.
 *  2. After adding v_i, X ∪ {v_i} is valid DFVS for G_i.
 *  3. COMPRESS(G_i, X ∪ {v_i}, |X|): enumerate 2^|X| subsets Z ⊆ X,
 *     check G[Y=X\Z] is a DAG (directed analog of "is a forest"),
 *     find restricted-BST solution from V\X.
 *
 *  Key difference from undirected:
 *    "G[Y] is a forest" becomes "G[Y] is a DAG" (no directed cycles in Y).
 *    "Any remaining cycle passes through V\X" still holds:
 *      If G[Y] is a DAG, any directed cycle in G–Z must pass through V\X.
 */

#include "directed_fvs.h"
#include <algorithm>
#include <functional>
#include <numeric>
#include <unordered_set>
#include <vector>

// ══════════════════════════════════════════════════════════════════════════════
//  Shared helpers
// ══════════════════════════════════════════════════════════════════════════════

/**
 * Check whether the directed subgraph induced by `verts` contains a
 * directed cycle.  Uses DFS coloring restricted to `verts`.
 *
 * Colors: WHITE=0 (unvisited), GRAY=1 (on stack), BLACK=2 (done).
 * A back-edge to a GRAY vertex signals a directed cycle.
 */
static bool induced_has_dcycle(const DirectedGraph &g,
                               const std::vector<int> &verts)
{
    if (verts.empty())
        return false;
    std::unordered_set<int> vset(verts.begin(), verts.end());
    std::unordered_map<int, int> color;
    for (int v : verts)
        color[v] = 0;

    std::function<bool(int)> dfs = [&](int u) -> bool
    {
        color[u] = 1; // GRAY: on current DFS path
        for (int nb : g.out_adj[u])
        {
            if (!g.active[nb] || !vset.count(nb))
                continue;
            if (color[nb] == 1)
                return true; // back-edge → directed cycle
            if (color[nb] == 0 && dfs(nb))
                return true;
        }
        color[u] = 2; // BLACK: done
        return false;
    };

    for (int v : verts)
        if (color[v] == 0 && dfs(v))
            return true;

    return false;
}

/**
 * Verify that `fvs` is a valid DFVS: G–fvs is a DAG.
 */
static bool is_valid_dfvs(const DirectedGraph &g,
                          const std::vector<int> &fvs)
{
    DirectedGraph tmp = g.copy();
    for (int v : fvs)
    {
        if (tmp.is_active(v))
        {
            std::vector<std::pair<int, int>> dummy;
            tmp.deactivate_full(v, dummy);
        }
    }
    return !tmp.has_directed_cycle();
}

/**
 * Delete all vertices that are NOT in a non-trivial SCC.
 * A non-trivial SCC has size > 1, OR has size 1 with a self-loop.
 * Self-loop vertices are handled by Rule D0 before this runs.
 */
static void scc_prune(DirectedGraph &g)
{
    auto sccs = g.find_SCCs();
    std::unordered_set<int> in_nontrivial;
    for (auto &scc : sccs)
        if (scc.size() > 1)
            for (int v : scc)
                in_nontrivial.insert(v);

    for (int v = 0; v < g.n; ++v)
        if (g.active[v] && !in_nontrivial.count(v))
        {
            std::vector<std::pair<int, int>> dummy;
            g.deactivate_full(v, dummy);
        }
}

// ══════════════════════════════════════════════════════════════════════════════
//  ALGORITHM 1: Directed BST
// ══════════════════════════════════════════════════════════════════════════════

/**
 * Directed BST recursive function.
 *
 * @param g        Current directed graph (copy)
 * @param k        Remaining budget
 * @param fvs_out  Accumulates chosen DFVS vertices
 * @return true if valid DFVS of size ≤ original_k found
 */
static bool bst_directed_recurse(DirectedGraph g, int k,
                                 std::vector<int> &fvs_out)
{
    // ── Step 1: Kernelization (D0, D1) + SCC pruning ─────────────────────────
    std::vector<int> forced;
    if (!kernelize_directed(g, forced, k))
        return false;

    // SCC pruning: vertices not in any non-trivial SCC can't be on any cycle
    scc_prune(g);

    // Apply D0/D1 again after SCC pruning (new sources/sinks may appear)
    if (!kernelize_directed(g, forced, k))
        return false;

    // ── Step 2: Find a directed cycle ─────────────────────────────────────────
    std::vector<int> cycle = g.find_directed_cycle();

    if (cycle.empty())
    {
        // Graph is a DAG → forced vertices complete the DFVS
        fvs_out.insert(fvs_out.end(), forced.begin(), forced.end());
        return true;
    }

    if (k <= 0)
        return false;

    // ── Step 3: Branch on each cycle vertex ──────────────────────────────────
    // Sort by degree (in + out) descending for better pruning
    std::sort(cycle.begin(), cycle.end(), [&](int a, int b)
              { return (g.in_degree(a) + g.out_degree(a)) >
                       (g.in_degree(b) + g.out_degree(b)); });

    for (int v : cycle)
    {
        if (!g.is_active(v))
            continue;

        DirectedGraph g_branch = g.copy();
        std::vector<std::pair<int, int>> dummy;
        g_branch.deactivate_full(v, dummy);

        std::vector<int> sub_fvs;
        if (bst_directed_recurse(g_branch, k - 1, sub_fvs))
        {
            fvs_out.insert(fvs_out.end(), forced.begin(), forced.end());
            fvs_out.push_back(v);
            fvs_out.insert(fvs_out.end(), sub_fvs.begin(), sub_fvs.end());
            return true;
        }
    }
    return false;
}

/**
 * Public directed BST solver.  Iterative deepening over k = 0, 1, 2, …
 */
std::vector<int> solve_directed_BST(int n,
                                    const std::vector<std::pair<int, int>> &edges)
{
    if (n == 0)
        return {};

    DirectedGraph g(n);
    for (auto &[u, v] : edges)
        if (u >= 0 && u < n && v >= 0 && v < n)
            g.add_edge(u, v);

    for (int k = 0; k <= n; ++k)
    {
        std::vector<int> fvs;
        if (bst_directed_recurse(g.copy(), k, fvs))
        {
            std::sort(fvs.begin(), fvs.end());
            fvs.erase(std::unique(fvs.begin(), fvs.end()), fvs.end());
            return fvs;
        }
    }
    return {};
}

// ══════════════════════════════════════════════════════════════════════════════
//  ALGORITHM 2: Directed Iterative Compression (IC)
// ══════════════════════════════════════════════════════════════════════════════

/**
 * DIRECTED RESTRICTED BST:
 *   Find a DFVS of `g` using ONLY vertices NOT in `forbidden`, budget ≤ k.
 *
 *   Since G[Y] (Y = X\Z) is a DAG (checked before calling), any directed
 *   cycle in G–Z must pass through at least one vertex from V\X (non-forbidden).
 *   So we can always branch on a non-forbidden vertex.
 */
static bool restricted_bst_directed(DirectedGraph g,
                                    const std::unordered_set<int> &forbidden,
                                    int budget,
                                    std::vector<int> &fvs_out)
{
    // Kernelization
    std::vector<int> forced;
    int k = budget;
    if (!kernelize_directed(g, forced, k))
        return false;

    for (int v : forced)
        if (forbidden.count(v))
            return false;

    // SCC prune
    scc_prune(g);
    if (!kernelize_directed(g, forced, k))
        return false;
    for (int v : forced)
        if (forbidden.count(v))
            return false;

    std::vector<int> cycle = g.find_directed_cycle();

    if (cycle.empty())
    {
        fvs_out.insert(fvs_out.end(), forced.begin(), forced.end());
        return true;
    }

    if (k <= 0)
        return false;

    // Find non-forbidden branching candidates in the cycle
    std::vector<int> candidates;
    for (int v : cycle)
        if (g.is_active(v) && !forbidden.count(v))
            candidates.push_back(v);

    if (candidates.empty())
        return false; // cycle trapped in forbidden → FAIL

    for (int v : candidates)
    {
        DirectedGraph g_branch = g.copy();
        std::vector<std::pair<int, int>> dummy;
        g_branch.deactivate_full(v, dummy);

        std::vector<int> sub_fvs;
        if (restricted_bst_directed(g_branch, forbidden, k - 1, sub_fvs))
        {
            fvs_out.insert(fvs_out.end(), forced.begin(), forced.end());
            fvs_out.push_back(v);
            fvs_out.insert(fvs_out.end(), sub_fvs.begin(), sub_fvs.end());
            return true;
        }
    }
    return false;
}

/**
 * DIRECTED COMPRESS(G, X, k):
 *   X is DFVS of size k+1.  Find DFVS of size ≤ k.
 *
 *   Enumerate 2^|X| subsets Z ⊆ X.
 *   For each Z:
 *     Y = X\Z must induce a DAG in G (directed analog of "forest")
 *     Call restricted_bst_directed on G–Z to find F_W from V\X.
 *     Return Z ∪ F_W if valid.
 */
static std::vector<int> compress_directed(const DirectedGraph &G,
                                          const std::vector<int> &X,
                                          int k)
{
    int sz = (int)X.size();
    std::unordered_set<int> forbidden(X.begin(), X.end());

    for (int mask = 0; mask < (1 << sz); ++mask)
    {
        std::vector<int> Z, Y;
        for (int i = 0; i < sz; ++i)
        {
            if (mask >> i & 1)
                Z.push_back(X[i]);
            else
                Y.push_back(X[i]);
        }

        if ((int)Z.size() > k)
            continue;

        // G[Y] must be a DAG (no directed cycles)
        if (induced_has_dcycle(G, Y))
            continue;

        // Build G' = G – Z
        DirectedGraph G_prime = G.copy();
        for (int v : Z)
        {
            if (G_prime.is_active(v))
            {
                std::vector<std::pair<int, int>> dummy;
                G_prime.deactivate_full(v, dummy);
            }
        }

        int budget = k - (int)Z.size();
        std::vector<int> F_W;
        if (restricted_bst_directed(G_prime, forbidden, budget, F_W))
        {
            std::vector<int> candidate = Z;
            candidate.insert(candidate.end(), F_W.begin(), F_W.end());
            if (is_valid_dfvs(G, candidate))
                return candidate;
        }
    }
    return {};
}

/**
 * Public directed IC solver.
 */
std::vector<int> solve_directed_IC(int n,
                                   const std::vector<std::pair<int, int>> &edges)
{
    if (n == 0)
        return {};

    // Build adjacency lists indexed per-vertex for incremental construction
    std::vector<std::vector<int>> out_full(n), in_full(n);
    for (auto &[u, v] : edges)
        if (u >= 0 && u < n && v >= 0 && v < n)
        {
            out_full[u].push_back(v);
            in_full[v].push_back(u);
        }

    // Process high-total-degree vertices first for better early compression.
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b)
              {
                  return (out_full[a].size() + in_full[a].size()) >
                         (out_full[b].size() + in_full[b].size());
              });

    DirectedGraph G_curr(n);
    for (int v = 0; v < n; ++v)
        G_curr.active[v] = false;

    std::vector<int> X; // current DFVS

    for (int i : order)
    {
        // Activate vertex i and add its outgoing edges to already-active vertices
        G_curr.active[i] = true;
        for (int nb : out_full[i])
            if (G_curr.is_active(nb))
                G_curr.add_edge(i, nb);
        // Also add incoming edges from already-active vertices
        for (int u : in_full[i])
            if (G_curr.is_active(u))
                G_curr.add_edge(u, i);

        // X ∪ {i} is trivially a valid DFVS for G_curr
        X.push_back(i);

        // Repeatedly compress until no further shrink is possible.
        while (true)
        {
            int target_k = (int)X.size() - 1;
            if (target_k < 0)
                break;

            std::vector<int> compressed = compress_directed(G_curr, X, target_k);
            if (!compressed.empty())
                X = compressed;
            else
                break;
        }
    }

    // Deduplicate
    std::sort(X.begin(), X.end());
    X.erase(std::unique(X.begin(), X.end()), X.end());

    // Final local cleanup: try removing each vertex one more time
    bool improved = true;
    while (improved)
    {
        improved = false;
        for (int i = 0; i < (int)X.size(); ++i)
        {
            std::vector<int> candidate = X;
            candidate.erase(candidate.begin() + i);
            if (is_valid_dfvs(G_curr, candidate))
            {
                X = candidate;
                improved = true;
                break;
            }
        }
    }

    return X;
}