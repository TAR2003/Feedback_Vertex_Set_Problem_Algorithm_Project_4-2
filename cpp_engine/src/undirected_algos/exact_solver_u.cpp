/**
 * @file exact_solver_u.cpp
 * @brief Exact FVS solvers: Bounded Search Tree (BST) and Iterative Compression (IC).
 *
 * ═══════════════════════════════════════════════════════════════════════════
 *  ALGORITHM 1: Bounded Search Tree (BST)
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *  Core observation:
 *    Pick ANY cycle C in the graph.  At least one vertex of C must be in the
 *    FVS (otherwise C survives in G–FVS and G–FVS is not a forest).
 *
 *  Search tree:
 *    branch on every vertex v ∈ C:
 *      "include v in FVS" → recurse on G–{v} with budget k–1
 *    If ANY branch succeeds → we found a valid FVS of size ≤ k.
 *
 *  Iterative deepening:
 *    Try k = 0, 1, 2, … until a branch succeeds.
 *    The first k that succeeds gives the MINIMUM FVS.
 *
 *  Pruning (kernelization before each branch):
 *    Apply Rules 0–2 first (degree-0/1 removal, self-loop forcing,
 *    degree-2 contraction).  These shrink the graph without expanding the
 *    search tree, cutting off many dead branches early.
 *
 *  Complexity: O(|C|^k × poly(n)).
 *    With shortest-cycle branching (|C| = 3 for triangles): O(3^k × poly(n)).
 *    This is the best known FPT bound for undirected FVS.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 *  ALGORITHM 2: Iterative Compression (IC)
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *  Based on: Reed, Smith, Vetta (2004) and Cygan et al. "Parameterized
 *  Algorithms" Chapter 4.3.
 *
 *  Key idea:
 *    Build the FVS INCREMENTALLY by adding one vertex at a time.
 *    After each step, we have an FVS X of size at most k+1.
 *    The COMPRESSION SUBROUTINE tries to shrink X from k+1 down to k.
 *
 *  Step-by-step:
 *    Order vertices: v_1, v_2, …, v_n.
 *    G_i = G[{v_1,…,v_i}]  (induced subgraph on first i vertices).
 *    X_0 = ∅ (FVS of empty graph).
 *
 *    For i = 1 to n:
 *      X = X ∪ {v_i}     ← X is now FVS for G_i (trivially correct: removing
 *                           v_i from G_i leaves G_{i-1}, which X_{i-1} handles)
 *      X = COMPRESS(G_i, X, |X|–1)   ← try to shrink X by 1
 *      (if COMPRESS fails, X stays size |X|; we accept the larger FVS)
 *
 *  COMPRESS(G, X, k):
 *    Given: X is an FVS of G of size k+1.
 *    Goal:  Find an FVS F of G of size ≤ k, or report FAIL.
 *
 *    Insight: partition X into Z (vertices from X in the new F) and Y = X\Z
 *    (vertices from X NOT in F).  Because Y ⊆ G–F and G–F is a forest,
 *    G[Y] must also be a forest.
 *
 *    Enumerate all 2^(k+1) subsets Z ⊆ X:
 *      1. Check: does G[Y] = G[X\Z] contain a cycle? If YES → skip this Z
 *                (Y can't stay in the graph without creating a cycle).
 *      2. Build G' = G – Z  (remove the chosen X-vertices from G).
 *      3. Find a set F_W ⊆ V\X of size ≤ k–|Z| that is an FVS of G'.
 *         This is called RESTRICTED-BST: same as BST but may only pick
 *         vertices from V\X for the FVS.
 *         Why can we always branch on a V\X vertex?
 *           Because G[Y] is a forest (checked above), any cycle in G'
 *           MUST pass through at least one vertex from V\X.
 *      4. If found: return Z ∪ F_W.
 *    Return FAIL if no Z works.
 *
 *  Complexity: O(2^k × k × n) per compression step, O(n × 2^k × k × n) overall.
 *  This is FPT in k.
 */

#include "undirected_fvs.h"
#include "kernelization_u.cpp" // provides kernelize_undirected()
#include <algorithm>
#include <functional>
#include <numeric>
#include <unordered_set>
#include <vector>

// ══════════════════════════════════════════════════════════════════════════════
//  Shared helpers
// ══════════════════════════════════════════════════════════════════════════════

/**
 * Check whether the subgraph induced by `verts` contains a cycle.
 * Uses Union-Find restricted to `verts`.
 */
static bool induced_has_cycle(const UndirectedGraph &g,
                              const std::vector<int> &verts)
{
    if (verts.empty())
        return false;
    std::unordered_set<int> vset(verts.begin(), verts.end());

    // Build Union-Find over verts
    std::unordered_map<int, int> parent;
    for (int v : verts)
        parent[v] = v;

    std::function<int(int)> find = [&](int x) -> int
    {
        while (parent[x] != x)
        {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };

    for (int v : verts)
    {
        for (int nb : g.adj[v])
        {
            if (!g.active[nb] || nb <= v)
                continue; // each edge once
            if (!vset.count(nb))
                continue; // only within verts
            int rv = find(v), rnb = find(nb);
            if (rv == rnb)
                return true; // same component → cycle
            parent[rv] = rnb;
        }
    }
    return false;
}

/**
 * Verify that `fvs` is a valid FVS for graph g.
 * Removes fvs vertices, then checks the remainder is a forest.
 */
static bool is_valid_fvs(const UndirectedGraph &g,
                         const std::vector<int> &fvs)
{
    UndirectedGraph tmp = g.copy();
    for (int v : fvs)
    {
        if (tmp.is_active(v))
        {
            std::vector<std::pair<int, int>> dummy;
            tmp.deactivate_full(v, dummy);
        }
    }
    return !tmp.has_cycle();
}

// ══════════════════════════════════════════════════════════════════════════════
//  ALGORITHM 1: Bounded Search Tree (BST)
// ══════════════════════════════════════════════════════════════════════════════

/**
 * Internal BST recursive function.
 *
 * Takes the graph BY VALUE (copy) because each branch explores a different
 * subgraph.  This is intentional: branching = "remove v, recurse".
 *
 * @param g        Current graph (copy — safe to modify)
 * @param k        Remaining budget (how many more vertices we may include)
 * @param fvs_out  Accumulates chosen FVS vertices for this branch
 * @return true if a valid FVS of total size ≤ original_k was found
 */
static bool bst_recurse(UndirectedGraph g, int k, std::vector<int> &fvs_out)
{
    // ── Step 1: Apply kernelization ──────────────────────────────────────────
    // Rules 0,1,2 may force some vertices into FVS (self-loops) or safely
    // remove others (degree ≤ 1, degree-2 contraction).
    std::vector<int> forced;
    if (!kernelize_undirected(g, forced, k))
    {
        // k went negative: forced inclusions exceeded budget → prune this branch
        return false;
    }

    // ── Step 2: Find a cycle ─────────────────────────────────────────────────
    std::vector<int> cycle = g.find_cycle();

    if (cycle.empty())
    {
        // No cycles remain → the forced vertices form a complete, valid FVS
        fvs_out.insert(fvs_out.end(), forced.begin(), forced.end());
        return true;
    }

    // ── Step 3: Budget check ──────────────────────────────────────────────────
    // There is still a cycle, but we have no budget left → infeasible
    if (k <= 0)
        return false;

    // ── Step 4: Branch on each vertex of the cycle ───────────────────────────
    //
    // At least one vertex of `cycle` MUST be in the FVS.
    // We try each one.  The first branch that succeeds wins.
    //
    // Optimisation: sort by degree descending.  Higher-degree vertices are
    // more likely to break many cycles at once, so we try them first.
    // This doesn't affect correctness, only speed (early termination).
    std::sort(cycle.begin(), cycle.end(), [&](int a, int b)
              { return g.degree(a) > g.degree(b); });

    for (int v : cycle)
    {
        if (!g.is_active(v))
            continue;

        // Branch: "include v in FVS"
        UndirectedGraph g_branch = g.copy();
        std::vector<std::pair<int, int>> removed_edges;
        g_branch.deactivate_full(v, removed_edges);

        std::vector<int> sub_fvs;
        if (bst_recurse(g_branch, k - 1, sub_fvs))
        {
            // This branch succeeded: commit all vertices
            fvs_out.insert(fvs_out.end(), forced.begin(), forced.end());
            fvs_out.push_back(v);
            fvs_out.insert(fvs_out.end(), sub_fvs.begin(), sub_fvs.end());
            return true;
        }
        // Branch failed → try next vertex in cycle
    }

    return false; // no branch worked within budget k
}

/**
 * Public BST solver.
 * Uses iterative deepening: try k = 0, 1, 2, … until success.
 * Returns the MINIMUM FVS.
 */
std::vector<int> solve_undirected_BST(int n,
                                      const std::vector<std::pair<int, int>> &edges)
{
    if (n == 0)
        return {};

    // Build graph
    UndirectedGraph g(n);
    for (auto &[u, v] : edges)
        if (u >= 0 && u < n && v >= 0 && v < n)
            g.add_edge(u, v);

    // Iterative deepening over budget k = 0, 1, 2, …
    for (int k = 0; k <= n; ++k)
    {
        std::vector<int> fvs;
        if (bst_recurse(g.copy(), k, fvs))
        {
            // Deduplicate (degree-2 contraction can theoretically produce dupes)
            std::sort(fvs.begin(), fvs.end());
            fvs.erase(std::unique(fvs.begin(), fvs.end()), fvs.end());
            return fvs;
        }
    }
    return {}; // unreachable: every graph has a trivial FVS of all vertices
}

// ══════════════════════════════════════════════════════════════════════════════
//  ALGORITHM 2: Iterative Compression (IC)
// ══════════════════════════════════════════════════════════════════════════════

/**
 * RESTRICTED BST: find an FVS of `g` using ONLY vertices not in `forbidden`,
 * with at most `budget` vertices.
 *
 * Why this is correct inside the compression step:
 *   - forbidden = X (the original FVS being compressed)
 *   - Vertices in Y = X\Z are "free" (in forest of G–F), so they can't be
 *     picked for the new FVS
 *   - We proved: since G[Y] is a forest, every cycle in G' = G–Z must pass
 *     through at least one vertex NOT in X (i.e., not in forbidden).
 *   - Therefore we can always find a non-forbidden vertex to branch on.
 *
 * @param g         Current graph (copy, safe to modify)
 * @param forbidden Set of vertices that may NOT be chosen for FVS
 * @param budget    Remaining budget
 * @param fvs_out   Accumulates chosen vertices
 * @return true if a valid restricted FVS of size ≤ budget was found
 */
static bool restricted_bst(UndirectedGraph g,
                           const std::unordered_set<int> &forbidden,
                           int budget,
                           std::vector<int> &fvs_out)
{
    // Apply kernelization: safe reductions even under the restriction.
    // Note: kernelization may force self-loop vertices even if they're in
    // forbidden — but a self-loop vertex MUST be in any FVS, which contradicts
    // it being in forbidden (forbidden = X = current FVS = already removed).
    // In practice, after removing Z from G, forbidden vertices have no self-loops.
    std::vector<int> forced;
    int k = budget;
    if (!kernelize_undirected(g, forced, k))
        return false;

    // Check: all forced vertices must not be forbidden
    for (int v : forced)
    {
        if (forbidden.count(v))
            return false; // can't include forbidden vertex
    }

    // Find a cycle in the reduced graph
    std::vector<int> cycle = g.find_cycle();

    if (cycle.empty())
    {
        // No cycles → forced vertices suffice
        fvs_out.insert(fvs_out.end(), forced.begin(), forced.end());
        return true;
    }

    if (k <= 0)
        return false;

    // Collect non-forbidden vertices in the cycle — these are our branching choices.
    // Key theorem: since G[Y] (Y = X\Z) is a forest, any cycle in G–Z passes
    // through at least one vertex from V\X = V\forbidden. So candidates is non-empty
    // as long as the graph has a cycle.
    std::vector<int> candidates;
    for (int v : cycle)
    {
        if (g.is_active(v) && !forbidden.count(v))
            candidates.push_back(v);
    }

    if (candidates.empty())
    {
        // All cycle vertices are forbidden — can't fix this cycle → FAIL
        return false;
    }

    // Branch on each candidate
    for (int v : candidates)
    {
        UndirectedGraph g_branch = g.copy();
        std::vector<std::pair<int, int>> dummy;
        g_branch.deactivate_full(v, dummy);

        std::vector<int> sub_fvs;
        if (restricted_bst(g_branch, forbidden, k - 1, sub_fvs))
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
 * COMPRESS(G, X, k):
 *   Given FVS X of size k+1, find an FVS of size ≤ k, or return empty.
 *
 *   Enumerate all 2^|X| subsets Z ⊆ X:
 *     Z = the X-vertices that GO INTO the new FVS F
 *     Y = X \ Z = the X-vertices that STAY IN THE GRAPH (must form a forest)
 *
 *   For each valid (Z, Y) pair:
 *     Call restricted_bst on G–Z to find F_W ⊆ V\X, |F_W| ≤ k–|Z|.
 *     Return Z ∪ F_W if found.
 */
static std::vector<int> compress(const UndirectedGraph &G,
                                 const std::vector<int> &X,
                                 int k)
{
    int sz = (int)X.size(); // = k+1

    // forbidden = X (the restricted BST may not pick from X)
    std::unordered_set<int> forbidden(X.begin(), X.end());

    // Enumerate all 2^sz subsets of X via bitmask
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

        // Budget check: Z alone must fit in k
        if ((int)Z.size() > k)
            continue;

        // ── Condition 1: G[Y] must be a forest ───────────────────────────
        // Y = X\Z will NOT be in the new FVS F.
        // Since G–F is a forest and Y ⊆ G–F, G[Y] must be a subgraph of a
        // forest → G[Y] must itself be a forest.
        if (induced_has_cycle(G, Y))
            continue;

        // ── Build G' = G – Z ──────────────────────────────────────────────
        // Remove the chosen Z-vertices from G.
        UndirectedGraph G_prime = G.copy();
        for (int v : Z)
        {
            if (G_prime.is_active(v))
            {
                std::vector<std::pair<int, int>> dummy;
                G_prime.deactivate_full(v, dummy);
            }
        }

        // ── Condition 2: Find F_W ⊆ V\X, |F_W| ≤ k–|Z| ─────────────────
        // F_W must be an FVS of G' restricted to V\X vertices.
        int budget = k - (int)Z.size();
        std::vector<int> F_W;

        if (restricted_bst(G_prime, forbidden, budget, F_W))
        {
            // Build candidate FVS = Z ∪ F_W
            std::vector<int> candidate = Z;
            candidate.insert(candidate.end(), F_W.begin(), F_W.end());

            // Final verification (defensive — should always pass)
            if (is_valid_fvs(G, candidate))
                return candidate;
        }
    }

    return {}; // FAIL: no valid (Z, F_W) pair found
}

/**
 * Public IC solver.
 *
 * Adds vertices one at a time (v_0, v_1, …, v_{n-1}).
 * After adding v_i, X ∪ {v_i} is a valid FVS for G_i.
 * Calls compress() to try to shrink back by 1.
 */
std::vector<int> solve_undirected_IC(int n,
                                     const std::vector<std::pair<int, int>> &edges)
{
    if (n == 0)
        return {};

    // Build the full graph (for compress() which needs the complete picture)
    // We also need to track which edges belong to G_i at each step.
    // For simplicity: build adjacency lists per vertex so we know what
    // edges to add when we "introduce" vertex i.
    std::vector<std::vector<int>> adj_full(n);
    for (auto &[u, v] : edges)
    {
        if (u >= 0 && u < n && v >= 0 && v < n)
        {
            adj_full[u].push_back(v);
            adj_full[v].push_back(u);
        }
    }

    // G_curr = induced subgraph on {v_0, …, v_i} built incrementally
    UndirectedGraph G_curr(n);
    // Start with all vertices inactive; activate as we go
    for (int v = 0; v < n; ++v)
        G_curr.active[v] = false;

    std::vector<int> X; // current FVS for G_curr

    for (int i = 0; i < n; ++i)
    {
        // ── Activate vertex i and add its edges to already-active vertices ──
        G_curr.active[i] = true;
        for (int nb : adj_full[i])
        {
            if (G_curr.is_active(nb))
                G_curr.add_edge(i, nb);
        }

        // ── X ∪ {i} is always a valid FVS for G_curr ─────────────────────
        // Proof: G_curr – (X ∪ {i}) = (G_{i-1} – X) which is a forest
        //        (X was FVS for G_{i-1}).
        X.push_back(i);

        // ── Try to compress X from size |X| down to |X|–1 ─────────────────
        int target_k = (int)X.size() - 1;

        // compress() only makes sense when |X| > 0 and the graph is non-trivial
        if (target_k >= 0 && (int)X.size() <= 20)
        {
            // Limit: 2^|X| enumeration is only feasible for small |X|.
            // For |X| > 20 we skip compression (FVS will be slightly larger).
            std::vector<int> compressed = compress(G_curr, X, target_k);
            if (!compressed.empty())
            {
                X = compressed;
            }
            // If compression fails, X stays at size target_k+1 (i.e., size |X|)
        }
    }

    // Deduplicate and validate
    std::sort(X.begin(), X.end());
    X.erase(std::unique(X.begin(), X.end()), X.end());

    // Keep only vertices still in the full graph that are actually needed
    // (some may have been kernelized away internally)
    // Re-verify and remove any unnecessary vertices
    bool improved = true;
    while (improved)
    {
        improved = false;
        for (int i = 0; i < (int)X.size(); ++i)
        {
            std::vector<int> candidate = X;
            candidate.erase(candidate.begin() + i);
            if (is_valid_fvs(G_curr, candidate))
            {
                X = candidate;
                improved = true;
                break;
            }
        }
    }

    return X;
}