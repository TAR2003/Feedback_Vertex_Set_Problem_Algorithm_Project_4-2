/**
 * @file kernelization_u.cpp
 * @brief Kernelization reduction rules for undirected FVS.
 *
 * Kernelization shrinks the graph using polynomial-time rules that provably
 * preserve the optimal solution size.  After kernelization, the graph is
 * either trivially solved or small enough for the exponential solver.
 *
 * ── Rules implemented ────────────────────────────────────────────────────────
 *
 * Rule 0 (Self-Loop):
 *   If vertex v has a self-loop, v MUST be in any FVS.
 *   Action: Add v to FVS, remove v from graph.
 *
 * Rule 1 (Degree-0 / Degree-1):
 *   A vertex of degree 0 or 1 cannot be on any cycle.
 *   Action: Remove v from graph (do NOT add to FVS).
 *
 * Rule 2 (Degree-2 Bypass):
 *   If v has exactly degree 2 with neighbors {a, b}:
 *     - If {a, b} is already an edge → v is on a triangle; treat like degree-3
 *       (this case is handled by the solver branching, not here).
 *     - Otherwise: contracting v is safe: remove v, add edge {a, b}.
 *       Any FVS for the contracted graph plus possibly v gives a valid FVS.
 *   Action: Remove v, add edge {a, b}.  Bump the degree-2 counter.
 *
 * Rule 3 (High-Degree Lower Bound — only when budget k is known):
 *   If degree(v) > k, then v MUST be in any FVS of size ≤ k (because
 *   removing all other vertices still leaves v on k+1 paths).
 *   Action: Add v to FVS, remove v, decrement k.
 *
 * These rules are applied exhaustively (repeat until no rule fires) before
 * each branch step of the BST solver.
 */

#include "undirected_fvs.h"
#include <vector>
#include <set>
#include <algorithm>

// ─── Helper: apply Rule 0 + Rule 1 + Rule 2 exhaustively ─────────────────────

/**
 * Apply kernelization rules 0, 1, 2 exhaustively to graph g.
 * Appends forced FVS vertices to `fvs_partial`.
 * Does NOT use budget k; these rules are always safe.
 *
 * @param g           Graph to reduce (modified in-place)
 * @param fvs_partial Accumulates vertices forced into FVS
 * @param edge_record Records edges removed by Rule 2 for possible restoration
 */
void apply_undirected_kernelization(
    UndirectedGraph &g,
    std::vector<int> &fvs_partial)
{
    bool changed = true;

    while (changed)
    {
        changed = false;

        for (int v = 0; v < g.n; ++v)
        {
            if (!g.active[v])
                continue;

            // ── Rule 0: Self-loop ────────────────────────────────────────────
            // In our representation, self-loops appear as v ∈ adj[v].
            if (g.adj[v].count(v))
            {
                fvs_partial.push_back(v);
                std::vector<std::pair<int, int>> dummy;
                g.deactivate_full(v, dummy);
                changed = true;
                continue;
            }

            int deg = g.degree(v);

            // ── Rule 1: Degree 0 or 1 ────────────────────────────────────────
            if (deg <= 1)
            {
                std::vector<std::pair<int, int>> dummy;
                g.deactivate_full(v, dummy);
                changed = true;
                continue;
            }

            // ── Rule 2: Degree-2 bypass ──────────────────────────────────────
            if (deg == 2)
            {
                // Find the two active neighbors
                std::vector<int> nbrs;
                for (int nb : g.adj[v])
                {
                    if (g.active[nb])
                        nbrs.push_back(nb);
                }
                if (nbrs.size() != 2)
                    continue; // safety check

                int a = nbrs[0], b = nbrs[1];

                // If a and b are already connected, v is part of a triangle.
                // Do NOT contract; leave for the branching step.
                if (g.adj[a].count(b))
                    continue;

                // Safe to contract: remove v, add edge {a, b}
                std::vector<std::pair<int, int>> dummy;
                g.deactivate_full(v, dummy);
                g.add_edge(a, b);
                changed = true;
                continue;
            }
        }
    }
}

/**
 * Apply Rule 3 (high-degree forced inclusion) when budget k is available.
 * Returns the updated remaining budget.
 *
 * @param g           Graph (modified)
 * @param fvs_partial Accumulates FVS vertices
 * @param k           Remaining budget
 * @return            Updated k (may become negative if problem is infeasible)
 */
int apply_high_degree_rule(
    UndirectedGraph &g,
    std::vector<int> &fvs_partial,
    int k)
{
    bool changed = true;
    while (changed)
    {
        changed = false;
        for (int v = 0; v < g.n; ++v)
        {
            if (!g.active[v])
                continue;
            if (g.degree(v) > k)
            {
                fvs_partial.push_back(v);
                std::vector<std::pair<int, int>> dummy;
                g.deactivate_full(v, dummy);
                --k;
                changed = true;
                break; // restart loop (k changed)
            }
        }
    }
    return k;
}

/**
 * Full kernelization pass combining Rules 0-2 only.
 * Rule 3 (high-degree) is deliberately excluded: the naive "degree > k" threshold
 * is INCORRECT for FVS (it only holds under specific conditions and causes BST to
 * force-include safe vertices, returning super-optimal solutions).
 * Rules 0-2 are always safe reductions.
 *
 * @return Updated k (may be negative only if self-loop forced inclusions exceed k).
 */
int full_kernelization_u(
    UndirectedGraph &g,
    std::vector<int> &fvs_partial,
    int k)
{
    int before = static_cast<int>(fvs_partial.size());
    apply_undirected_kernelization(g, fvs_partial);
    int delta = static_cast<int>(fvs_partial.size()) - before;
    return k - delta;
}