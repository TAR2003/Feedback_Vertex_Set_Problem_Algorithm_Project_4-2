/**
 * @file kernelization_u.cpp
 * @brief Safe polynomial-time reduction rules for undirected FVS.
 *
 * ── Rules implemented (and WHY each is correct) ──────────────────────────────
 *
 * RULE 0 — Self-loop  [forced inclusion]
 *   v has edge (v,v) → v is on a 1-cycle → MUST be in any FVS.
 *   Action: add v to FVS, remove v, decrement k.
 *
 * RULE 1 — Degree ≤ 1  [safe deletion]
 *   A vertex of degree ≤ 1 cannot be on any cycle (cycles require degree ≥ 2).
 *   Action: remove v. Do NOT add to FVS.
 *
 * RULE 2 — Degree-2 contraction  [graph simplification]
 *   If deg(v) = 2, neighbors a,b, and edge {a,b} does NOT exist:
 *     v can only be on a cycle of the form a–v–b–…–a.
 *     Any such cycle still exists in G' = G–{v} + edge{a,b}.
 *     Therefore min_FVS(G) = min_FVS(G'), so contracting is safe.
 *   If edge {a,b} EXISTS: v is on triangle a–v–b–a; skip, leave for branching.
 *   Action: remove v, add edge {a,b}.
 */
#include "undirected_fvs.h"
#include <vector>
#include <utility>

/**
 * Apply reduction rules 0,1,2 exhaustively.
 * Forced FVS vertices appended to `forced`, budget k decremented.
 * Returns false if k goes negative (infeasible at this budget).
 */
bool kernelize_undirected(UndirectedGraph &g,
                          std::vector<int> &forced,
                          int &k,
                          const std::unordered_set<int> *forbidden)
{
    bool changed = true;
    while (changed)
    {
        changed = false;
        for (int v = 0; v < g.n; ++v)
        {
            if (!g.active[v])
                continue;

            // Rule 0: self-loop
            if (g.adj[v].count(v))
            {
                forced.push_back(v);
                --k;
                std::vector<std::pair<int, int>> dummy;
                g.deactivate_full(v, dummy);
                changed = true;
                if (k < 0)
                    return false;
                break;
            }

            // Collect active neighbors
            std::vector<int> nbrs;
            for (int nb : g.adj[v])
                if (g.active[nb])
                    nbrs.push_back(nb);
            int deg = (int)nbrs.size();

            // Rule 1: degree 0 or 1
            if (deg <= 1)
            {
                std::vector<std::pair<int, int>> dummy;
                g.deactivate_full(v, dummy);
                changed = true;
                break;
            }

            // Rule 2: degree-2 contraction (only when a-b not adjacent)
            if (deg == 2)
            {
                // In restricted BST, keep degree-2 vertices to avoid creating
                // cycles entirely inside forbidden vertices after contraction.
                if (forbidden)
                    continue;

                int a = nbrs[0], b = nbrs[1];
                if (!g.adj[a].count(b))
                {
                    std::vector<std::pair<int, int>> dummy;
                    g.deactivate_full(v, dummy);
                    if (g.active[a] && g.active[b])
                        g.add_edge(a, b);
                    changed = true;
                    break;
                }
            }
        }
    }
    return true;
}