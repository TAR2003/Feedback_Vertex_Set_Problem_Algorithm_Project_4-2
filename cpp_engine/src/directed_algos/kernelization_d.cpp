/**
 * @file kernelization_d.cpp
 * @brief Safe reduction rules for directed FVS (DFVS).
 *
 * ── Why directed kernelization is different ───────────────────────────────────
 *
 * In a directed graph, cycles require a directed path back to the start.
 * A vertex can only participate in a directed cycle if it has BOTH at least
 * one incoming edge AND at least one outgoing edge (otherwise no directed
 * path can traverse it in both directions).
 *
 * ── Rules (provably safe) ─────────────────────────────────────────────────────
 *
 * RULE D0 — Self-loop  [forced inclusion]
 *   Edge v → v means v is on a directed cycle of length 1.
 *   v MUST be in every DFVS.
 *   Action: add v to FVS, remove v, decrement k.
 *
 * RULE D1 — Source or Sink  [safe deletion]
 *   If in_degree(v) = 0: v has no incoming edges, so no directed cycle
 *     can reach v, so v is never the "end" of a back-edge in any directed
 *     cycle → v is not on any directed cycle → safe to delete.
 *   If out_degree(v) = 0: symmetric argument (v can't continue any cycle).
 *   Action: remove v. Do NOT add to FVS.
 *
 * ── What we deliberately skip ────────────────────────────────────────────────
 *
 * The "chain bypass" rules D2/D3 (bypass vertices with in_degree=1 or
 * out_degree=1 by adding shortcut edges) ARE valid in theory, but require
 * careful implementation to avoid creating 2-cycles or self-loops that
 * change the FVS structure.  We skip them to keep the code correct and
 * simple.  Rules D0 and D1 alone provide the essential reductions.
 *
 * The SCC-based rule (delete vertices in trivial SCCs) is correct in
 * principle, but running Tarjan's at every kernelization iteration is
 * expensive.  Instead, we integrate SCC decomposition once before the
 * BST/IC solvers start (see the solvers below).
 */
#include "directed_fvs.h"
#include <vector>
#include <utility>

/**
 * Apply directed reduction rules D0 and D1 exhaustively.
 *
 * @param g      Graph, modified in-place.
 * @param forced Accumulates vertices that MUST be in the DFVS (Rule D0).
 * @param k      Budget, decremented for Rule D0 inclusions.
 * @return false if k goes negative (infeasible).
 */
bool kernelize_directed(DirectedGraph &g,
                        std::vector<int> &forced,
                        int &k)
{
    bool changed = true;
    while (changed)
    {
        changed = false;
        for (int v = 0; v < g.n; ++v)
        {
            if (!g.active[v])
                continue;

            // Rule D0: self-loop
            if (g.out_adj[v].count(v))
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

            // Rule D1: source (in=0) or sink (out=0)
            if (g.in_degree(v) == 0 || g.out_degree(v) == 0)
            {
                std::vector<std::pair<int, int>> dummy;
                g.deactivate_full(v, dummy);
                changed = true;
                break;
            }
        }
    }
    return true;
}