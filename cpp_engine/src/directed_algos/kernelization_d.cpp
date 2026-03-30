/**
 * @file kernelization_d.cpp
 * @brief Kernelization reduction rules for directed FVS.
 *
 * ── Directed-Graph-Specific Rules ────────────────────────────────────────────
 *
 * Rule D0 (Self-loop):
 *   If vertex v has a self-loop (edge v → v), then v is on a trivial cycle
 *   of length 1 and MUST be in any DFVS.
 *   Action: Add v to FVS, remove v.
 *
 * Rule D1 (Source / Sink):
 *   If in_degree(v) == 0, v cannot be the LAST vertex on any directed cycle
 *   (nothing points to it).  Similarly, if out_degree(v) == 0, v is a sink
 *   and cannot start a directed cycle.
 *   Action: Remove v from graph (NOT added to FVS).
 *
 * Rule D2 (PI-node / IN-node elimination):
 *   A vertex v with in_degree == 1:
 *     Let p be v's unique predecessor.
 *     Any cycle through v must enter via p and exit to some successor of v.
 *     If p == v (already caught by D0), skip.
 *     Otherwise: "bypass" v by adding edge p → w for each w ∈ out_adj(v),
 *     then remove v.  This preserves all directed cycle lengths (or shrinks them).
 *
 * Rule D3 (PO-node / OUT-node elimination):
 *   Symmetric to D2 for out_degree == 1.
 *
 * Rule D4 (SCC Decomposition):
 *   Any vertex NOT inside a non-trivial SCC (or a self-loop) is NOT on any
 *   directed cycle and can be safely removed.
 *   Action: Remove all vertices in trivial SCCs (size 1, no self-loop).
 *
 * Rules are applied exhaustively before each BST branch step.
 */

#include "directed_fvs.h"
#include <vector>
#include <set>
#include <algorithm>

// ─── Rule D0 + D1 + D2 + D3 (exhaustive) ────────────────────────────────────

void apply_directed_kernelization(
    DirectedGraph &g,
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

            // ── Rule D0: Self-loop ───────────────────────────────────────────
            if (g.out_adj[v].count(v))
            {
                fvs_partial.push_back(v);
                std::vector<std::pair<int, int>> dummy;
                g.deactivate_full(v, dummy);
                changed = true;
                continue;
            }

            int in_deg = g.in_degree(v);
            int out_deg = g.out_degree(v);

            // ── Rule D1: Source or Sink ──────────────────────────────────────
            // A source (in=0) cannot be on any directed cycle (nothing leads to it).
            // A sink (out=0) cannot be on any directed cycle (it leads nowhere).
            if (in_deg == 0 || out_deg == 0)
            {
                std::vector<std::pair<int, int>> dummy;
                g.deactivate_full(v, dummy);
                changed = true;
                continue;
            }

            // ── Rule D2 & D3: Bypass IN-degree 1 or OUT-degree 1 ───────────
            if (in_deg == 1 || out_deg == 1)
            {
                std::vector<int> predecessors;
                std::vector<int> successors;

                for (int u = 0; u < g.n; ++u)
                {
                    if (g.active[u] && g.out_adj[u].count(v))
                        predecessors.push_back(u);
                }
                for (int w : g.out_adj[v])
                {
                    if (g.active[w])
                        successors.push_back(w);
                }

                for (int p : predecessors)
                {
                    for (int s : successors)
                    {
                        // Keep p == s as a self-loop; D0 will force-include it.
                        g.add_edge(p, s);
                    }
                }

                std::vector<std::pair<int, int>> dummy;
                g.deactivate_full(v, dummy);
                changed = true;
                continue;
            }
        }
    }
}

// ─── Rule D4: SCC-based pruning ───────────────────────────────────────────────

/**
 * Remove all vertices that are NOT in a non-trivial SCC.
 * After Tarjan's decomposition, keep only vertices in SCCs of size > 1
 * (or size 1 with a self-loop, but D0 already handled self-loops).
 */
void apply_scc_reduction(DirectedGraph &g)
{
    auto sccs = g.find_SCCs();

    // Mark vertices that are in trivial SCCs
    std::vector<bool> in_nontrivial(g.n, false);
    for (auto &scc : sccs)
    {
        if (scc.size() > 1)
        {
            for (int v : scc)
                in_nontrivial[v] = true;
        }
        // A size-1 SCC with self-loop was already handled by D0.
    }

    for (int v = 0; v < g.n; ++v)
    {
        if (g.active[v] && !in_nontrivial[v])
        {
            std::vector<std::pair<int, int>> dummy;
            g.deactivate_full(v, dummy);
        }
    }
}

/**
 * Full directed kernelization pipeline.
 * Apply D0-D3 exhaustively, then D4 (SCC reduction), then D0-D3 again.
 * Repeat until stable.
 *
 * Also applies the high-degree rule: if max(in_deg, out_deg) > k, force-include v.
 *
 * @return Updated budget k (may be negative if infeasible).
 */
int full_kernelization_d(
    DirectedGraph &g,
    std::vector<int> &fvs_partial,
    int k)
{
    bool changed = true;
    while (changed)
    {
        changed = false;

        int old_size = static_cast<int>(fvs_partial.size());

        apply_directed_kernelization(g, fvs_partial);
        apply_scc_reduction(g);

        // Account for newly forced vertices
        int delta = static_cast<int>(fvs_partial.size()) - old_size;
        k -= delta;
        if (delta > 0)
            changed = true;

        // High-degree rule for directed: if min(in,out) > k, must include v
        for (int v = 0; v < g.n; ++v)
        {
            if (!g.active[v])
                continue;
            int min_deg = std::min(g.in_degree(v), g.out_degree(v));
            if (min_deg > k)
            {
                fvs_partial.push_back(v);
                std::vector<std::pair<int, int>> dummy;
                g.deactivate_full(v, dummy);
                --k;
                changed = true;
            }
        }
    }
    return k;
}