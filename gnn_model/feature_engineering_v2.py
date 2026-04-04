"""
Advanced structural node features for GNN-KMA-2.

Feature layout (per node):
  0  degree/in-degree normalized
  1  clustering/out-degree normalized
  2  log-degree/min(in,out) normalized
  3  RWSE step 2 return probability
  4  RWSE step 3 return probability
  5  RWSE step 4 return probability
  6  RWSE step 5 return probability
  7  triangle participation count (normalized)
  8  4-cycle participation count (normalized)
  9  4-clique participation count (normalized)
 10  core number (normalized)

For directed graphs, motif/core features are computed on the undirected projection,
while directed degree features remain directional.
"""

from __future__ import annotations

import itertools
import math
from typing import Callable, List, Sequence, Tuple

import networkx as nx


# Exact 4-node motif enumeration is O(n^4); cap it to keep inference practical.
MAX_EXACT_4NODE_MOTIF_NODES = 90


def _rwse_return_probs_undirected(
    g: nx.Graph,
    max_step: int = 5,
    should_abort: Callable[[], bool] | None = None,
) -> List[List[float]] | None:
    n = g.number_of_nodes()
    if n == 0:
        return []

    # Sparse row-stochastic transition map keeps runtime manageable on larger kernels.
    p_rows: List[dict[int, float]] = [dict() for _ in range(n)]
    for i in range(n):
        if should_abort and should_abort():
            return None
        nbrs = list(g.neighbors(i))
        if not nbrs:
            continue
        w = 1.0 / len(nbrs)
        for j in nbrs:
            p_rows[i][j] = w

    cur_rows = [row.copy() for row in p_rows]
    diag_by_step = [[0.0 for _ in range(n)] for _ in range(max_step + 1)]
    for s in range(1, max_step + 1):
        if should_abort and should_abort():
            return None
        for i in range(n):
            if should_abort and should_abort():
                return None
            diag_by_step[s][i] = cur_rows[i].get(i, 0.0)
        if s < max_step:
            nxt_rows: List[dict[int, float]] = [dict() for _ in range(n)]
            for i in range(n):
                if should_abort and should_abort():
                    return None
                row = cur_rows[i]
                if not row:
                    continue
                dst = nxt_rows[i]
                for k, aik in row.items():
                    for j, pkj in p_rows[k].items():
                        dst[j] = dst.get(j, 0.0) + aik * pkj
            cur_rows = nxt_rows

    out: List[List[float]] = []
    for v in range(n):
        if should_abort and should_abort():
            return None
        out.append([diag_by_step[s][v] for s in range(2, max_step + 1)])
    return out


def _rwse_return_probs_directed(
    n: int,
    edges: Sequence[Tuple[int, int]],
    max_step: int = 5,
    should_abort: Callable[[], bool] | None = None,
) -> List[List[float]] | None:
    if n == 0:
        return []
    out_adj = [[] for _ in range(n)]
    for u, v in edges:
        if should_abort and should_abort():
            return None
        if 0 <= u < n and 0 <= v < n and u != v:
            out_adj[u].append(v)

    p_rows: List[dict[int, float]] = [dict() for _ in range(n)]
    for i in range(n):
        if should_abort and should_abort():
            return None
        deg = len(out_adj[i])
        if deg == 0:
            continue
        w = 1.0 / deg
        for j in out_adj[i]:
            p_rows[i][j] = p_rows[i].get(j, 0.0) + w

    cur_rows = [row.copy() for row in p_rows]
    diag_by_step = [[0.0 for _ in range(n)] for _ in range(max_step + 1)]
    for s in range(1, max_step + 1):
        if should_abort and should_abort():
            return None
        for i in range(n):
            if should_abort and should_abort():
                return None
            diag_by_step[s][i] = cur_rows[i].get(i, 0.0)
        if s < max_step:
            nxt_rows: List[dict[int, float]] = [dict() for _ in range(n)]
            for i in range(n):
                if should_abort and should_abort():
                    return None
                row = cur_rows[i]
                if not row:
                    continue
                dst = nxt_rows[i]
                for k, aik in row.items():
                    for j, pkj in p_rows[k].items():
                        dst[j] = dst.get(j, 0.0) + aik * pkj
            cur_rows = nxt_rows

    out: List[List[float]] = []
    for v in range(n):
        if should_abort and should_abort():
            return None
        out.append([diag_by_step[s][v] for s in range(2, max_step + 1)])
    return out


def _motif_counts_undirected(
    g: nx.Graph,
    should_abort: Callable[[], bool] | None = None,
) -> Tuple[List[int], List[int], List[int]] | None:
    n = g.number_of_nodes()
    tri = [0] * n
    four_cycle = [0] * n
    four_clique = [0] * n

    # Keep motif extraction bounded: skip exact motif counting on large kernels.
    if n > MAX_EXACT_4NODE_MOTIF_NODES:
        return tri, four_cycle, four_clique

    tri_map = nx.triangles(g)
    for v, c in tri_map.items():
        tri[v] = int(c)

    nodes = list(g.nodes())
    for a, b, c, d in itertools.combinations(nodes, 4):
        if should_abort and should_abort():
            return None
        verts = (a, b, c, d)
        sub = g.subgraph(verts)
        m = sub.number_of_edges()
        if m == 6:
            for v in verts:
                four_clique[v] += 1
            continue
        if m != 4:
            continue
        if all(sub.degree(v) == 2 for v in verts):
            for v in verts:
                four_cycle[v] += 1

    return tri, four_cycle, four_clique


def _normalize_counts(values: Sequence[int], n: int) -> List[float]:
    if n <= 1:
        return [0.0 for _ in values]
    scale = math.log(n + 1.0)
    return [math.log(v + 1.0) / scale for v in values]


def compute_node_features_undirected_v2(
    n: int,
    edges: Sequence[Tuple[int, int]],
    should_abort: Callable[[], bool] | None = None,
) -> List[List[float]] | None:
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edges)

    if should_abort and should_abort():
        return None

    degrees = dict(g.degree())
    clust = nx.clustering(g)
    core_map = nx.core_number(g) if g.number_of_edges() > 0 else {v: 0 for v in range(n)}
    rwse = _rwse_return_probs_undirected(g, max_step=5, should_abort=should_abort)
    if rwse is None:
        return None

    motif_counts = _motif_counts_undirected(g, should_abort=should_abort)
    if motif_counts is None:
        return None
    tri, c4, k4 = motif_counts

    tri_n = _normalize_counts(tri, n)
    c4_n = _normalize_counts(c4, n)
    k4_n = _normalize_counts(k4, n)
    core_max = max(core_map.values()) if core_map else 1

    feats: List[List[float]] = []
    for v in range(n):
        if should_abort and should_abort():
            return None
        deg = degrees.get(v, 0)
        core = core_map.get(v, 0)
        feats.append(
            [
                deg / max(n - 1, 1),
                clust.get(v, 0.0),
                math.log(deg + 1) / math.log(n + 1),
                *rwse[v],
                tri_n[v],
                c4_n[v],
                k4_n[v],
                core / max(core_max, 1),
            ]
        )
    return feats


def compute_node_features_directed_v2(
    n: int,
    edges: Sequence[Tuple[int, int]],
    should_abort: Callable[[], bool] | None = None,
) -> List[List[float]] | None:
    in_deg = [0] * n
    out_deg = [0] * n
    for u, v in edges:
        if should_abort and should_abort():
            return None
        if 0 <= u < n and 0 <= v < n and u != v:
            out_deg[u] += 1
            in_deg[v] += 1

    g_proj = nx.Graph()
    g_proj.add_nodes_from(range(n))
    for u, v in edges:
        if should_abort and should_abort():
            return None
        if 0 <= u < n and 0 <= v < n and u != v:
            a, b = (u, v) if u <= v else (v, u)
            g_proj.add_edge(a, b)

    core_map = nx.core_number(g_proj) if g_proj.number_of_edges() > 0 else {v: 0 for v in range(n)}
    motif_counts = _motif_counts_undirected(g_proj, should_abort=should_abort)
    if motif_counts is None:
        return None
    tri, c4, k4 = motif_counts

    rwse = _rwse_return_probs_directed(n, edges, max_step=5, should_abort=should_abort)
    if rwse is None:
        return None

    tri_n = _normalize_counts(tri, n)
    c4_n = _normalize_counts(c4, n)
    k4_n = _normalize_counts(k4, n)
    core_max = max(core_map.values()) if core_map else 1

    feats: List[List[float]] = []
    for v in range(n):
        if should_abort and should_abort():
            return None
        ind = in_deg[v]
        outd = out_deg[v]
        core = core_map.get(v, 0)
        feats.append(
            [
                ind / max(n - 1, 1),
                outd / max(n - 1, 1),
                min(ind, outd) / max(n - 1, 1),
                *rwse[v],
                tri_n[v],
                c4_n[v],
                k4_n[v],
                core / max(core_max, 1),
            ]
        )
    return feats
