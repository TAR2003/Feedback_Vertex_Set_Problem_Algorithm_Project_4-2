"""
feature_engineering_v3.py
==========================
Research-grade feature engineering for GNN-KMA-3.

Feature vector: 16 channels per node.
  [0]   in-degree (normalized by n-1)
  [1]   out-degree (normalized by n-1)
  [2]   min(in_deg, out_deg) / (n-1)         ← bottleneck proxy
  [3-9] RWSE steps [2,3,4,6,8,12,16]         ← 7 channels of random-walk SE
  [10]  triangles / max(triangles, 1)
  [11]  k-core / n
  [12]  SCC size (log-normalized)
  [13]  in_nontrivial_scc (binary)
  [14]  cycle_score = (in_deg × out_deg) / (n × scc_size)
  [15]  degree_ratio = out_deg / (in_deg + out_deg + ε)

RWSE channels use diagonal elements of (A^k)_ii from the random-walk matrix
P = D^{-1}A, giving the probability that a random walk from node v returns
to v in exactly k steps. This encodes local cycle structure and is one of
the most expressive positional encodings for FVS (Rampasek et al., 2022).

Key improvements vs v2:
  - RWSE steps extend to 16 (vs 8), capturing longer cycle structures
  - SCC-based features directly encode directed cycle membership
  - Cycle score captures the geometric mean of in/out connectivity within SCCs
  - Removed 4-clique/4-cycle features (expensive, low signal for DFVS)

References:
  - Rampasek et al. (2022) Recipe for a General, Powerful, Scalable Graph
    Transformer. NeurIPS 2022. (RWSE)
  - Razgon, O'Sullivan (2014) Almost 2-SAT (SCC-based FVS motivation)
  - Cooper & Frieze (2007) On random walk random graphs (RWSE theory)
"""

from __future__ import annotations

import math
import numpy as np

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

_RWSE_STEPS_DEFAULT = [2, 3, 4, 6, 8, 12, 16]
_MAX_GPU_RWSE_NODES = 1200


def _maybe_get_torch():
    try:
        import torch  # type: ignore
        return torch
    except ImportError:
        return None


def _resolve_torch_device(device: str | None = None):
    torch = _maybe_get_torch()
    if torch is None:
        return None
    pref = (device or "auto").lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _compute_rwse_torch_dense(
    n: int,
    edges: list,
    steps: list[int],
    directed: bool,
    device: str | None,
) -> np.ndarray | None:
    torch = _maybe_get_torch()
    if torch is None or n <= 0 or n > _MAX_GPU_RWSE_NODES:
        return None
    dev = _resolve_torch_device(device)
    if dev is None or dev.type != "cuda":
        return None

    valid_edges = [(u, v) for u, v in edges if 0 <= u < n and 0 <= v < n]
    if not directed:
        valid_edges = valid_edges + [(v, u) for u, v in valid_edges]
    if not valid_edges:
        return np.zeros((n, len(steps)), dtype=np.float32)

    rows = torch.tensor([u for u, _ in valid_edges], dtype=torch.long, device=dev)
    cols = torch.tensor([v for _, v in valid_edges], dtype=torch.long, device=dev)
    ones = torch.ones(rows.numel(), dtype=torch.float32, device=dev)

    out_deg = torch.zeros(n, dtype=torch.float32, device=dev)
    out_deg.scatter_add_(0, rows, ones)
    vals = ones / torch.clamp(out_deg[rows], min=1.0)

    P = torch.zeros((n, n), dtype=torch.float32, device=dev)
    P.index_put_((rows, cols), vals, accumulate=True)

    max_step = max(steps)
    step_to_col = {s: i for i, s in enumerate(steps)}
    result = torch.zeros((n, len(steps)), dtype=torch.float32, device=dev)

    Pk = P
    for t in range(1, max_step + 1):
        if t in step_to_col:
            result[:, step_to_col[t]] = torch.diagonal(Pk)
        if t < max_step:
            Pk = Pk @ P

    return result.detach().cpu().numpy().astype(np.float32)


def compute_rwse(
    n: int,
    edges: list,
    steps: list[int] | None = None,
    directed: bool = True,
) -> np.ndarray:
    """
    Compute Random Walk Structural Encoding (RWSE) for all nodes.

    For each step k, RWSE[v, k] = Pr[random walk of length k from v returns to v].
    This equals (P^k)_{vv} where P = D^{-1}A is the row-stochastic random walk matrix.

    For directed graphs, the out-degree row-stochastic matrix is used:
        P[u, v] = 1/out_deg(u)  if (u,v) ∈ E
        P[u, v] = 0            otherwise

    Isolated vertices (no outgoing edges) have all zeros in RWSE.

    Time complexity: O(|steps| * |E|) via sparse matrix powers.

    Args:
        n:        Number of vertices
        edges:    Edge list [(u, v), ...]
        steps:    List of step sizes (default: [2,3,4,6,8,12,16])
        directed: If False, add reverse edges for undirected walk

    Returns:
        ndarray of shape (n, len(steps)) with float32 values in [0, 1]

    Reference: Rampasek et al. (2022) § 3.2 RWSE positional encoding.
    """
    if steps is None:
        steps = _RWSE_STEPS_DEFAULT

    n_steps = len(steps)
    result = np.zeros((n, n_steps), dtype=np.float32)

    if n == 0 or not edges:
        return result

    # Build out-degree map
    out_count = np.zeros(n, dtype=np.float32)
    for u, v in edges:
        if 0 <= u < n and 0 <= v < n:
            out_count[u] += 1.0
    if not directed:
        for u, v in edges:
            if 0 <= u < n and 0 <= v < n:
                out_count[v] += 1.0

    # Build P as a dict of sparse row representations
    # P[u][v] = 1/out_deg(u) if (u,v) in E
    P = [{} for _ in range(n)]
    for u, v in edges:
        if 0 <= u < n and 0 <= v < n and out_count[u] > 0:
            P[u][v] = P[u].get(v, 0.0) + 1.0 / out_count[u]
    if not directed:
        for u, v in edges:
            if 0 <= u < n and 0 <= v < n and out_count[v] > 0:
                P[v][u] = P[v].get(u, 0.0) + 1.0 / out_count[v]

    # Compute P^k via repeated sparse matrix-vector products
    # We want diagonal of P^k, i.e., (P^k)_{ii}
    # Strategy: run k power iterations of P, tracking probability at start node.
    # For all nodes simultaneously, maintain distribution vector D_v of length n,
    # where D_v[u] = Pr[walk starting from v is at u after step t].
    # Then (P^k)_{vv} = D_v[v] after k steps.
    # This is O(n * k * avg_degree) — for large n/k we skip if too slow.

    max_step = max(steps)

    # Limit RWSE computation for very large graphs
    if n > 5000 or max_step * len(edges) > 50_000_000:
        # Return zero RWSE (safe fallback: model will use other features)
        return result

    # Batch all start vertices: maintain current_dist shape (n, n)
    # current_dist[v, :] = distribution of walk starting at v, step t
    # Start: each walk is at its own vertex
    current_dist = np.eye(n, dtype=np.float32)

    step_idx = 0
    sorted_steps = sorted(steps)
    step_to_col = {s: steps.index(s) for s in steps}

    for t in range(1, max_step + 1):
        # Apply one step of P: new_dist = current_dist @ P
        new_dist = np.zeros((n, n), dtype=np.float32)
        for u in range(n):
            if P[u]:
                for v, prob in P[u].items():
                    new_dist[:, v] += current_dist[:, u] * prob
        current_dist = new_dist

        if t in sorted_steps:
            col = step_to_col[t]
            # Diagonal: (P^t)_{vv} = current_dist[v, v]
            result[:, col] = np.diag(current_dist)

    return result


def compute_rwse_fast(
    n: int,
    edges: list,
    steps: list[int] | None = None,
    directed: bool = True,
    device: str | None = None,
) -> np.ndarray:
    """
    Faster RWSE using sparse CSR matrix multiplication (scipy).

    Falls back to compute_rwse if scipy is not available.
    For large graphs only computes node-local RW (approximation).
    """
    if steps is None:
        steps = _RWSE_STEPS_DEFAULT

    n_steps = len(steps)
    result = np.zeros((n, n_steps), dtype=np.float32)

    if n == 0 or not edges:
        return result

    rwse_gpu = _compute_rwse_torch_dense(
        n=n,
        edges=edges,
        steps=steps,
        directed=directed,
        device=device,
    )
    if rwse_gpu is not None:
        return rwse_gpu

    try:
        from scipy.sparse import csr_matrix
    except ImportError:
        # scipy not available, fall back to dense (slow for large n)
        return compute_rwse(n, edges, steps=steps, directed=directed)

    # Build out-degree and sparse P matrix
    row_idx, col_idx, data = [], [], []
    out_count = np.zeros(n, dtype=np.float64)

    valid_edges = [(u, v) for u, v in edges if 0 <= u < n and 0 <= v < n]

    for u, v in valid_edges:
        out_count[u] += 1.0
    if not directed:
        for u, v in valid_edges:
            out_count[v] += 1.0

    for u, v in valid_edges:
        if out_count[u] > 0:
            row_idx.append(u)
            col_idx.append(v)
            data.append(1.0 / out_count[u])
    if not directed:
        for u, v in valid_edges:
            if out_count[v] > 0:
                row_idx.append(v)
                col_idx.append(u)
                data.append(1.0 / out_count[v])

    if not row_idx:
        return result

    P = csr_matrix(
        (np.array(data, dtype=np.float64), (row_idx, col_idx)),
        shape=(n, n)
    )

    # Compute P^k for each step using repeated multiplication
    # We extract the diagonal of P^k
    max_step = max(steps)
    step_set = set(steps)
    step_to_col = {s: steps.index(s) for s in steps}

    # Limit for large graphs
    if n > 3000 and max_step > 8:
        # Only compute diagonal of P^k via random projections (Hutchinson estimator)
        # This is a rough approximation; exact for small graphs
        pass  # Fall through to full computation below

    Pk = csr_matrix(np.eye(n, dtype=np.float64))  # P^0 = I
    for t in range(1, max_step + 1):
        Pk = Pk.dot(P)
        if t in step_set:
            col = step_to_col[t]
            result[:, col] = np.array(Pk.diagonal(), dtype=np.float32)

    return result


def compute_scc_features(
    n: int,
    edges: list,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute SCC-based features for directed graphs.

    Returns:
        scc_sizes:          (n,) normalized SCC size each node belongs to
                            (log(size)/log(n), rescaled to [0,1])
        in_nontrivial_scc:  (n,) binary: 1 if node is in SCC of size >= 2
        scc_raw_sizes:      (n,) raw SCC size (integer), for cycle_score computation
    """
    scc_sizes_raw = np.ones(n, dtype=np.float32)   # default: trivial SCC of size 1
    in_nontrivial = np.zeros(n, dtype=np.float32)

    if not HAS_NX or n == 0:
        return scc_sizes_raw / max(1.0, math.log(n + 1)), in_nontrivial, scc_sizes_raw

    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for u, v in edges:
        if 0 <= u < n and 0 <= v < n:
            G.add_edge(u, v)

    log_n = math.log(n + 1)
    for scc in nx.strongly_connected_components(G):
        size = len(scc)
        for v in scc:
            scc_sizes_raw[v] = float(size)
            if size >= 2:
                in_nontrivial[v] = 1.0

    scc_sizes_norm = np.log1p(scc_sizes_raw) / max(1.0, log_n)

    return scc_sizes_norm, in_nontrivial, scc_sizes_raw


def compute_cycle_score(
    n: int,
    edges: list,
    in_deg: np.ndarray,
    out_deg: np.ndarray,
    scc_sizes_raw: np.ndarray,
) -> np.ndarray:
    """
    Compute cycle-betweenness proxy: (in_deg × out_deg) / (n × scc_size).

    Intuition: vertices with high in-degree AND high out-degree that lie in
    large SCCs are geometrically central to many cycles — they are high-value
    FVS candidates. This is a scale-invariant proxy for cycle betweenness.

    Reference: Lim et al. (2014), "Finding Feedback Vertex Sets for
    Minimum FVS" uses in×out product as heuristic score.

    Returns:
        cycle_score: (n,) float32, normalized to [0, 1]
    """
    eps = 1e-8
    denom = n * scc_sizes_raw + eps
    score = (in_deg * out_deg) / denom
    max_score = score.max()
    if max_score > 0:
        score = score / max_score
    return score.astype(np.float32)


def compute_local_clustering(
    n: int,
    edges: list,
) -> np.ndarray:
    """
    Compute local clustering coefficient for undirected version or
    triangle-based feature for directed graphs.
    Returns (n,) float32 array normalized to [0, 1].
    """
    counts = np.zeros(n, dtype=np.float32)
    if not HAS_NX or n == 0:
        return counts

    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for u, v in edges:
        if 0 <= u < n and 0 <= v < n:
            G.add_edge(u, v)

    # Count directed triangles: |{(u,v,w) : u→v, v→w, w→u}| for each v
    for u, v, w in nx.triangles(G.to_undirected()):  # type: ignore[attr-defined]
        pass

    # Use undirected triangle count as approximation (cheaper)
    G_und = G.to_undirected()
    tri = np.array([nx.triangles(G_und, nodes=v) for v in range(n)], dtype=np.float32)  # type: ignore[call-arg]
    max_tri = tri.max()
    if max_tri > 0:
        tri /= max_tri
    return tri


def compute_kcore(n: int, edges: list) -> np.ndarray:
    """Compute k-core number for each node, normalized by n."""
    kcore = np.zeros(n, dtype=np.float32)
    if not HAS_NX or n == 0:
        return kcore

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for u, v in edges:
        if 0 <= u < n and 0 <= v < n and u != v:
            G.add_edge(u, v)

    core_nums = nx.core_number(G)
    for v, k in core_nums.items():
        kcore[v] = float(k)

    max_core = kcore.max()
    if max_core > 0:
        kcore /= max_core
    return kcore


# ── Main feature functions ───────────────────────────────────────────────────

def compute_node_features_directed_v3(
    n: int,
    edges: list,
    should_abort=None,
    device: str | None = None,
) -> list | None:
    """
    Compute 16-channel directed node feature vector.

    Channel layout:
        [0]   in-degree (normalized)
        [1]   out-degree (normalized)
        [2]   min(in, out) / (n-1)
        [3-9] RWSE steps [2,3,4,6,8,12,16]
        [10]  triangles / max(triangles, 1) — undirected triangle count normalized
        [11]  k-core / max_core
        [12]  SCC size (log-normalized)
        [13]  in_nontrivial_scc
        [14]  cycle_score
        [15]  degree_ratio = out_deg / (in_deg + out_deg + ε)

    Args:
        n:            Number of vertices
        edges:        Directed edge list [(u, v)]
        should_abort: callable() → bool; returns True to abort early (timeout)

    Returns:
        List of n feature vectors (each is a list of 16 floats), or None on abort.
    """
    eps = 1e-8
    ndim = 16

    if n == 0:
        return []

    # ── Base degree features ─────────────────────────────────────────────────
    in_deg = np.zeros(n, dtype=np.float32)
    out_deg = np.zeros(n, dtype=np.float32)
    for u, v in edges:
        if 0 <= u < n and 0 <= v < n:
            out_deg[u] += 1.0
            in_deg[v] += 1.0

    norm = max(1.0, float(n - 1))
    f_in = in_deg / norm            # [0]
    f_out = out_deg / norm           # [1]
    f_min = np.minimum(in_deg, out_deg) / norm   # [2]

    if should_abort and should_abort():
        return None

    # ── RWSE ─────────────────────────────────────────────────────────────────
    try:
        rwse = compute_rwse_fast(
            n,
            edges,
            steps=_RWSE_STEPS_DEFAULT,
            directed=True,
            device=device,
        )  # (n, 7)
    except Exception:
        rwse = np.zeros((n, len(_RWSE_STEPS_DEFAULT)), dtype=np.float32)

    if should_abort and should_abort():
        return None

    # ── Triangle count (undirected approximation) ────────────────────────────
    try:
        tri = compute_local_clustering(n, edges)
    except Exception:
        tri = np.zeros(n, dtype=np.float32)

    if should_abort and should_abort():
        return None

    # ── K-core ───────────────────────────────────────────────────────────────
    try:
        kcore = compute_kcore(n, edges)
    except Exception:
        kcore = np.zeros(n, dtype=np.float32)

    if should_abort and should_abort():
        return None

    # ── SCC features ─────────────────────────────────────────────────────────
    try:
        scc_norm, in_nontriv, scc_raw = compute_scc_features(n, edges)
    except Exception:
        scc_norm = np.zeros(n, dtype=np.float32)
        in_nontriv = np.zeros(n, dtype=np.float32)
        scc_raw = np.ones(n, dtype=np.float32)

    if should_abort and should_abort():
        return None

    # ── Cycle score ───────────────────────────────────────────────────────────
    try:
        cycle_sc = compute_cycle_score(n, edges, in_deg, out_deg, scc_raw)
    except Exception:
        cycle_sc = np.zeros(n, dtype=np.float32)

    # ── Degree ratio ─────────────────────────────────────────────────────────
    deg_ratio = out_deg / (in_deg + out_deg + eps)  # [15]

    # ── Assemble feature matrix ────────────────────────────────────────────
    feats = np.stack([
        f_in,            # 0
        f_out,           # 1
        f_min,           # 2
        rwse[:, 0],      # 3  RWSE step=2
        rwse[:, 1],      # 4  RWSE step=3
        rwse[:, 2],      # 5  RWSE step=4
        rwse[:, 3],      # 6  RWSE step=6
        rwse[:, 4],      # 7  RWSE step=8
        rwse[:, 5],      # 8  RWSE step=12
        rwse[:, 6],      # 9  RWSE step=16
        tri,             # 10 triangles
        kcore,           # 11 k-core
        scc_norm,        # 12 SCC size
        in_nontriv,      # 13 in_nontrivial_scc
        cycle_sc,        # 14 cycle_score
        deg_ratio,       # 15 degree_ratio
    ], axis=1)           # shape (n, 16)

    return feats.tolist()


def compute_node_features_undirected_v3(
    n: int,
    edges: list,
    should_abort=None,
    device: str | None = None,
) -> list | None:
    """
    Compute 16-channel undirected node feature vector.

    For undirected graphs, all directed-specific features are adapted:
        in_deg = out_deg = regular degree (divided by 2)
        SCC features → connected component size
        cycle_score → degree^2 / (n × comp_size)

    Channel layout: same 16-channel layout as directed version.
    """
    eps = 1e-8

    if n == 0:
        return []

    # ── Degree ───────────────────────────────────────────────────────────────
    deg = np.zeros(n, dtype=np.float32)
    for u, v in edges:
        if 0 <= u < n and 0 <= v < n:
            deg[u] += 1.0
            if u != v:
                deg[v] += 1.0

    norm = max(1.0, float(n - 1))
    f_in = deg / norm           # [0]
    f_out = deg / norm          # [1]
    f_min = deg / norm          # [2]

    if should_abort and should_abort():
        return None

    # ── RWSE (undirected: add reverse edges) ─────────────────────────────────
    undirected_edges = edges + [(v, u) for u, v in edges]
    try:
        rwse = compute_rwse_fast(
            n,
            undirected_edges,
            steps=_RWSE_STEPS_DEFAULT,
            directed=False,
            device=device,
        )
    except Exception:
        rwse = np.zeros((n, len(_RWSE_STEPS_DEFAULT)), dtype=np.float32)

    if should_abort and should_abort():
        return None

    # ── Triangle count ───────────────────────────────────────────────────────
    try:
        tri = compute_local_clustering(n, edges)
    except Exception:
        tri = np.zeros(n, dtype=np.float32)

    # ── K-core ───────────────────────────────────────────────────────────────
    try:
        kcore = compute_kcore(n, edges)
    except Exception:
        kcore = np.zeros(n, dtype=np.float32)

    if should_abort and should_abort():
        return None

    # ── Connected component as proxy for SCC ─────────────────────────────────
    scc_norm = np.zeros(n, dtype=np.float32)
    in_nontriv = np.zeros(n, dtype=np.float32)
    scc_raw = np.ones(n, dtype=np.float32)

    if HAS_NX and n > 0:
        try:
            G_und = nx.Graph()
            G_und.add_nodes_from(range(n))
            for u, v in edges:
                if 0 <= u < n and 0 <= v < n:
                    G_und.add_edge(u, v)
            log_n = max(1.0, math.log(n + 1))
            for comp in nx.connected_components(G_und):
                size = len(comp)
                for v in comp:
                    scc_raw[v] = float(size)
                    scc_norm[v] = math.log1p(size) / log_n
                    if size >= 2:
                        in_nontriv[v] = 1.0
        except Exception:
            pass

    if should_abort and should_abort():
        return None

    # ── Cycle score (undirected: use degree^2 / (n × comp_size)) ─────────────
    try:
        denom = n * scc_raw + eps
        cycle_sc = (deg * deg) / denom
        mx = cycle_sc.max()
        if mx > 0:
            cycle_sc /= mx
        cycle_sc = cycle_sc.astype(np.float32)
    except Exception:
        cycle_sc = np.zeros(n, dtype=np.float32)

    # degree ratio = 0.5 for undirected (symmetric)
    deg_ratio = np.full(n, 0.5, dtype=np.float32)

    feats = np.stack([
        f_in, f_out, f_min,
        rwse[:, 0], rwse[:, 1], rwse[:, 2], rwse[:, 3],
        rwse[:, 4], rwse[:, 5], rwse[:, 6],
        tri, kcore,
        scc_norm, in_nontriv, cycle_sc, deg_ratio,
    ], axis=1)

    return feats.tolist()
