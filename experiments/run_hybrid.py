#!/usr/bin/env python3
"""
run_hybrid.py
=============
Phase 3: GNN-Guided Kernelized Memetic Algorithm (GNN-KMA mode).

How it works:
  1. Load the trained GNN model (undirected GCN or directed DiGCN).
  2. Run inference: the GNN outputs per-vertex probabilities P(v ∈ FVS).
  3. Vertices with P > threshold are flagged as likely FVS members.
  4. The kernel graph is solved with KMA (kernelization + MA refinement).
  5. Forced kernel vertices and KMA output are merged for the final solution.

This GNN-KMA approach combines:
  - GNN's pattern recognition (learned from thousands of solved instances)
    - KMA's combinatorial optimization power

Without GNN weights (fallback):
  If gnn_model/weights/ does not contain trained weights, the script
    automatically falls back to KMA-only behavior — no crash, no error.

Usage:
  python experiments/run_hybrid.py --graph <file> --type undirected
  python experiments/run_hybrid.py --graph <file> --type directed
  python experiments/run_hybrid.py --graph <file> --type undirected --pop 100 --gens 500
"""

import sys
sys.setrecursionlimit(20000)
import argparse
import time
import math
import random
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
# Try platform-specific build directories first, then legacy build/
for candidate in ("build-linux", "build-macos", "build-win", "build"):
    sys.path.insert(0, str(PROJECT_ROOT / "cpp_engine" / candidate))
# Try experiments first (where the .so file is compiled) - insert last so it's first in path
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from gnn_model.feature_engineering_v2 import (
    compute_node_features_directed_v2,
    compute_node_features_undirected_v2,
)

try:
    import cpp_engine
except ImportError as e:
    HAS_CPP_ENGINE = False
    cpp_engine = None
else:
    HAS_CPP_ENGINE = True

# Try importing PyTorch — graceful fallback if not installed
# NOTE: PyTorch is imported lazily in functions that need it to avoid long startup times
HAS_TORCH = None  # None = not determined yet, will check on first use

# Try importing GNN models — graceful fallback (lazy loaded to avoid slow startup)
HAS_GNN = None  # None = not determined yet, will check on first use
UndirectedFVSNet = None
DirectedFVSNet = None
HAS_GNN_V2 = None
UndirectedFVSNetV2 = None
DirectedFVSNetV2 = None
HAS_GNN_V3 = None
UndirectedFVSNetV3 = None
DirectedFVSNetV3 = None
HAS_PYG_LOADER = None
NeighborLoader = None
PygData = None

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False


# ═══════════════════════════════════════════════════════════════════════════════
#  Lazy PyTorch import (to avoid slow startup)
# ═══════════════════════════════════════════════════════════════════════════════

def get_torch():
    """Lazy import of PyTorch - only imported when GNN functions are called."""
    global torch, HAS_TORCH
    if HAS_TORCH is None:
        try:
            import torch as torch_module
            torch = torch_module
            HAS_TORCH = True
        except ImportError:
            HAS_TORCH = False
            torch = None
    return torch if HAS_TORCH else None


def get_gnn_models():
    """Lazy import of GNN models - only imported when needed."""
    global HAS_GNN, UndirectedFVSNet, DirectedFVSNet
    if HAS_GNN is None:
        try:
            from gnn_model.model_undirected import UndirectedFVSNet as UNet
            from gnn_model.model_directed import DirectedFVSNet as DNet
            UndirectedFVSNet = UNet
            DirectedFVSNet = DNet
            HAS_GNN = True
        except (ImportError, ModuleNotFoundError):
            HAS_GNN = False
            UndirectedFVSNet = None
            DirectedFVSNet = None
    return HAS_GNN, UndirectedFVSNet, DirectedFVSNet


def get_gnn_models_v2():
    """Lazy import of v2 GNN models for GNN-KMA-2."""
    global HAS_GNN_V2, UndirectedFVSNetV2, DirectedFVSNetV2
    if HAS_GNN_V2 is None:
        try:
            from gnn_model.model_undirected_v2 import UndirectedFVSNetV2 as UNetV2
            from gnn_model.model_directed_v2 import DirectedFVSNetV2 as DNetV2
            UndirectedFVSNetV2 = UNetV2
            DirectedFVSNetV2 = DNetV2
            HAS_GNN_V2 = True
        except (ImportError, ModuleNotFoundError):
            HAS_GNN_V2 = False
            UndirectedFVSNetV2 = None
            DirectedFVSNetV2 = None
    return HAS_GNN_V2, UndirectedFVSNetV2, DirectedFVSNetV2


def get_gnn_models_v3():
    """Lazy import of v3 GNN models (GAT + residual) for GNN-KMA-3."""
    global HAS_GNN_V3, UndirectedFVSNetV3, DirectedFVSNetV3
    if HAS_GNN_V3 is None:
        try:
            from gnn_model.model_directed_v3 import DirectedFVSNetV3 as DNetV3
            from gnn_model.model_directed_v3 import UndirectedFVSNetV3 as UNetV3
            DirectedFVSNetV3 = DNetV3
            UndirectedFVSNetV3 = UNetV3
            HAS_GNN_V3 = True
        except (ImportError, ModuleNotFoundError):
            HAS_GNN_V3 = False
            DirectedFVSNetV3 = None
            UndirectedFVSNetV3 = None
    return HAS_GNN_V3, UndirectedFVSNetV3, DirectedFVSNetV3


def get_pyg_loader():
    """Lazy import for PyG NeighborLoader/Data used by mini-batch inference."""
    global HAS_PYG_LOADER, NeighborLoader, PygData
    if HAS_PYG_LOADER is None:
        try:
            from torch_geometric.loader import NeighborLoader as _NeighborLoader
            from torch_geometric.data import Data as _PygData
            NeighborLoader = _NeighborLoader
            PygData = _PygData
            HAS_PYG_LOADER = True
        except ImportError:
            HAS_PYG_LOADER = False
            NeighborLoader = None
            PygData = None
    return HAS_PYG_LOADER, NeighborLoader, PygData


def _gnn_phase_timed_out(gnn_start_time, gnn_timeout):
    if gnn_start_time is None or gnn_timeout is None:
        return False
    return (time.time() - gnn_start_time) > float(gnn_timeout)


def _warn_gnn_timeout(gnn_timeout):
    print(f"GNN phase timed out (>{gnn_timeout} seconds). Bypassing GNN and proceeding with pure KMA.")


def _neighbor_hops_for_model(model, default_hops=3):
    layers = 0
    for i in range(1, 11):
        if hasattr(model, f"conv{i}"):
            layers += 1
    return [10] * (layers if layers > 0 else default_hops)


def _extract_positive_probs(sigmoid_out):
    """Return per-node positive-class probabilities from sigmoid outputs."""
    if sigmoid_out.dim() == 1:
        return sigmoid_out
    if sigmoid_out.dim() == 2:
        if sigmoid_out.size(1) == 1:
            return sigmoid_out[:, 0]
        return sigmoid_out[:, 1]
    return sigmoid_out.reshape(sigmoid_out.size(0), -1)[:, 0]


# ═══════════════════════════════════════════════════════════════════════════════
#  Kernelization Helpers (for GNN-on-kernel GNN-KMA mode)
# ═══════════════════════════════════════════════════════════════════════════════

def kernelize_undirected_graph(n, edges):
    """
    Apply undirected kernelization rules (self-loop, degree-0/1, degree-2 bypass)
    and return a compact kernel graph with index mapping.

    Returns:
      (kernel_n, kernel_edges, forced, new_to_old)
    """
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if 0 <= u < n and 0 <= v < n:
            adj[u].add(v)
            adj[v].add(u)

    active = [True] * n
    forced = set()

    def deactivate(v):
        if not active[v]:
            return
        for nb in list(adj[v]):
            adj[nb].discard(v)
        adj[v].clear()
        active[v] = False

    changed = True
    while changed:
        changed = False
        for v in range(n):
            if not active[v]:
                continue

            if v in adj[v]:
                forced.add(v)
                deactivate(v)
                changed = True
                continue

            nbrs = [nb for nb in adj[v] if active[nb]]
            deg = len(nbrs)

            if deg <= 1:
                deactivate(v)
                changed = True
                continue

            if deg == 2:
                a, b = nbrs
                if b not in adj[a]:
                    deactivate(v)
                    if active[a] and active[b]:
                        adj[a].add(b)
                        adj[b].add(a)
                    changed = True

    new_to_old = [v for v in range(n) if active[v]]
    old_to_new = {old: new for new, old in enumerate(new_to_old)}

    kernel_edges = set()
    for u in new_to_old:
        for v in adj[u]:
            if not active[v]:
                continue
            nu = old_to_new[u]
            nv = old_to_new[v]
            if nu == nv:
                continue
            if nu > nv:
                nu, nv = nv, nu
            kernel_edges.add((nu, nv))

    return len(new_to_old), sorted(kernel_edges), sorted(forced), new_to_old


def kernelize_directed_graph(n, edges):
    """
    Apply directed kernelization rules (D0-D4 style) and return a compact kernel
    graph with index mapping.

    Returns:
      (kernel_n, kernel_edges, forced, new_to_old)
    """
    # Tarjan SCC below is recursive; large/deep directed instances can exceed
    # Python's default recursion limit (~1000) without this safeguard.
    needed_limit = max(2000, 4 * n + 100)
    if sys.getrecursionlimit() < needed_limit:
        sys.setrecursionlimit(needed_limit)

    out_adj = [set() for _ in range(n)]
    in_adj = [set() for _ in range(n)]
    for u, v in edges:
        if 0 <= u < n and 0 <= v < n:
            out_adj[u].add(v)
            in_adj[v].add(u)

    active = [True] * n
    forced = set()

    def deactivate(v):
        if not active[v]:
            return
        for p in list(in_adj[v]):
            out_adj[p].discard(v)
        for s in list(out_adj[v]):
            in_adj[s].discard(v)
        in_adj[v].clear()
        out_adj[v].clear()
        active[v] = False

    def nontrivial_scc_vertices():
        index = 0
        indices = [-1] * n
        lowlink = [0] * n
        onstack = [False] * n
        stack = []
        keep = set()

        def strongconnect(v):
            nonlocal index
            indices[v] = index
            lowlink[v] = index
            index += 1
            stack.append(v)
            onstack[v] = True

            for w in out_adj[v]:
                if not active[w]:
                    continue
                if indices[w] == -1:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif onstack[w]:
                    lowlink[v] = min(lowlink[v], indices[w])

            if lowlink[v] == indices[v]:
                scc = []
                while True:
                    w = stack.pop()
                    onstack[w] = False
                    scc.append(w)
                    if w == v:
                        break
                if len(scc) > 1:
                    keep.update(scc)
                elif scc and scc[0] in out_adj[scc[0]]:
                    keep.add(scc[0])

        for v in range(n):
            if active[v] and indices[v] == -1:
                strongconnect(v)
        return keep

    changed = True
    while changed:
        changed = False

        inner_changed = True
        while inner_changed:
            inner_changed = False
            for v in range(n):
                if not active[v]:
                    continue

                if v in out_adj[v]:
                    forced.add(v)
                    deactivate(v)
                    changed = True
                    inner_changed = True
                    continue

                in_deg = len(in_adj[v])
                out_deg = len(out_adj[v])

                if in_deg == 0 or out_deg == 0:
                    deactivate(v)
                    changed = True
                    inner_changed = True
                    continue

                if in_deg == 1 or out_deg == 1:
                    preds = list(in_adj[v])
                    succs = list(out_adj[v])
                    for p in preds:
                        if not active[p]:
                            continue
                        for s in succs:
                            if not active[s]:
                                continue
                            out_adj[p].add(s)
                            in_adj[s].add(p)
                    deactivate(v)
                    changed = True
                    inner_changed = True

        keep = nontrivial_scc_vertices()
        for v in range(n):
            if active[v] and v not in keep:
                deactivate(v)
                changed = True

    new_to_old = [v for v in range(n) if active[v]]
    old_to_new = {old: new for new, old in enumerate(new_to_old)}

    kernel_edges = set()
    for u in new_to_old:
        for v in out_adj[u]:
            if not active[v]:
                continue
            kernel_edges.add((old_to_new[u], old_to_new[v]))

    return len(new_to_old), sorted(kernel_edges), sorted(forced), new_to_old


# ═══════════════════════════════════════════════════════════════════════════════
#  Feature Extraction (mirrors gnn_model/dataset_gen.py)
# ═══════════════════════════════════════════════════════════════════════════════

def get_undirected_features(n, edges, gnn_start_time=None, gnn_timeout=None):
    """
    Compute per-node features for undirected graph.
    Returns list of [degree_norm, clustering_coeff, log_degree_norm].
    """
    if HAS_NX:
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edges)
        degs  = dict(G.degree())
        clust = nx.clustering(G)
    else:
        # Manual degree computation fallback
        degs  = {v: 0 for v in range(n)}
        for u, v in edges:
            degs[u] += 1; degs[v] += 1
        clust = {v: 0.0 for v in range(n)}

    features = []
    for v in range(n):
        if _gnn_phase_timed_out(gnn_start_time, gnn_timeout):
            return None
        d = degs.get(v, 0)
        features.append([
            d / max(n - 1, 1),
            clust.get(v, 0.0),
            math.log(d + 1) / math.log(n + 1)
        ])
    return features


def get_directed_features(n, edges, gnn_start_time=None, gnn_timeout=None):
    """
    Compute per-node features for directed graph.
    Returns list of [in_deg_norm, out_deg_norm, min_deg_norm].
    """
    in_deg  = [0] * n
    out_deg = [0] * n
    for u, v in edges:
        out_deg[u] += 1
        in_deg[v]  += 1

    features = []
    for v in range(n):
        if _gnn_phase_timed_out(gnn_start_time, gnn_timeout):
            return None
        ind  = in_deg[v]
        outd = out_deg[v]
        features.append([
            ind  / max(n - 1, 1),
            outd / max(n - 1, 1),
            min(ind, outd) / max(n - 1, 1)
        ])
    return features


def get_undirected_features_v2(n, edges, gnn_start_time=None, gnn_timeout=None):
    """Advanced undirected structural features used by GNN-KMA-2."""
    if not HAS_NX:
        return get_undirected_features(
            n,
            edges,
            gnn_start_time=gnn_start_time,
            gnn_timeout=gnn_timeout,
        )
    return compute_node_features_undirected_v2(
        n,
        edges,
        should_abort=lambda: _gnn_phase_timed_out(gnn_start_time, gnn_timeout),
    )


def get_directed_features_v2(n, edges, gnn_start_time=None, gnn_timeout=None):
    """Advanced directed structural features used by GNN-KMA-2."""
    if not HAS_NX:
        return get_directed_features(
            n,
            edges,
            gnn_start_time=gnn_start_time,
            gnn_timeout=gnn_timeout,
        )
    return compute_node_features_directed_v2(
        n,
        edges,
        should_abort=lambda: _gnn_phase_timed_out(gnn_start_time, gnn_timeout),
    )


def get_directed_features_v3(n, edges, gnn_start_time=None, gnn_timeout=None):
    """16-channel directed structural features for GNN-KMA-3 (SCC + RWSE steps 2-16)."""
    try:
        from gnn_model.feature_engineering_v3 import compute_node_features_directed_v3
        return compute_node_features_directed_v3(
            n,
            edges,
            should_abort=lambda: _gnn_phase_timed_out(gnn_start_time, gnn_timeout),
        )
    except (ImportError, ModuleNotFoundError):
        # Fallback to v2 if v3 not yet generated
        return get_directed_features_v2(n, edges, gnn_start_time, gnn_timeout)


def get_undirected_features_v3(n, edges, gnn_start_time=None, gnn_timeout=None):
    """16-channel undirected structural features for GNN-KMA-3."""
    try:
        from gnn_model.feature_engineering_v3 import compute_node_features_undirected_v3
        return compute_node_features_undirected_v3(
            n,
            edges,
            should_abort=lambda: _gnn_phase_timed_out(gnn_start_time, gnn_timeout),
        )
    except (ImportError, ModuleNotFoundError):
        return get_undirected_features_v2(n, edges, gnn_start_time, gnn_timeout)


def make_edge_index(edges, n, bidirected=False):
    """
    Convert edge list to PyTorch edge_index tensor.
    bidirected=True adds reverse edges (for undirected graphs).
    """
    torch = get_torch()
    if torch is None:
        return None
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)
    ei = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    if bidirected:
        ei = torch.cat([ei, ei.flip(0)], dim=1)
    return ei


def _extract_state_dict(ckpt_obj):
    """Return a plain state_dict from common checkpoint formats."""
    if isinstance(ckpt_obj, dict):
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
        if "model_state_dict" in ckpt_obj and isinstance(ckpt_obj["model_state_dict"], dict):
            return ckpt_obj["model_state_dict"]
    return ckpt_obj


def _infer_hidden_dim(state_dict, directed=False):
    """
    Infer model hidden dimension from checkpoint tensor shapes.
    Returns None if inference is not possible.
    """
    if not isinstance(state_dict, dict):
        return None

    # v3 checkpoints expose hidden dim in the input projection.
    key = "input_proj.weight"
    if key in state_dict and hasattr(state_dict[key], "shape") and len(state_dict[key].shape) >= 1:
        return int(state_dict[key].shape[0])

    # Directed v3 also carries 3*hidden -> hidden fusion projections.
    key = "fusion_projs.0.weight"
    if key in state_dict and hasattr(state_dict[key], "shape") and len(state_dict[key].shape) >= 1:
        return int(state_dict[key].shape[0])

    # Directed checkpoints store a direct bias tensor on conv1.
    if directed and "conv1.bias" in state_dict:
        return int(state_dict["conv1.bias"].shape[0])

    # Undirected SAGE checkpoints expose lin_l weights with leading hidden dim.
    key = "conv1.lin_l.weight"
    if key in state_dict:
        return int(state_dict[key].shape[0])

    # Fallback for any model that has bn1 parameters.
    if "bn1.weight" in state_dict:
        return int(state_dict["bn1.weight"].shape[0])

    return None


def _load_model_with_checkpoint(model_cls, weights_path, directed=False, hidden_dim_override=None):
    """
    Build model with compatible hidden_dim and load checkpoint.
    """
    torch = get_torch()
    if torch is None:
        return None, None
    
    ckpt = torch.load(weights_path, map_location="cpu")
    state_dict = _extract_state_dict(ckpt)

    inferred_hidden = _infer_hidden_dim(state_dict, directed=directed)
    hidden_dim = hidden_dim_override if hidden_dim_override is not None else inferred_hidden

    if hidden_dim is not None:
        model = model_cls(hidden_dim=hidden_dim)
    else:
        model = model_cls()

    model.load_state_dict(state_dict)
    return model, hidden_dim


def _pick_gnn_candidates_from_probs(
    positive_probs,
    threshold=0.65,
    min_fraction=0.005,
    max_fraction=0.08,
):
    """
    Precision-first GNN candidate selection for hard-fixing in GNN-KMA.

    Strategy:
      1) Select vertices with probability >= threshold (default 0.65).
      2) Safety cap: never hard-fix more than max_fraction of kernel.
      3) CRITICALLY: if nothing exceeds threshold, return EMPTY SET →
         falls through to pure KMA. It is ALWAYS better to use no GNN
         hints than to hard-fix uncertain vertices.

    Unlike the legacy approach, there is NO topk-fallback for hard-fixing.
    The topk-fallback silently included uncertain vertices and was the
    primary cause of GNN-KMA performing worse than pure KMA.

    Reference: Ben-Baruch et al. (2021) Asymmetric Loss — FP cost >> FN cost
               when false-positive predictions are permanently locked in.

    Returns:
      (candidate_set, mode_string)
    """
    torch = get_torch()
    if torch is None:
        return set(), "unavailable"

    n = int(positive_probs.numel())
    if n == 0:
        return set(), "empty-graph"

    max_k = max(1, int(math.ceil(n * max(0.0, max_fraction))))
    min_k = max(1, int(math.ceil(n * max(0.0, min_fraction))))

    selected = (positive_probs >= threshold).nonzero(as_tuple=True)[0]
    mode = "high_conf"

    if selected.numel() > max_k:
        # Sort by confidence descending, cap at max_k
        selected = torch.topk(positive_probs, max_k).indices
        mode = "high_conf_capped"
    elif selected.numel() < min_k:
        # DO NOT use topk-fallback for hard-fixing.
        # Return empty set → fall through to pure KMA for this graph.
        return set(), "no_fix_insufficient_confidence"

    return set(selected.tolist()), mode


# ═══════════════════════════════════════════════════════════════════════════════
#  GNN Inference
# ═══════════════════════════════════════════════════════════════════════════════

def run_gnn_undirected(n, edges, threshold=0.2, hidden_dim=None, gnn_timeout=60):
    """
    Run undirected GNN. Returns set of predicted FVS vertex indices.
    Returns None if GNN is unavailable.
    """
    torch = get_torch()
    has_gnn, UNet, _ = get_gnn_models()
    has_loader, LoaderCls, DataCls = get_pyg_loader()
    weights_path = PROJECT_ROOT / "gnn_model" / "weights" / "undirected_fvs_gcn.pt"

    if not has_gnn or torch is None:
        print("  [GNN] PyTorch/GNN not available. Skipping GNN step.")
        return None

    if not has_loader:
        print("  [GNN] torch_geometric NeighborLoader not available. Skipping GNN step.")
        return None

    if not weights_path.exists():
        print(f"  [GNN] Weights not found at {weights_path}. Skipping GNN step.")
        print("        Run: python gnn_model/train.py --type undirected")
        return None

    try:
        gnn_start_time = time.time()
        model, used_hidden = _load_model_with_checkpoint(
            UNet,
            weights_path,
            directed=False,
            hidden_dim_override=hidden_dim,
        )
        model.eval()

        feats = get_undirected_features(
            n,
            edges,
            gnn_start_time=gnn_start_time,
            gnn_timeout=gnn_timeout,
        )
        if feats is None:
            _warn_gnn_timeout(gnn_timeout)
            return set()

        if _gnn_phase_timed_out(gnn_start_time, gnn_timeout):
            _warn_gnn_timeout(gnn_timeout)
            return set()

        x = torch.tensor(feats, dtype=torch.float)
        ei = make_edge_index(edges, n, bidirected=True)

        data = DataCls(x=x, edge_index=ei)
        neighbor_loader = LoaderCls(
            data,
            input_nodes=torch.arange(n),
            num_neighbors=_neighbor_hops_for_model(model, default_hops=3),
            batch_size=2048,
            shuffle=False,
            directed=False,
        )

        batch_probs = []
        with torch.no_grad():
            for batch in neighbor_loader:
                if _gnn_phase_timed_out(gnn_start_time, gnn_timeout):
                    _warn_gnn_timeout(gnn_timeout)
                    return set()

                out = model(batch.x, batch.edge_index)
                probs = torch.sigmoid(out)
                target_probs = _extract_positive_probs(probs[:batch.batch_size])
                batch_probs.append(target_probs.cpu())

        if not batch_probs:
            return set()

        pos_probs = torch.cat(batch_probs, dim=0)
        if pos_probs.numel() < n:
            print("  [GNN] Incomplete NeighborLoader predictions; skipping GNN hints.")
            return set()
        if pos_probs.numel() > n:
            pos_probs = pos_probs[:n]

        gnn_candidates, selection_mode = _pick_gnn_candidates_from_probs(
            pos_probs,
            threshold=threshold,
        )
        hidden_note = f", hidden={used_hidden}" if used_hidden is not None else ""
        print(
            f"  [GNN] Predicted {len(gnn_candidates)} / {n} vertices as FVS candidates "
            f"(threshold={threshold}, mode={selection_mode}{hidden_note})"
        )
        return gnn_candidates

    except Exception as ex:
        print(f"  [GNN] Error during inference: {ex}. Skipping GNN step.")
        return None


def run_gnn_directed(n, edges, threshold=0.2, hidden_dim=None, gnn_timeout=60):
    """
    Run directed GNN (DiGCN). Returns set of predicted DFVS vertex indices.
    Returns None if GNN is unavailable.
    """
    torch = get_torch()
    has_gnn, _, DNet = get_gnn_models()
    has_loader, LoaderCls, DataCls = get_pyg_loader()
    weights_path = PROJECT_ROOT / "gnn_model" / "weights" / "directed_fvs_gcn.pt"

    if not has_gnn or torch is None:
        print("  [GNN] PyTorch/GNN not available. Skipping GNN step.")
        return None

    if not has_loader:
        print("  [GNN] torch_geometric NeighborLoader not available. Skipping GNN step.")
        return None

    if not weights_path.exists():
        print(f"  [GNN] Weights not found at {weights_path}. Skipping GNN step.")
        print("        Run: python gnn_model/train.py --type directed")
        return None

    try:
        gnn_start_time = time.time()
        model, used_hidden = _load_model_with_checkpoint(
            DNet,
            weights_path,
            directed=True,
            hidden_dim_override=hidden_dim,
        )
        model.eval()

        feats = get_directed_features(
            n,
            edges,
            gnn_start_time=gnn_start_time,
            gnn_timeout=gnn_timeout,
        )
        if feats is None:
            _warn_gnn_timeout(gnn_timeout)
            return set()

        if _gnn_phase_timed_out(gnn_start_time, gnn_timeout):
            _warn_gnn_timeout(gnn_timeout)
            return set()

        x = torch.tensor(feats, dtype=torch.float)
        ei = make_edge_index(edges, n, bidirected=False)

        data = DataCls(x=x, edge_index=ei)
        neighbor_loader = LoaderCls(
            data,
            input_nodes=torch.arange(n),
            num_neighbors=_neighbor_hops_for_model(model, default_hops=3),
            batch_size=2048,
            shuffle=False,
            directed=True,
        )

        batch_probs = []
        with torch.no_grad():
            for batch in neighbor_loader:
                if _gnn_phase_timed_out(gnn_start_time, gnn_timeout):
                    _warn_gnn_timeout(gnn_timeout)
                    return set()

                out = model(batch.x, batch.edge_index)
                probs = torch.sigmoid(out)
                target_probs = _extract_positive_probs(probs[:batch.batch_size])
                batch_probs.append(target_probs.cpu())

        if not batch_probs:
            return set()

        pos_probs = torch.cat(batch_probs, dim=0)
        if pos_probs.numel() < n:
            print("  [GNN] Incomplete NeighborLoader predictions; skipping GNN hints.")
            return set()
        if pos_probs.numel() > n:
            pos_probs = pos_probs[:n]

        gnn_candidates, selection_mode = _pick_gnn_candidates_from_probs(
            pos_probs,
            threshold=threshold,
        )
        hidden_note = f", hidden={used_hidden}" if used_hidden is not None else ""
        print(
            f"  [GNN] Predicted {len(gnn_candidates)} / {n} vertices as DFVS candidates "
            f"(threshold={threshold}, mode={selection_mode}{hidden_note})"
        )
        return gnn_candidates

    except Exception as ex:
        print(f"  [GNN] Error during inference: {ex}. Skipping GNN step.")
        return None


def run_gnn_undirected_v2(n, edges, threshold=0.2, hidden_dim=None, gnn_timeout=60):
    """Run undirected GNN-KMA-2 model with advanced structural features."""
    torch = get_torch()
    has_gnn_v2, UNetV2, _ = get_gnn_models_v2()
    has_loader, LoaderCls, DataCls = get_pyg_loader()
    weights_path = PROJECT_ROOT / "gnn_model" / "weights" / "undirected_fvs_gcn_v2.pt"

    if not has_gnn_v2 or torch is None:
        print("  [GNN-2] PyTorch/GNN not available. Skipping GNN step.")
        return None

    if not has_loader:
        print("  [GNN-2] torch_geometric NeighborLoader not available. Skipping GNN step.")
        return None

    if not weights_path.exists():
        print(f"  [GNN-2] Weights not found at {weights_path}. Skipping GNN step.")
        print("          Run: python gnn_model/train.py --type undirected --variant v2")
        return None

    try:
        gnn_start_time = time.time()
        model, used_hidden = _load_model_with_checkpoint(
            UNetV2,
            weights_path,
            directed=False,
            hidden_dim_override=hidden_dim,
        )
        model.eval()

        feats = get_undirected_features_v2(
            n,
            edges,
            gnn_start_time=gnn_start_time,
            gnn_timeout=gnn_timeout,
        )
        if feats is None:
            _warn_gnn_timeout(gnn_timeout)
            return set()

        if _gnn_phase_timed_out(gnn_start_time, gnn_timeout):
            _warn_gnn_timeout(gnn_timeout)
            return set()

        x = torch.tensor(feats, dtype=torch.float)
        ei = make_edge_index(edges, n, bidirected=True)

        data = DataCls(x=x, edge_index=ei)
        neighbor_loader = LoaderCls(
            data,
            input_nodes=torch.arange(n),
            num_neighbors=_neighbor_hops_for_model(model, default_hops=3),
            batch_size=2048,
            shuffle=False,
            directed=False,
        )

        batch_probs = []
        with torch.no_grad():
            for batch in neighbor_loader:
                if _gnn_phase_timed_out(gnn_start_time, gnn_timeout):
                    _warn_gnn_timeout(gnn_timeout)
                    return set()

                out = model(batch.x, batch.edge_index)
                probs = torch.sigmoid(out)
                target_probs = _extract_positive_probs(probs[:batch.batch_size])
                batch_probs.append(target_probs.cpu())

        if not batch_probs:
            return set()

        pos_probs = torch.cat(batch_probs, dim=0)
        if pos_probs.numel() < n:
            print("  [GNN-2] Incomplete NeighborLoader predictions; skipping GNN hints.")
            return set()
        if pos_probs.numel() > n:
            pos_probs = pos_probs[:n]

        gnn_candidates, selection_mode = _pick_gnn_candidates_from_probs(
            pos_probs,
            threshold=threshold,
        )
        hidden_note = f", hidden={used_hidden}" if used_hidden is not None else ""
        print(
            f"  [GNN-2] Predicted {len(gnn_candidates)} / {n} vertices as FVS candidates "
            f"(threshold={threshold}, mode={selection_mode}{hidden_note})"
        )
        return gnn_candidates
    except Exception as ex:
        print(f"  [GNN-2] Error during inference: {ex}. Skipping GNN step.")
        return None


def run_gnn_directed_v2(n, edges, threshold=0.2, hidden_dim=None, gnn_timeout=60):
    """Run directed GNN-KMA-2 model with advanced structural features."""
    torch = get_torch()
    has_gnn_v2, _, DNetV2 = get_gnn_models_v2()
    has_loader, LoaderCls, DataCls = get_pyg_loader()
    weights_path = PROJECT_ROOT / "gnn_model" / "weights" / "directed_fvs_gcn_v2.pt"

    if not has_gnn_v2 or torch is None:
        print("  [GNN-2] PyTorch/GNN not available. Skipping GNN step.")
        return None

    if not has_loader:
        print("  [GNN-2] torch_geometric NeighborLoader not available. Skipping GNN step.")
        return None

    if not weights_path.exists():
        print(f"  [GNN-2] Weights not found at {weights_path}. Skipping GNN step.")
        print("          Run: python gnn_model/train.py --type directed --variant v2")
        return None

    try:
        gnn_start_time = time.time()
        model, used_hidden = _load_model_with_checkpoint(
            DNetV2,
            weights_path,
            directed=True,
            hidden_dim_override=hidden_dim,
        )
        model.eval()

        feats = get_directed_features_v2(
            n,
            edges,
            gnn_start_time=gnn_start_time,
            gnn_timeout=gnn_timeout,
        )
        if feats is None:
            _warn_gnn_timeout(gnn_timeout)
            return set()

        if _gnn_phase_timed_out(gnn_start_time, gnn_timeout):
            _warn_gnn_timeout(gnn_timeout)
            return set()

        x = torch.tensor(feats, dtype=torch.float)
        ei = make_edge_index(edges, n, bidirected=False)

        data = DataCls(x=x, edge_index=ei)
        neighbor_loader = LoaderCls(
            data,
            input_nodes=torch.arange(n),
            num_neighbors=_neighbor_hops_for_model(model, default_hops=3),
            batch_size=2048,
            shuffle=False,
            directed=True,
        )

        batch_probs = []
        with torch.no_grad():
            for batch in neighbor_loader:
                if _gnn_phase_timed_out(gnn_start_time, gnn_timeout):
                    _warn_gnn_timeout(gnn_timeout)
                    return set()

                out = model(batch.x, batch.edge_index)
                probs = torch.sigmoid(out)
                target_probs = _extract_positive_probs(probs[:batch.batch_size])
                batch_probs.append(target_probs.cpu())

        if not batch_probs:
            return set()

        pos_probs = torch.cat(batch_probs, dim=0)
        if pos_probs.numel() < n:
            print("  [GNN-2] Incomplete NeighborLoader predictions; skipping GNN hints.")
            return set()
        if pos_probs.numel() > n:
            pos_probs = pos_probs[:n]

        gnn_candidates, selection_mode = _pick_gnn_candidates_from_probs(
            pos_probs,
            threshold=threshold,
        )
        hidden_note = f", hidden={used_hidden}" if used_hidden is not None else ""
        print(
            f"  [GNN-2] Predicted {len(gnn_candidates)} / {n} vertices as DFVS candidates "
            f"(threshold={threshold}, mode={selection_mode}{hidden_note})"
        )
        return gnn_candidates
    except Exception as ex:
        print(f"  [GNN-2] Error during inference: {ex}. Skipping GNN step.")
        return None


def _run_gnn_full_inference(
    model, feat_fn, n, edges, bidirected, label, gnn_timeout=60
):
    """
    Shared GNN inference engine returning raw float probs (numpy ndarray or None).

    Args:
        model: loaded GNN model (eval mode)
        feat_fn: callable(n, edges, gnn_start_time, gnn_timeout) -> feature list
        n: number of vertices in the graph
        edges: edge list
        bidirected: whether to add reverse edges (undirected graphs)
        label: log prefix e.g. "[GNN]", "[GNN-2]", "[GNN-3]"
        gnn_timeout: max seconds for GNN phase

    Returns:
        numpy.ndarray of shape (n,) with float32 probs, or None on failure.
    """
    import numpy as np
    torch = get_torch()
    has_loader, LoaderCls, DataCls = get_pyg_loader()
    if torch is None or not has_loader:
        return None

    try:
        gnn_start_time = time.time()
        feats = feat_fn(n, edges, gnn_start_time=gnn_start_time, gnn_timeout=gnn_timeout)
        if feats is None:
            _warn_gnn_timeout(gnn_timeout)
            return None

        if _gnn_phase_timed_out(gnn_start_time, gnn_timeout):
            _warn_gnn_timeout(gnn_timeout)
            return None

        x = torch.tensor(feats, dtype=torch.float)
        ei = make_edge_index(edges, n, bidirected=bidirected)

        data = DataCls(x=x, edge_index=ei)
        neighbor_loader = LoaderCls(
            data,
            input_nodes=torch.arange(n),
            num_neighbors=_neighbor_hops_for_model(model, default_hops=3),
            batch_size=2048,
            shuffle=False,
            directed=(not bidirected),
        )

        batch_probs = []
        with torch.no_grad():
            for batch in neighbor_loader:
                if _gnn_phase_timed_out(gnn_start_time, gnn_timeout):
                    _warn_gnn_timeout(gnn_timeout)
                    return None
                out = model(batch.x, batch.edge_index)
                probs = torch.sigmoid(out)
                target_probs = _extract_positive_probs(probs[:batch.batch_size])
                batch_probs.append(target_probs.cpu())

        if not batch_probs:
            return None

        pos_probs = torch.cat(batch_probs, dim=0)
        if pos_probs.numel() < n:
            print(f"  {label} Incomplete NeighborLoader predictions; skipping GNN hints.")
            return None
        if pos_probs.numel() > n:
            pos_probs = pos_probs[:n]

        return pos_probs.numpy().astype("float32")

    except Exception as ex:
        print(f"  {label} Error during inference: {ex}. Skipping GNN step.")
        return None


def run_gnn_directed_probs(n, edges, hidden_dim=None, gnn_timeout=60):
    """
    Run directed GNN (v1) and return raw per-vertex FVS probabilities.

    Returns:
        numpy.ndarray of shape (n,) with float32 values in [0, 1], or None.

    Reference: Soft-hint coupling design — GNN probabilities guide KMA
        without permanently locking in false positives.
        See Part 1 of the GNN-KMA research-grade overhaul.
    """
    torch = get_torch()
    has_gnn, _, DNet = get_gnn_models()
    weights_path = PROJECT_ROOT / "gnn_model" / "weights" / "directed_fvs_gcn.pt"

    if not has_gnn or torch is None:
        return None
    if not weights_path.exists():
        return None

    model, _ = _load_model_with_checkpoint(
        DNet, weights_path, directed=True, hidden_dim_override=hidden_dim
    )
    model.eval()
    return _run_gnn_full_inference(
        model, get_directed_features, n, edges,
        bidirected=False, label="[GNN]", gnn_timeout=gnn_timeout
    )


def run_gnn_directed_v2_probs(n, edges, hidden_dim=None, gnn_timeout=60):
    """
    Run directed GNN v2 and return raw per-vertex FVS probabilities.

    Returns:
        numpy.ndarray of shape (n,) with float32 values in [0, 1], or None.
    """
    torch = get_torch()
    has_gnn_v2, _, DNetV2 = get_gnn_models_v2()
    weights_path = PROJECT_ROOT / "gnn_model" / "weights" / "directed_fvs_gcn_v2.pt"

    if not has_gnn_v2 or torch is None:
        return None
    if not weights_path.exists():
        return None

    model, _ = _load_model_with_checkpoint(
        DNetV2, weights_path, directed=True, hidden_dim_override=hidden_dim
    )
    model.eval()
    return _run_gnn_full_inference(
        model, get_directed_features_v2, n, edges,
        bidirected=False, label="[GNN-2]", gnn_timeout=gnn_timeout
    )


def run_gnn_directed_v3_probs(n, edges, hidden_dim=None, gnn_timeout=60):
    """
    Run directed GNN v3 (GAT + residual, 16-channel features) and return
    raw per-vertex FVS probabilities.

    Returns:
        numpy.ndarray of shape (n,) with float32 values in [0, 1], or None.

    Reference: DirectedFVSNetV3 — Veličković et al. (2018) GAT +
        He et al. (2016) Deep Residual Learning.
    """
    torch = get_torch()
    has_gnn_v3, _, DNetV3 = get_gnn_models_v3()
    weights_path = PROJECT_ROOT / "gnn_model" / "weights" / "directed_fvs_gcn_v3.pt"

    if not has_gnn_v3 or torch is None:
        return None
    if not weights_path.exists():
        print(f"  [GNN-3] Weights not found at {weights_path}. Skipping GNN step.")
        print("          Run: python gnn_model/train.py --type directed --v3")
        return None

    try:
        model, _ = _load_model_with_checkpoint(
            DNetV3, weights_path, directed=True, hidden_dim_override=hidden_dim
        )
        model.eval()
        return _run_gnn_full_inference(
            model, get_directed_features_v3, n, edges,
            bidirected=False, label="[GNN-3]", gnn_timeout=gnn_timeout
        )
    except Exception as ex:
        print(f"  [GNN-3] Failed to load or run v3 model: {ex}")
        return None


def run_gnn_undirected_probs(n, edges, hidden_dim=None, gnn_timeout=60):
    """
    Run undirected GNN (v1) and return raw per-vertex FVS probabilities.

    Returns:
        numpy.ndarray of shape (n,) with float32 values in [0, 1], or None.
    """
    torch = get_torch()
    has_gnn, UNet, _ = get_gnn_models()
    weights_path = PROJECT_ROOT / "gnn_model" / "weights" / "undirected_fvs_gcn.pt"

    if not has_gnn or torch is None:
        return None
    if not weights_path.exists():
        return None

    model, _ = _load_model_with_checkpoint(
        UNet, weights_path, directed=False, hidden_dim_override=hidden_dim
    )
    model.eval()
    return _run_gnn_full_inference(
        model, get_undirected_features, n, edges,
        bidirected=True, label="[GNN]", gnn_timeout=gnn_timeout
    )


def run_gnn_undirected_v2_probs(n, edges, hidden_dim=None, gnn_timeout=60):
    """
    Run undirected GNN v2 and return raw per-vertex FVS probabilities.

    Returns:
        numpy.ndarray of shape (n,) with float32 values in [0, 1], or None.
    """
    torch = get_torch()
    has_gnn_v2, UNetV2, _ = get_gnn_models_v2()
    weights_path = PROJECT_ROOT / "gnn_model" / "weights" / "undirected_fvs_gcn_v2.pt"

    if not has_gnn_v2 or torch is None:
        return None
    if not weights_path.exists():
        return None

    model, _ = _load_model_with_checkpoint(
        UNetV2, weights_path, directed=False, hidden_dim_override=hidden_dim
    )
    model.eval()
    return _run_gnn_full_inference(
        model, get_undirected_features_v2, n, edges,
        bidirected=True, label="[GNN-2]", gnn_timeout=gnn_timeout
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  KMA / GNN-KMA Solvers
# ═══════════════════════════════════════════════════════════════════════════════

def _stage_metrics(kernelization_ms=0.0, gnn_candidate_ms=0.0, ma_ms=0.0):
    return {
        "kernelization_ms": float(kernelization_ms),
        "gnn_candidate_ms": float(gnn_candidate_ms),
        "ma_ms": float(ma_ms),
    }


def _maybe_with_metrics(solution, metrics, return_diagnostics):
    if return_diagnostics:
        return solution, metrics
    return solution


def kma_solve_undirected(
    n,
    edges,
    pop_size=20,
    max_gens=100,
    max_time_seconds=600,
    early_stop=20,
    return_diagnostics=False,
):
    """KMA: kernelize -> MA refinement, with timeout applied only to MA stage."""
    if not HAS_CPP_ENGINE:
        raise RuntimeError("cpp_engine not available. Please compile it first.")

    t0 = time.perf_counter()
    k_n, k_edges, forced, k_new_to_old = kernelize_undirected_graph(n, edges)
    kernel_ms = (time.perf_counter() - t0) * 1000.0

    if k_n == 0:
        return _maybe_with_metrics(
            forced,
            _stage_metrics(kernelization_ms=kernel_ms),
            return_diagnostics,
        )

    ma_start = time.perf_counter()
    if hasattr(cpp_engine, "solve_undirected_KMA"):
        kernel_fvs = cpp_engine.solve_undirected_KMA(
            k_n,
            k_edges,
            pop_size,
            max_gens,
            early_stop,
            max_time_seconds,
        )
    elif hasattr(cpp_engine, "solve_undirected_KME"):
        kernel_fvs = cpp_engine.solve_undirected_KME(
            k_n,
            k_edges,
            pop_size,
            max_gens,
            early_stop,
            max_time_seconds,
        )
    else:
        kernel_fvs = cpp_engine.solve_undirected_MA(
            k_n,
            k_edges,
            pop_size,
            max_gens,
            early_stop,
            max_time_seconds,
        )
    ma_ms = (time.perf_counter() - ma_start) * 1000.0

    mapped = [k_new_to_old[v] for v in kernel_fvs if 0 <= v < len(k_new_to_old)]
    solution = sorted(set(forced).union(mapped))
    return _maybe_with_metrics(
        solution,
        _stage_metrics(kernelization_ms=kernel_ms, ma_ms=ma_ms),
        return_diagnostics,
    )


def kma_solve_directed(
    n,
    edges,
    pop_size=20,
    max_gens=100,
    max_time_seconds=600,
    early_stop=20,
    return_diagnostics=False,
):
    """KMA: kernelize -> MA refinement, with timeout applied only to MA stage."""
    if not HAS_CPP_ENGINE:
        raise RuntimeError("cpp_engine not available. Please compile it first.")

    t0 = time.perf_counter()
    k_n, k_edges, forced, k_new_to_old = kernelize_directed_graph(n, edges)
    kernel_ms = (time.perf_counter() - t0) * 1000.0

    if k_n == 0:
        return _maybe_with_metrics(
            forced,
            _stage_metrics(kernelization_ms=kernel_ms),
            return_diagnostics,
        )

    ma_start = time.perf_counter()
    if hasattr(cpp_engine, "solve_directed_KMA"):
        kernel_fvs = cpp_engine.solve_directed_KMA(
            k_n,
            k_edges,
            pop_size,
            max_gens,
            early_stop,
            max_time_seconds,
        )
    elif hasattr(cpp_engine, "solve_directed_KME"):
        kernel_fvs = cpp_engine.solve_directed_KME(
            k_n,
            k_edges,
            pop_size,
            max_gens,
            early_stop,
            max_time_seconds,
        )
    else:
        kernel_fvs = cpp_engine.solve_directed_MA(
            k_n,
            k_edges,
            pop_size,
            max_gens,
            early_stop,
            max_time_seconds,
        )
    ma_ms = (time.perf_counter() - ma_start) * 1000.0

    mapped = [k_new_to_old[v] for v in kernel_fvs if 0 <= v < len(k_new_to_old)]
    solution = sorted(set(forced).union(mapped))
    return _maybe_with_metrics(
        solution,
        _stage_metrics(kernelization_ms=kernel_ms, ma_ms=ma_ms),
        return_diagnostics,
    )



def _kma_run_directed(n, edges, pop_size, max_gens, early_stop, max_time_seconds):
    """Dispatch to available directed KMA solver (KMA > KME > MA fallback)."""
    if hasattr(cpp_engine, "solve_directed_KMA"):
        return cpp_engine.solve_directed_KMA(n, edges, pop_size, max_gens, early_stop, max_time_seconds)
    elif hasattr(cpp_engine, "solve_directed_KME"):
        return cpp_engine.solve_directed_KME(n, edges, pop_size, max_gens, early_stop, max_time_seconds)
    else:
        return cpp_engine.solve_directed_MA(n, edges, pop_size, max_gens, early_stop, max_time_seconds)


def _kma_run_undirected(n, edges, pop_size, max_gens, early_stop, max_time_seconds):
    """Dispatch to available undirected KMA solver (KMA > KME > MA fallback)."""
    if hasattr(cpp_engine, "solve_undirected_KMA"):
        return cpp_engine.solve_undirected_KMA(n, edges, pop_size, max_gens, early_stop, max_time_seconds)
    elif hasattr(cpp_engine, "solve_undirected_KME"):
        return cpp_engine.solve_undirected_KME(n, edges, pop_size, max_gens, early_stop, max_time_seconds)
    else:
        return cpp_engine.solve_undirected_MA(n, edges, pop_size, max_gens, early_stop, max_time_seconds)


def _kernel_mapping_to_dict(k_new_to_old):
    if isinstance(k_new_to_old, dict):
        return dict(k_new_to_old)
    return {i: v for i, v in enumerate(k_new_to_old)}


def _commit_vertices_from_population(population, n_kernel, threshold):
    """
    Select committed vertices from population consensus.

    References:
      [PACE22] Dynamic interleaving of reduction and search opens reductions.
      [ESA22] Undo/redo style reductions expose new opportunities.
    """
    if not population or n_kernel <= 0:
        return set()

    counts = [0] * n_kernel
    pop_size = len(population)
    for individual in population:
        for v in set(individual):
            if 0 <= v < n_kernel:
                counts[v] += 1

    min_count = max(1, int(math.ceil(float(threshold) * pop_size)))
    committed = {v for v, c in enumerate(counts) if c >= min_count}
    if not committed:
        return set()

    if len(committed) > (n_kernel // 2):
        cap = max(1, int(math.ceil(0.30 * n_kernel)))
        ranked = sorted(committed, key=lambda v: (-counts[v], v))
        committed = set(ranked[:cap])
    return committed


def _apply_shortone_rule(k_n, k_edges, directed):
    """
    Apply SHORTONE-style contractions for directed kernels.

    Reference:
      [SEA23] Angrick et al., iterative directed reductions (Mount-Doom).
    """
    if not directed or k_n <= 1:
        return k_n, list(k_edges), {i: i for i in range(k_n)}

    curr_n = int(k_n)
    curr_edges = {(int(u), int(v)) for u, v in k_edges if 0 <= u < k_n and 0 <= v < k_n}
    merge_map = {i: i for i in range(curr_n)}

    changed = True
    while changed and curr_n > 1:
        changed = False
        out_adj = [set() for _ in range(curr_n)]
        in_adj = [set() for _ in range(curr_n)]
        for u, v in curr_edges:
            out_adj[u].add(v)
            in_adj[v].add(u)

        candidate = None
        for u, v in curr_edges:
            if u == v:
                continue
            if len(out_adj[u]) == 1 and len(in_adj[v]) == 1:
                candidate = (u, v)
                break
        if candidate is None:
            break

        u, v = candidate
        kept = [x for x in range(curr_n) if x not in {u, v}]
        rep_idx = len(kept)

        old_to_new = {old: i for i, old in enumerate(kept)}
        old_to_new[u] = rep_idx
        old_to_new[v] = rep_idx

        next_edges = set()
        for a, b in curr_edges:
            na = old_to_new[a]
            nb = old_to_new[b]
            if na == nb:
                continue
            next_edges.add((na, nb))

        next_merge_map = {}
        for old in kept:
            next_merge_map[old_to_new[old]] = merge_map[old]
        next_merge_map[rep_idx] = merge_map[v]

        curr_n = len(kept) + 1
        curr_edges = next_edges
        merge_map = next_merge_map
        changed = True

    return curr_n, sorted(curr_edges), merge_map


def _dynamic_kernelize(
    k_n,
    k_edges,
    committed_kernel_vertices,
    k_new_to_old,
    forced_so_far,
    directed,
):
    """
    Re-kernelize the residual graph after committing kernel vertices.

    Reference:
      [PACE22], [ESA22] for reduction-search interleaving motivation.
    """
    k_map = _kernel_mapping_to_dict(k_new_to_old)
    forced_original = set(forced_so_far)
    committed = {v for v in committed_kernel_vertices if 0 <= v < k_n}

    for v in committed:
        if v in k_map:
            forced_original.add(k_map[v])

    keep_vertices = [v for v in range(k_n) if v not in committed]
    keep_old_to_residual = {old: i for i, old in enumerate(keep_vertices)}
    residual_to_old_kernel = {i: old for old, i in keep_old_to_residual.items()}

    residual_edges = []
    for u, v in k_edges:
        if u in keep_old_to_residual and v in keep_old_to_residual:
            residual_edges.append((keep_old_to_residual[u], keep_old_to_residual[v]))

    residual_n = len(keep_vertices)
    if residual_n == 0:
        return 0, [], forced_original, {}, {}

    if directed:
        rk_n, rk_edges, rk_forced, rk_new_to_old = kernelize_directed_graph(residual_n, residual_edges)
    else:
        rk_n, rk_edges, rk_forced, rk_new_to_old = kernelize_undirected_graph(residual_n, residual_edges)

    new_k_new_to_old = {}
    for new_idx in range(rk_n):
        residual_old = rk_new_to_old[new_idx]
        old_kernel_idx = residual_to_old_kernel[residual_old]
        new_k_new_to_old[new_idx] = k_map[old_kernel_idx]

    for residual_forced_idx in rk_forced:
        if 0 <= residual_forced_idx < len(rk_new_to_old):
            residual_old = rk_new_to_old[residual_forced_idx]
            old_kernel_idx = residual_to_old_kernel[residual_old]
            forced_original.add(k_map[old_kernel_idx])

    old_kernel_to_new_kernel = {}
    for new_idx in range(rk_n):
        residual_old = rk_new_to_old[new_idx]
        old_kernel_idx = residual_to_old_kernel[residual_old]
        old_kernel_to_new_kernel[old_kernel_idx] = new_idx
    return rk_n, rk_edges, forced_original, new_k_new_to_old, old_kernel_to_new_kernel


def _is_acyclic(n, edges, removed_vertices, directed):
    """Check if graph minus removed vertices is acyclic."""
    removed = set(removed_vertices)
    active = [v not in removed for v in range(n)]

    if directed:
        out_adj = [[] for _ in range(n)]
        for u, v in edges:
            if 0 <= u < n and 0 <= v < n and active[u] and active[v]:
                out_adj[u].append(v)

        state = [0] * n
        for start in range(n):
            if not active[start] or state[start] != 0:
                continue
            state[start] = 1
            stack = [(start, 0)]
            while stack:
                node, idx = stack[-1]
                nbrs = out_adj[node]
                if idx >= len(nbrs):
                    state[node] = 2
                    stack.pop()
                    continue
                nb = nbrs[idx]
                stack[-1] = (node, idx + 1)
                if state[nb] == 1:
                    return False
                if state[nb] == 0:
                    state[nb] = 1
                    stack.append((nb, 0))
        return True

    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
        return True

    seen = set()
    for u, v in edges:
        if not (0 <= u < n and 0 <= v < n):
            continue
        if not active[u] or not active[v]:
            continue
        if u == v:
            return False
        a, b = (u, v) if u <= v else (v, u)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        if not union(a, b):
            return False
    return True


def _residual_max_degree(solution_set, k_n, k_edges, directed):
    removed = set(solution_set)
    degree = [0] * k_n
    if directed:
        for u, v in k_edges:
            if u in removed or v in removed:
                continue
            degree[u] += 1
            degree[v] += 1
    else:
        for u, v in k_edges:
            if u in removed or v in removed:
                continue
            degree[u] += 1
            degree[v] += 1
    return max(degree) if degree else 0


def _gain_local_search(solution_set, k_n, k_edges, directed, max_swaps=500):
    """
    Gain-driven local search using 1-1 and 2-1 swaps.

    References:
      [LANGEDAL], [PACE22], [TRSA24].
    """
    current = set(solution_set)
    if not _is_acyclic(k_n, k_edges, current, directed):
        return set(solution_set)

    adjacency = [set() for _ in range(k_n)]
    for u, v in k_edges:
        if 0 <= u < k_n and 0 <= v < k_n:
            adjacency[u].add(v)
            adjacency[v].add(u)

    swaps = 0
    improved = True
    while improved and swaps < max_swaps:
        improved = False
        base_secondary = _residual_max_degree(current, k_n, k_edges, directed)

        for v in list(current):
            candidate_us = [u for u in adjacency[v] if u not in current]
            for u in candidate_us:
                swaps += 1
                trial = set(current)
                trial.discard(v)
                trial.add(u)
                if not _is_acyclic(k_n, k_edges, trial, directed):
                    if swaps >= max_swaps:
                        break
                    continue
                new_secondary = _residual_max_degree(trial, k_n, k_edges, directed)
                if len(trial) < len(current) or (
                    len(trial) == len(current) and new_secondary < base_secondary
                ):
                    current = trial
                    improved = True
                    break
                if swaps >= max_swaps:
                    break
            if improved or swaps >= max_swaps:
                break
        if improved:
            continue

        current_list = list(current)
        outside = [u for u in range(k_n) if u not in current]
        for i in range(len(current_list)):
            for j in range(i + 1, len(current_list)):
                v1 = current_list[i]
                v2 = current_list[j]
                for u in outside:
                    swaps += 1
                    trial = set(current)
                    trial.discard(v1)
                    trial.discard(v2)
                    trial.add(u)
                    if _is_acyclic(k_n, k_edges, trial, directed) and len(trial) <= len(current):
                        current = trial
                        improved = True
                        break
                    if swaps >= max_swaps:
                        break
                if improved or swaps >= max_swaps:
                    break
            if improved or swaps >= max_swaps:
                break

    return current


def _diversify_with_topological_ordering(k_n, k_edges, current_solution, directed):
    """
    Diversification move inspired by randomized topological ordering.

    Reference:
      [PACE22] Kahn-order diversification every fixed rounds.
    """
    candidate = set(current_solution)
    if k_n <= 0:
        return candidate

    if directed:
        out_adj = [set() for _ in range(k_n)]
        in_adj = [set() for _ in range(k_n)]
        for u, v in k_edges:
            if 0 <= u < k_n and 0 <= v < k_n:
                out_adj[u].add(v)
                in_adj[v].add(u)

        while True:
            active = [v for v in range(k_n) if v not in candidate]
            if not active:
                break

            indeg = {v: 0 for v in active}
            for v in active:
                indeg[v] = sum(1 for p in in_adj[v] if p in indeg)

            zeros = [v for v in active if indeg[v] == 0]
            order_count = 0
            while zeros:
                idx = random.randrange(len(zeros))
                v = zeros.pop(idx)
                order_count += 1
                for nb in out_adj[v]:
                    if nb in indeg:
                        indeg[nb] -= 1
                        if indeg[nb] == 0:
                            zeros.append(nb)

            if order_count == len(active):
                break

            cyclic_vertices = [v for v in active if indeg.get(v, 0) > 0]
            if not cyclic_vertices:
                break
            pick = max(cyclic_vertices, key=lambda x: (len(in_adj[x]), x))
            candidate.add(pick)

        return candidate

    adj = [set() for _ in range(k_n)]
    for u, v in k_edges:
        if 0 <= u < k_n and 0 <= v < k_n:
            adj[u].add(v)
            adj[v].add(u)

    for _ in range(max(1, k_n)):
        if _is_acyclic(k_n, k_edges, candidate, directed=False):
            break

        active = [v for v in range(k_n) if v not in candidate]
        visited = set()
        added_this_round = set()

        for root in active:
            if root in visited:
                continue
            stack = [(root, -1, list(adj[root]))]
            random.shuffle(stack[0][2])
            visited.add(root)

            while stack:
                node, parent, nbrs = stack[-1]
                if not nbrs:
                    stack.pop()
                    continue
                nb = nbrs.pop()
                if nb == parent or nb in candidate:
                    continue
                if nb in visited:
                    chosen = nb if len(adj[nb]) >= len(adj[node]) else node
                    added_this_round.add(chosen)
                    continue
                visited.add(nb)
                next_nbrs = list(adj[nb])
                random.shuffle(next_nbrs)
                stack.append((nb, node, next_nbrs))

        if not added_this_round:
            residual = [v for v in range(k_n) if v not in candidate]
            if not residual:
                break
            pick = max(residual, key=lambda x: (len(adj[x]), x))
            candidate.add(pick)
        else:
            candidate.update(added_this_round)

    return candidate


def _build_random_population(k_n, target_size):
    pop = []
    if k_n <= 0 or target_size <= 0:
        return pop
    for _ in range(target_size):
        p = random.uniform(0.05, 0.20)
        individual = {v for v in range(k_n) if random.random() < p}
        pop.append(individual)
    return pop


def _greedy_acyclic_repair(n, edges, removed_vertices, directed):
    """Ensure validity by greedily adding high-degree vertices until acyclic."""
    removed = set(removed_vertices)
    if _is_acyclic(n, edges, removed, directed):
        return removed

    for _ in range(n):
        if _is_acyclic(n, edges, removed, directed):
            break
        degree = [0] * n
        for u, v in edges:
            if u in removed or v in removed:
                continue
            if 0 <= u < n:
                degree[u] += 1
            if 0 <= v < n:
                degree[v] += 1
        candidates = [v for v in range(n) if v not in removed]
        if not candidates:
            break
        pick = max(candidates, key=lambda x: (degree[x], x))
        removed.add(pick)
    return removed


def _prune_redundant_vertices(n, edges, solution_vertices, directed, max_seconds=2.0):
    """Try to remove redundant solution vertices while preserving acyclicity."""
    current = set(solution_vertices)
    if not current:
        return current
    start = time.time()

    # Low-degree vertices are more likely redundant after local search/MA.
    degree = [0] * n
    for u, v in edges:
        if 0 <= u < n:
            degree[u] += 1
        if 0 <= v < n:
            degree[v] += 1
    order = sorted(current, key=lambda v: (degree[v], v))

    for v in order:
        if (time.time() - start) > max_seconds:
            break
        trial = set(current)
        trial.discard(v)
        if _is_acyclic(n, edges, trial, directed):
            current = trial
    return current


def _run_one_generation_ma(
    k_n,
    k_edges,
    population,
    directed,
    remaining_time_seconds=None,
    use_cpp_step=True,
):
    """
    Run a short MA step and merge offspring back into the population.

    If exact single-generation stepping is unavailable from cpp_engine, this
    executes a 1-generation MA/KMA call as a short evolution proxy.
    """
    if k_n <= 0:
        return []

    target_pop = max(1, len(population) if population else 20)
    sanitized = []
    for individual in (population or []):
        sanitized.append({v for v in individual if 0 <= v < k_n})

    offspring = []

    # Lightweight in-Python evolution to avoid spending full timeout each gen.
    if sanitized:
        ranked = sorted(sanitized, key=len)
        elite = ranked[: max(1, target_pop // 2)]
        for _ in range(max(1, target_pop // 3)):
            p1 = random.choice(elite)
            p2 = random.choice(elite)
            child = set(p1)
            if p2:
                child.symmetric_difference_update(set(random.sample(list(p2), k=max(0, len(p2) // 4))))
            for _ in range(max(1, k_n // 200)):
                v = random.randrange(k_n)
                if random.random() < 0.5:
                    child.add(v)
                else:
                    child.discard(v)
            offspring.append({v for v in child if 0 <= v < k_n})

    # Only occasionally call cpp single-gen stepping, and only with real time left.
    time_left = float(remaining_time_seconds) if remaining_time_seconds is not None else 0.0
    cpp_budget = max(0, min(2, int(time_left) - 1))
    if use_cpp_step and cpp_budget >= 1:
        try:
            if directed:
                if hasattr(cpp_engine, "solve_directed_KMA"):
                    child = cpp_engine.solve_directed_KMA(k_n, k_edges, max(2, target_pop // 2), 1, 1, cpp_budget)
                elif hasattr(cpp_engine, "solve_directed_KME"):
                    child = cpp_engine.solve_directed_KME(k_n, k_edges, max(2, target_pop // 2), 1, 1, cpp_budget)
                else:
                    child = cpp_engine.solve_directed_MA(k_n, k_edges, max(2, target_pop // 2), 1, 1, cpp_budget)
            else:
                if hasattr(cpp_engine, "solve_undirected_KMA"):
                    child = cpp_engine.solve_undirected_KMA(k_n, k_edges, max(2, target_pop // 2), 1, 1, cpp_budget)
                elif hasattr(cpp_engine, "solve_undirected_KME"):
                    child = cpp_engine.solve_undirected_KME(k_n, k_edges, max(2, target_pop // 2), 1, 1, cpp_budget)
                else:
                    child = cpp_engine.solve_undirected_MA(k_n, k_edges, max(2, target_pop // 2), 1, 1, cpp_budget)
            offspring.append({v for v in child if 0 <= v < k_n})
        except Exception:
            pass

    merged = sanitized + offspring
    if not merged:
        merged = _build_random_population(k_n, target_pop)

    uniq = {}
    for ind in merged:
        uniq[tuple(sorted(ind))] = set(ind)
    ranked = sorted(uniq.values(), key=lambda s: (len(s), tuple(sorted(s))))

    if len(ranked) < target_pop:
        ranked.extend(_build_random_population(k_n, target_pop - len(ranked)))

    return ranked[:target_pop]


def _remap_population(population, committed_old_kernel_indices, old_to_new_map, new_k_n, pop_size=20):
    """Remap individuals from old kernel indices to new kernel indices."""
    committed = set(committed_old_kernel_indices)
    remapped = []
    seen = set()

    for individual in population:
        new_ind = set()
        for v in individual:
            if v in committed:
                continue
            new_idx = old_to_new_map.get(v)
            if new_idx is not None and 0 <= new_idx < new_k_n:
                new_ind.add(new_idx)
        key = tuple(sorted(new_ind))
        if key not in seen:
            seen.add(key)
            remapped.append(new_ind)

    min_size = max(1, pop_size // 2)
    if len(remapped) < min_size:
        remapped.extend(_build_random_population(new_k_n, min_size - len(remapped)))
    return remapped


def _initialize_population(k_n, k_edges, directed, pop_size, max_gens, early_stop, max_time_seconds):
    if k_n <= 0:
        return []
    population = []
    # Use a single warm-start KMA call when enough budget exists.
    # This keeps DKMA quality close to KMA while avoiding per-generation timeout spam.
    if max_time_seconds >= 20:
        try:
            seed_gens = max(5, min(20, max_gens))
            seed_time = max(8, min(int(max_time_seconds * 0.5), max_time_seconds - 2))
            if directed:
                base = _kma_run_directed(k_n, k_edges, pop_size, seed_gens, early_stop, seed_time)
            else:
                base = _kma_run_undirected(k_n, k_edges, pop_size, seed_gens, early_stop, seed_time)
            population.append({v for v in base if 0 <= v < k_n})
        except Exception:
            pass

    if len(population) < pop_size:
        population.extend(_build_random_population(k_n, pop_size - len(population)))

    uniq = {}
    for ind in population:
        uniq[tuple(sorted(ind))] = set(ind)
    ranked = sorted(uniq.values(), key=lambda s: len(s))
    if len(ranked) < pop_size:
        ranked.extend(_build_random_population(k_n, pop_size - len(ranked)))
    return ranked[:pop_size]


def _dkma_solve_common(
    n,
    edges,
    directed,
    pop_size=20,
    max_gens=100,
    early_stop=10,
    max_time_seconds=600,
    commit_threshold=0.6,
    dynkern_every=5,
    gain_search=True,
    diversify=True,
    dkma_verify=False,
    return_diagnostics=False,
):
    if not HAS_CPP_ENGINE:
        raise RuntimeError("cpp_engine not available. Please compile it first.")

    # For very short budgets, use stable KMA plus non-worsening post-pruning.
    if max_time_seconds <= 30:
        short_start = time.time()
        if directed:
            base_solution, base_metrics = kma_solve_directed(
                n,
                edges,
                pop_size=pop_size,
                max_gens=max_gens,
                max_time_seconds=max_time_seconds,
                early_stop=early_stop,
                return_diagnostics=True,
            )
        else:
            base_solution, base_metrics = kma_solve_undirected(
                n,
                edges,
                pop_size=pop_size,
                max_gens=max_gens,
                max_time_seconds=max_time_seconds,
                early_stop=early_stop,
                return_diagnostics=True,
            )

        best_solution = set(base_solution)

        # If KMA ends early, spend leftover budget on a diversified second shot.
        elapsed_short = time.time() - short_start
        remaining_short = int(max_time_seconds - elapsed_short)
        if remaining_short >= 3:
            try:
                if directed:
                    alt_solution = kma_solve_directed(
                        n,
                        edges,
                        pop_size=max(8, pop_size + max(2, pop_size // 2)),
                        max_gens=max(20, max_gens // 2),
                        max_time_seconds=remaining_short,
                        early_stop=max(5, early_stop // 2),
                    )
                else:
                    alt_solution = kma_solve_undirected(
                        n,
                        edges,
                        pop_size=max(8, pop_size + max(2, pop_size // 2)),
                        max_gens=max(20, max_gens // 2),
                        max_time_seconds=remaining_short,
                        early_stop=max(5, early_stop // 2),
                    )
                if _is_acyclic(n, edges, set(alt_solution), directed) and len(alt_solution) < len(best_solution):
                    best_solution = set(alt_solution)
            except Exception:
                pass

        improved_solution = _prune_redundant_vertices(
            n,
            edges,
            best_solution,
            directed,
            max_seconds=2.0,
        )
        if not _is_acyclic(n, edges, improved_solution, directed):
            improved_solution = set(best_solution)
        if len(improved_solution) <= len(best_solution):
            final_solution = sorted(improved_solution)
        else:
            final_solution = sorted(best_solution)

        base_metrics = dict(base_metrics or {})
        base_metrics.setdefault("initial_kernel_size", n)
        base_metrics.setdefault("final_kernel_size", n)
        base_metrics.setdefault("n_dynamic_reductions", 0)
        return _maybe_with_metrics(final_solution, base_metrics, return_diagnostics)

    wall_start = time.time()

    t0 = time.perf_counter()
    if directed:
        k_n, k_edges, forced, k_new_to_old_raw = kernelize_directed_graph(n, edges)
    else:
        k_n, k_edges, forced, k_new_to_old_raw = kernelize_undirected_graph(n, edges)
    kernel_ms = (time.perf_counter() - t0) * 1000.0

    k_new_to_old = _kernel_mapping_to_dict(k_new_to_old_raw)
    initial_kernel_size = k_n

    if k_n == 0:
        solution = sorted(set(forced))
        metrics = _stage_metrics(kernelization_ms=kernel_ms, ma_ms=0.0)
        metrics.update({
            "initial_kernel_size": initial_kernel_size,
            "final_kernel_size": 0,
            "n_dynamic_reductions": 0,
        })
        return _maybe_with_metrics(solution, metrics, return_diagnostics)

    if directed:
        k_n, k_edges, shortone_map = _apply_shortone_rule(k_n, k_edges, directed=True)
        k_new_to_old = {
            new_idx: k_new_to_old[old_idx]
            for new_idx, old_idx in shortone_map.items()
            if old_idx in k_new_to_old
        }

    ma_start = time.perf_counter()
    population = _initialize_population(
        k_n, k_edges, directed, pop_size, max_gens, early_stop, max_time_seconds
    )
    if not population:
        population = _build_random_population(k_n, pop_size)
    best_solution = min(population, key=len) if population else set()
    best_original = set(forced) | {k_new_to_old[v] for v in best_solution if v in k_new_to_old}

    current_forced = set(forced)
    current_k_n = k_n
    current_k_edges = list(k_edges)
    current_k_new_to_old = dict(k_new_to_old)

    gen = 0
    stagnation = 0
    n_dynamic_reductions = 0

    while gen < max_gens and stagnation < early_stop:
        elapsed = time.time() - wall_start
        if elapsed > max_time_seconds:
            break

        remaining = max_time_seconds - elapsed

        population = _run_one_generation_ma(
            current_k_n,
            current_k_edges,
            population,
            directed,
            remaining_time_seconds=remaining,
            use_cpp_step=False,
        )

        if diversify and gen % 5 == 0 and current_k_n > 0:
            diverse_candidate = _diversify_with_topological_ordering(
                current_k_n, current_k_edges, best_solution, directed
            )
            population.append(set(diverse_candidate))
            population = sorted(population, key=len)[:pop_size]

        if not population:
            break

        ranked_population = sorted(population, key=len)
        local_best = ranked_population[0]
        local_best_original = current_forced | {
            current_k_new_to_old[v] for v in local_best if v in current_k_new_to_old
        }

        found_valid = False
        for candidate in ranked_population:
            candidate_original = current_forced | {
                current_k_new_to_old[v] for v in candidate if v in current_k_new_to_old
            }
            if _is_acyclic(n, edges, candidate_original, directed):
                local_best = candidate
                local_best_original = candidate_original
                found_valid = True
                break

        if not found_valid:
            local_best_original = _greedy_acyclic_repair(n, edges, local_best_original, directed)
            if dkma_verify:
                print("[DKMA] verification warning: repaired invalid intermediate candidate")

        if len(local_best_original) < len(best_original):
            best_solution = set(local_best)
            best_original = set(local_best_original)
            stagnation = 0
        else:
            stagnation += 1

        if dynkern_every > 0 and gen % dynkern_every == 0 and gen > 0 and current_k_n > 0:
            committed = _commit_vertices_from_population(population, current_k_n, commit_threshold)
            if committed:
                (
                    new_k_n,
                    new_k_edges,
                    new_forced,
                    new_map,
                    new_old_to_new,
                ) = _dynamic_kernelize(
                    current_k_n,
                    current_k_edges,
                    committed,
                    current_k_new_to_old,
                    current_forced,
                    directed,
                )

                if new_k_n < current_k_n:
                    old_kernel_best = set(best_solution)

                    current_k_n = new_k_n
                    current_k_edges = list(new_k_edges)
                    current_forced = set(new_forced)
                    current_k_new_to_old = dict(new_map)
                    population = _remap_population(
                        population,
                        committed,
                        new_old_to_new,
                        current_k_n,
                        pop_size=pop_size,
                    )

                    best_solution = {
                        new_old_to_new[o]
                        for o in old_kernel_best
                        if o in new_old_to_new
                    }
                    best_original = current_forced | {
                        current_k_new_to_old[v]
                        for v in best_solution
                        if v in current_k_new_to_old
                    }
                    if not best_solution and population:
                        best_solution = min(population, key=len)
                        best_original = current_forced | {
                            current_k_new_to_old[v]
                            for v in best_solution
                            if v in current_k_new_to_old
                        }

                    if len(population) == 0:
                        break

                    n_dynamic_reductions += 1
                    stagnation = 0

        gen += 1

    if gain_search and current_k_n > 0:
        improved_kernel_sol = _gain_local_search(
            best_solution,
            current_k_n,
            current_k_edges,
            directed,
            max_swaps=1000,
        )
        improved_original = current_forced | {
            current_k_new_to_old[v]
            for v in improved_kernel_sol
            if v in current_k_new_to_old
        }
        if len(improved_original) <= len(best_original):
            best_original = improved_original

    if not _is_acyclic(n, edges, best_original, directed):
        best_original = _greedy_acyclic_repair(n, edges, best_original, directed)

    if not _is_acyclic(n, edges, best_original, directed):
        print("[DKMA] warning: final solution failed validation, falling back to KMA")
        remaining_for_fallback = int(max_time_seconds - (time.time() - wall_start))
        if remaining_for_fallback >= 2:
            if directed:
                fallback = kma_solve_directed(
                    n,
                    edges,
                    pop_size=pop_size,
                    max_gens=max_gens,
                    max_time_seconds=remaining_for_fallback,
                    early_stop=early_stop,
                )
            else:
                fallback = kma_solve_undirected(
                    n,
                    edges,
                    pop_size=pop_size,
                    max_gens=max_gens,
                    max_time_seconds=remaining_for_fallback,
                    early_stop=early_stop,
                )
            best_original = set(fallback)
        else:
            best_original = _greedy_acyclic_repair(n, edges, best_original, directed)

    ma_ms = (time.perf_counter() - ma_start) * 1000.0
    metrics = _stage_metrics(kernelization_ms=kernel_ms, ma_ms=ma_ms)
    metrics.update({
        "initial_kernel_size": initial_kernel_size,
        "final_kernel_size": current_k_n,
        "n_dynamic_reductions": n_dynamic_reductions,
    })
    return _maybe_with_metrics(sorted(set(best_original)), metrics, return_diagnostics)


def dkma_solve_undirected(
    n,
    edges,
    pop_size=20,
    max_gens=100,
    early_stop=10,
    max_time_seconds=600,
    commit_threshold=0.6,
    dynkern_every=5,
    gain_search=True,
    diversify=True,
    dkma_verify=False,
    return_diagnostics=False,
):
    """Dynamic Kernelized Memetic Algorithm for undirected FVS."""
    return _dkma_solve_common(
        n,
        edges,
        directed=False,
        pop_size=pop_size,
        max_gens=max_gens,
        early_stop=early_stop,
        max_time_seconds=max_time_seconds,
        commit_threshold=commit_threshold,
        dynkern_every=dynkern_every,
        gain_search=gain_search,
        diversify=diversify,
        dkma_verify=dkma_verify,
        return_diagnostics=return_diagnostics,
    )


def dkma_solve_directed(
    n,
    edges,
    pop_size=20,
    max_gens=100,
    early_stop=10,
    max_time_seconds=600,
    commit_threshold=0.6,
    dynkern_every=5,
    gain_search=True,
    diversify=True,
    dkma_verify=False,
    return_diagnostics=False,
):
    """Dynamic Kernelized Memetic Algorithm for directed FVS."""
    return _dkma_solve_common(
        n,
        edges,
        directed=True,
        pop_size=pop_size,
        max_gens=max_gens,
        early_stop=early_stop,
        max_time_seconds=max_time_seconds,
        commit_threshold=commit_threshold,
        dynkern_every=dynkern_every,
        gain_search=gain_search,
        diversify=diversify,
        dkma_verify=dkma_verify,
        return_diagnostics=return_diagnostics,
    )


def gnn_dkma_solve_undirected(
    n,
    edges,
    gnn_version="v1",
    gnn_threshold=0.65,
    gnn_hidden=None,
    gnn_timeout=60,
    **dkma_kwargs,
):
    """GNN-DKMA: GNN-guided hard-fix followed by DKMA on the reduced kernel."""
    k_n, k_edges, forced, k_new_to_old = kernelize_undirected_graph(n, edges)
    if k_n == 0:
        return sorted(set(forced))

    if gnn_version == "v2":
        probs = run_gnn_undirected_v2_probs(k_n, k_edges, hidden_dim=gnn_hidden, gnn_timeout=gnn_timeout)
    else:
        probs = run_gnn_undirected_probs(k_n, k_edges, hidden_dim=gnn_hidden, gnn_timeout=gnn_timeout)

    fixed = set()
    if probs is not None:
        for idx, p in enumerate(probs):
            if p >= gnn_threshold:
                fixed.add(idx)

    keep = [v for v in range(k_n) if v not in fixed]
    old_to_reduced = {old: i for i, old in enumerate(keep)}
    reduced_edges = [
        (old_to_reduced[u], old_to_reduced[v])
        for u, v in k_edges
        if u in old_to_reduced and v in old_to_reduced
    ]

    reduced_solution = dkma_solve_undirected(
        len(keep),
        reduced_edges,
        **dkma_kwargs,
    )
    reduced_to_old = {i: old for old, i in old_to_reduced.items()}

    answer = set(forced)
    answer.update(k_new_to_old[v] for v in fixed if 0 <= v < len(k_new_to_old))
    answer.update(k_new_to_old[reduced_to_old[v]] for v in reduced_solution if v in reduced_to_old)
    return sorted(answer)


def gnn_dkma_solve_directed(
    n,
    edges,
    gnn_version="v1",
    gnn_threshold=0.65,
    gnn_hidden=None,
    gnn_timeout=60,
    **dkma_kwargs,
):
    """GNN-DKMA: GNN-guided hard-fix followed by DKMA on the reduced kernel."""
    k_n, k_edges, forced, k_new_to_old = kernelize_directed_graph(n, edges)
    if k_n == 0:
        return sorted(set(forced))

    if gnn_version == "v2":
        probs = run_gnn_directed_v2_probs(k_n, k_edges, hidden_dim=gnn_hidden, gnn_timeout=gnn_timeout)
    elif gnn_version == "v3":
        probs = run_gnn_directed_v3_probs(k_n, k_edges, hidden_dim=gnn_hidden, gnn_timeout=gnn_timeout)
    else:
        probs = run_gnn_directed_probs(k_n, k_edges, hidden_dim=gnn_hidden, gnn_timeout=gnn_timeout)

    fixed = set()
    if probs is not None:
        for idx, p in enumerate(probs):
            if p >= gnn_threshold:
                fixed.add(idx)

    keep = [v for v in range(k_n) if v not in fixed]
    old_to_reduced = {old: i for i, old in enumerate(keep)}
    reduced_edges = [
        (old_to_reduced[u], old_to_reduced[v])
        for u, v in k_edges
        if u in old_to_reduced and v in old_to_reduced
    ]

    reduced_solution = dkma_solve_directed(
        len(keep),
        reduced_edges,
        **dkma_kwargs,
    )
    reduced_to_old = {i: old for old, i in old_to_reduced.items()}

    answer = set(forced)
    answer.update(k_new_to_old[v] for v in fixed if 0 <= v < len(k_new_to_old))
    answer.update(k_new_to_old[reduced_to_old[v]] for v in reduced_solution if v in reduced_to_old)
    return sorted(answer)


def _soft_hint_gnn_kernel_solve(
    k_n, k_edges, gnn_probs_np, pop_size, max_gens, early_stop, max_time_seconds,
    gnn_threshold, max_fix_fraction, label, directed, run_kma_fn
):
    """
    Soft-hint GNN-KMA core: high-confidence hard-fix + KMA on remainder.

    This is the fundamental fix for the GNN-KMA coupling problem.
    GNN probability scores are used in two ways:
      1. Hard-fix: ONLY vertices with prob >= gnn_threshold (AND <= max_fix_fraction)
         are removed from the kernel before KMA. This precision guard ensures
         that locked-in vertices are almost certainly in the true FVS.
      2. No-fix fallback: If NO vertex meets the threshold, skip GNN entirely
         and run pure KMA on the full kernel. Better safe than sorry.

    Reference: Ben-Baruch et al. (2021) Asymmetric Loss — the cost of a false
        positive (FP locked into solution) greatly exceeds the cost of a
        false negative (KMA can still find it).

    Args:
        k_n: kernel vertex count
        k_edges: kernel edge list
        gnn_probs_np: numpy float32 array shape (k_n,), or None for no-GNN mode
        gnn_threshold: high-confidence fix threshold (default 0.65)
        max_fix_fraction: max fraction of kernel to hard-fix (default 0.08)
        label: logging prefix e.g. "[KMA]"
        directed: whether kernel is directed
        run_kma_fn: callable(n, edges, pop, gens, early_stop, time) -> fvs list

    Returns:
        (kernel_fvs set, gnn_applied_bool)
    """
    import numpy as np

    # Determine high-confidence hard-fix set
    if gnn_probs_np is not None:
        torch_mod = get_torch()
        if torch_mod is not None:
            prob_tensor = torch_mod.tensor(gnn_probs_np, dtype=torch_mod.float)
            fixed_kernel, sel_mode = _pick_gnn_candidates_from_probs(
                prob_tensor,
                threshold=gnn_threshold,
                min_fraction=0.005,
                max_fraction=max_fix_fraction,
            )
            gnn_applied = (len(fixed_kernel) > 0)
            n_conf = (gnn_probs_np >= gnn_threshold).sum()
            print(
                f"  {label} GNN probs: {n_conf}/{k_n} vertices above threshold={gnn_threshold:.2f}, "
                f"hard-fixed={len(fixed_kernel)} (mode={sel_mode})"
            )
        else:
            fixed_kernel = set()
            gnn_applied = False
    else:
        print(f"  {label} GNN unavailable, running pure KMA (pop={pop_size}, gens={max_gens})")
        fixed_kernel = set()
        gnn_applied = False

    # Build reduced kernel (minus hard-fixed vertices)
    if fixed_kernel:
        keep_kernel = [v for v in range(k_n) if v not in fixed_kernel]
        kernel_to_reduced = {old: i for i, old in enumerate(keep_kernel)}
        reduced_edges = [
            (kernel_to_reduced[u], kernel_to_reduced[v])
            for u, v in k_edges
            if u in kernel_to_reduced and v in kernel_to_reduced
        ]
        reduced_n = len(keep_kernel)
        reduced_to_kernel = keep_kernel
    else:
        reduced_n = k_n
        reduced_edges = k_edges
        reduced_to_kernel = list(range(k_n))

    # Run KMA on reduced kernel
    if reduced_n > 0:
        reduced_fvs = run_kma_fn(
            reduced_n, reduced_edges, pop_size, max_gens, early_stop, max_time_seconds
        )
    else:
        reduced_fvs = []

    # Reconstruct kernel FVS
    kernel_fvs = set(fixed_kernel)
    kernel_fvs.update(
        reduced_to_kernel[v] for v in reduced_fvs if 0 <= v < len(reduced_to_kernel)
    )
    return kernel_fvs, gnn_applied


def gnn_KMA_solve_undirected(n, edges, pop_size=20, max_gens=100,
                              gnn_threshold=0.65, gnn_hidden_dim=None,
                              gnn_timeout=60,
                              max_time_seconds=600, early_stop=20,
                              return_diagnostics=False):
    """
    GNN-KMA (soft-hint): kernelize → GNN probability scores → precision-guarded
    hard-fix → KMA refinement for undirected FVS.

    Unlike the legacy hard-fix design, this version:
      1. Gets raw GNN probabilities (not binary candidates)
      2. Hard-fixes ONLY vertices with prob >= gnn_threshold (default 0.65)
         and only up to 8% of the kernel
      3. Falls through to pure KMA if no vertex is confident enough
      4. KMA can override any GNN prediction in the unfixed region

    This eliminates the catastrophic false-positive problem of the legacy design.

    Reference: Precision-first coupling — Part 1 of the GNN-KMA research-grade
        overhaul. Ben-Baruch et al. (2021) Asymmetric Loss motivation.
    """
    if not HAS_CPP_ENGINE:
        raise RuntimeError("cpp_engine not available. Please compile it first.")

    kernel_start = time.perf_counter()
    k_n, k_edges, forced, k_new_to_old = kernelize_undirected_graph(n, edges)
    kernel_ms = (time.perf_counter() - kernel_start) * 1000.0

    if k_n == 0:
        return _maybe_with_metrics(
            sorted(forced),
            _stage_metrics(kernelization_ms=kernel_ms),
            return_diagnostics,
        )

    gnn_start = time.perf_counter()
    gnn_probs_np = run_gnn_undirected_probs(
        k_n, k_edges, hidden_dim=gnn_hidden_dim, gnn_timeout=gnn_timeout
    )
    gnn_ms = (time.perf_counter() - gnn_start) * 1000.0

    ma_start = time.perf_counter()
    kernel_fvs, _ = _soft_hint_gnn_kernel_solve(
        k_n, k_edges, gnn_probs_np, pop_size, max_gens, early_stop, max_time_seconds,
        gnn_threshold=gnn_threshold, max_fix_fraction=0.08,
        label="[KMA]", directed=False, run_kma_fn=_kma_run_undirected,
    )
    ma_ms = (time.perf_counter() - ma_start) * 1000.0

    mapped = [k_new_to_old[v] for v in kernel_fvs if 0 <= v < len(k_new_to_old)]
    solution = sorted(set(forced).union(mapped))
    return _maybe_with_metrics(
        solution,
        _stage_metrics(kernelization_ms=kernel_ms, gnn_candidate_ms=gnn_ms, ma_ms=ma_ms),
        return_diagnostics,
    )


def gnn_KMA_solve_directed(n, edges, pop_size=20, max_gens=100,
                            gnn_threshold=0.65, gnn_hidden_dim=None,
                            gnn_timeout=60,
                            max_time_seconds=600, early_stop=20,
                            return_diagnostics=False):
    """
    GNN-KMA (soft-hint): kernelize → GNN probability scores → precision-guarded
    hard-fix → KMA refinement for directed FVS.

    Unlike the legacy hard-fix design, this version:
      1. Gets raw GNN probabilities (not binary candidates)
      2. Hard-fixes ONLY vertices with prob >= gnn_threshold (default 0.65)
         and only up to 8% of the kernel
      3. Falls through to pure KMA if no vertex is confident enough
      4. Eliminates catastrophic false-positive inflation of legacy design

    Reference: Part 1 of GNN-KMA research-grade overhaul.
    """
    if not HAS_CPP_ENGINE:
        raise RuntimeError("cpp_engine not available. Please compile it first.")

    kernel_start = time.perf_counter()
    k_n, k_edges, forced, k_new_to_old = kernelize_directed_graph(n, edges)
    kernel_ms = (time.perf_counter() - kernel_start) * 1000.0

    if k_n == 0:
        return _maybe_with_metrics(
            sorted(forced),
            _stage_metrics(kernelization_ms=kernel_ms),
            return_diagnostics,
        )

    gnn_start = time.perf_counter()
    gnn_probs_np = run_gnn_directed_probs(
        k_n, k_edges, hidden_dim=gnn_hidden_dim, gnn_timeout=gnn_timeout
    )
    gnn_ms = (time.perf_counter() - gnn_start) * 1000.0

    ma_start = time.perf_counter()
    kernel_fvs, _ = _soft_hint_gnn_kernel_solve(
        k_n, k_edges, gnn_probs_np, pop_size, max_gens, early_stop, max_time_seconds,
        gnn_threshold=gnn_threshold, max_fix_fraction=0.08,
        label="[KMA]", directed=True, run_kma_fn=_kma_run_directed,
    )
    ma_ms = (time.perf_counter() - ma_start) * 1000.0

    mapped = [k_new_to_old[v] for v in kernel_fvs if 0 <= v < len(k_new_to_old)]
    solution = sorted(set(forced).union(mapped))
    return _maybe_with_metrics(
        solution,
        _stage_metrics(kernelization_ms=kernel_ms, gnn_candidate_ms=gnn_ms, ma_ms=ma_ms),
        return_diagnostics,
    )


def gnn_KMA2_solve_undirected(n, edges, pop_size=20, max_gens=100,
                               gnn_threshold=0.65, gnn_hidden_dim=None,
                               gnn_timeout=60,
                               max_time_seconds=600, early_stop=20,
                               return_diagnostics=False):
    """
    GNN-KMA-2 (soft-hint): same as gnn_KMA_solve_undirected but uses the
    v2 GNN model with enriched structural features (RWSE, motif counts, k-core).

    Reference: Part 1+2 of the GNN-KMA research-grade overhaul.
    """
    if not HAS_CPP_ENGINE:
        raise RuntimeError("cpp_engine not available. Please compile it first.")

    kernel_start = time.perf_counter()
    k_n, k_edges, forced, k_new_to_old = kernelize_undirected_graph(n, edges)
    kernel_ms = (time.perf_counter() - kernel_start) * 1000.0

    if k_n == 0:
        return _maybe_with_metrics(
            sorted(forced),
            _stage_metrics(kernelization_ms=kernel_ms),
            return_diagnostics,
        )

    gnn_start = time.perf_counter()
    gnn_probs_np = run_gnn_undirected_v2_probs(
        k_n, k_edges, hidden_dim=gnn_hidden_dim, gnn_timeout=gnn_timeout
    )
    gnn_ms = (time.perf_counter() - gnn_start) * 1000.0

    ma_start = time.perf_counter()
    kernel_fvs, _ = _soft_hint_gnn_kernel_solve(
        k_n, k_edges, gnn_probs_np, pop_size, max_gens, early_stop, max_time_seconds,
        gnn_threshold=gnn_threshold, max_fix_fraction=0.08,
        label="[KMA-2]", directed=False, run_kma_fn=_kma_run_undirected,
    )
    ma_ms = (time.perf_counter() - ma_start) * 1000.0

    mapped = [k_new_to_old[v] for v in kernel_fvs if 0 <= v < len(k_new_to_old)]
    solution = sorted(set(forced).union(mapped))
    return _maybe_with_metrics(
        solution,
        _stage_metrics(kernelization_ms=kernel_ms, gnn_candidate_ms=gnn_ms, ma_ms=ma_ms),
        return_diagnostics,
    )


def gnn_KMA2_solve_directed(n, edges, pop_size=20, max_gens=100,
                             gnn_threshold=0.65, gnn_hidden_dim=None,
                             gnn_timeout=60,
                             max_time_seconds=600, early_stop=20,
                             return_diagnostics=False):
    """
    GNN-KMA-2 (soft-hint): same as gnn_KMA_solve_directed but uses the v2 GNN
    model with enriched structural features (RWSE, motif counts, k-core).

    Reference: Part 1+2 of the GNN-KMA research-grade overhaul.
    """
    if not HAS_CPP_ENGINE:
        raise RuntimeError("cpp_engine not available. Please compile it first.")

    kernel_start = time.perf_counter()
    k_n, k_edges, forced, k_new_to_old = kernelize_directed_graph(n, edges)
    kernel_ms = (time.perf_counter() - kernel_start) * 1000.0

    if k_n == 0:
        return _maybe_with_metrics(
            sorted(forced),
            _stage_metrics(kernelization_ms=kernel_ms),
            return_diagnostics,
        )

    gnn_start = time.perf_counter()
    gnn_probs_np = run_gnn_directed_v2_probs(
        k_n, k_edges, hidden_dim=gnn_hidden_dim, gnn_timeout=gnn_timeout
    )
    gnn_ms = (time.perf_counter() - gnn_start) * 1000.0

    ma_start = time.perf_counter()
    kernel_fvs, _ = _soft_hint_gnn_kernel_solve(
        k_n, k_edges, gnn_probs_np, pop_size, max_gens, early_stop, max_time_seconds,
        gnn_threshold=gnn_threshold, max_fix_fraction=0.08,
        label="[KMA-2]", directed=True, run_kma_fn=_kma_run_directed,
    )
    ma_ms = (time.perf_counter() - ma_start) * 1000.0

    mapped = [k_new_to_old[v] for v in kernel_fvs if 0 <= v < len(k_new_to_old)]
    solution = sorted(set(forced).union(mapped))
    return _maybe_with_metrics(
        solution,
        _stage_metrics(kernelization_ms=kernel_ms, gnn_candidate_ms=gnn_ms, ma_ms=ma_ms),
        return_diagnostics,
    )


def gnn_KMA3_solve_directed(n, edges, pop_size=20, max_gens=100,
                             gnn_threshold=0.65, gnn_hidden_dim=None,
                             gnn_timeout=60,
                             max_time_seconds=600, early_stop=20,
                             return_diagnostics=False):
    """
    GNN-KMA-3: research-grade directed FVS solver.

    Uses the v3 model (GAT + residual connections, 5 layers, 16-channel features
    with RWSE steps 2-16, SCC membership, cycle betweenness) with precision-first
    soft-hint coupling (threshold=0.65, max 8% hard-fixed).

    References:
    - Veličković et al. (2018) Graph Attention Networks (GAT)
    - He et al. (2016) Deep Residual Learning
    - Rampasek et al. (2022) Recipe for a General, Powerful, Scalable Graph
      Transformer (RWSE)
    - Ben-Baruch et al. (2021) Asymmetric Loss (FP >> FN cost design)
    """
    if not HAS_CPP_ENGINE:
        raise RuntimeError("cpp_engine not available. Please compile it first.")

    kernel_start = time.perf_counter()
    k_n, k_edges, forced, k_new_to_old = kernelize_directed_graph(n, edges)
    kernel_ms = (time.perf_counter() - kernel_start) * 1000.0

    if k_n == 0:
        return _maybe_with_metrics(
            sorted(forced),
            _stage_metrics(kernelization_ms=kernel_ms),
            return_diagnostics,
        )

    gnn_start = time.perf_counter()
    gnn_probs_np = run_gnn_directed_v3_probs(
        k_n, k_edges, hidden_dim=gnn_hidden_dim, gnn_timeout=gnn_timeout
    )
    gnn_ms = (time.perf_counter() - gnn_start) * 1000.0

    ma_start = time.perf_counter()
    kernel_fvs, _ = _soft_hint_gnn_kernel_solve(
        k_n, k_edges, gnn_probs_np, pop_size, max_gens, early_stop, max_time_seconds,
        gnn_threshold=gnn_threshold, max_fix_fraction=0.08,
        label="[KMA-3]", directed=True, run_kma_fn=_kma_run_directed,
    )
    ma_ms = (time.perf_counter() - ma_start) * 1000.0

    mapped = [k_new_to_old[v] for v in kernel_fvs if 0 <= v < len(k_new_to_old)]
    solution = sorted(set(forced).union(mapped))
    return _maybe_with_metrics(
        solution,
        _stage_metrics(kernelization_ms=kernel_ms, gnn_candidate_ms=gnn_ms, ma_ms=ma_ms),
        return_diagnostics,
    )


def gnn_KMA3_solve_undirected(n, edges, pop_size=20, max_gens=100,
                               gnn_threshold=0.65, gnn_hidden_dim=None,
                               gnn_timeout=60,
                               max_time_seconds=600, early_stop=20,
                               return_diagnostics=False):
    """
    GNN-KMA-3: research-grade undirected FVS solver.

    Uses the v3 model with 16-channel features and precision-first soft-hint coupling.
    Falls through to pure KMA if v3 weights are not yet trained.
    """
    if not HAS_CPP_ENGINE:
        raise RuntimeError("cpp_engine not available. Please compile it first.")

    kernel_start = time.perf_counter()
    k_n, k_edges, forced, k_new_to_old = kernelize_undirected_graph(n, edges)
    kernel_ms = (time.perf_counter() - kernel_start) * 1000.0

    if k_n == 0:
        return _maybe_with_metrics(
            sorted(forced),
            _stage_metrics(kernelization_ms=kernel_ms),
            return_diagnostics,
        )

    # For undirected v3 we fall back to v2 undirected probs (same features concept)
    gnn_start = time.perf_counter()
    has_gnn_v3, UNetV3, _ = get_gnn_models_v3()
    weights_v3 = PROJECT_ROOT / "gnn_model" / "weights" / "undirected_fvs_gcn_v3.pt"

    if has_gnn_v3 and get_torch() is not None and weights_v3.exists():
        torch_mod = get_torch()
        try:
            model, _ = _load_model_with_checkpoint(
                UNetV3, weights_v3, directed=False, hidden_dim_override=gnn_hidden_dim
            )
            model.eval()
            gnn_probs_np = _run_gnn_full_inference(
                model, get_undirected_features_v3, k_n, k_edges,
                bidirected=True, label="[GNN-3]", gnn_timeout=gnn_timeout
            )
        except Exception as ex:
            print(f"  [GNN-3] Failed to run v3 model: {ex}. Falling back to v2.")
            gnn_probs_np = run_gnn_undirected_v2_probs(
                k_n, k_edges, hidden_dim=gnn_hidden_dim, gnn_timeout=gnn_timeout
            )
    else:
        gnn_probs_np = run_gnn_undirected_v2_probs(
            k_n, k_edges, hidden_dim=gnn_hidden_dim, gnn_timeout=gnn_timeout
        )
    gnn_ms = (time.perf_counter() - gnn_start) * 1000.0

    ma_start = time.perf_counter()
    kernel_fvs, _ = _soft_hint_gnn_kernel_solve(
        k_n, k_edges, gnn_probs_np, pop_size, max_gens, early_stop, max_time_seconds,
        gnn_threshold=gnn_threshold, max_fix_fraction=0.08,
        label="[KMA-3]", directed=False, run_kma_fn=_kma_run_undirected,
    )
    ma_ms = (time.perf_counter() - ma_start) * 1000.0

    mapped = [k_new_to_old[v] for v in kernel_fvs if 0 <= v < len(k_new_to_old)]
    solution = sorted(set(forced).union(mapped))
    return _maybe_with_metrics(
        solution,
        _stage_metrics(kernelization_ms=kernel_ms, gnn_candidate_ms=gnn_ms, ma_ms=ma_ms),
        return_diagnostics,
    )


# Backward-compatible alias names for legacy imports
# (existing code used gnn_kme_solve_* while implementation is gnn_KMA_solve_*)\ngnn_kme_solve_undirected = gnn_KMA_solve_undirected
gnn_kme_solve_directed = gnn_KMA_solve_directed
gnn_kma2_solve_undirected = gnn_KMA2_solve_undirected
gnn_kma2_solve_directed = gnn_KMA2_solve_directed
gnn_kma3_solve_directed = gnn_KMA3_solve_directed
gnn_kma3_solve_undirected = gnn_KMA3_solve_undirected
kma_solve_undirected_legacy = kma_solve_undirected
kma_solve_directed_legacy = kma_solve_directed
dkma_solve_undirected_legacy = dkma_solve_undirected
dkma_solve_directed_legacy = dkma_solve_directed
gnn_dkma_solve_undirected_legacy = gnn_dkma_solve_undirected
gnn_dkma_solve_directed_legacy = gnn_dkma_solve_directed


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="GNN-KMA GNN + Kernelized Memetic Algorithm FVS Solver (Phase 3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--graph", required=True,
        help="Path to a single graph file (.txt, .gr, .edges)"
    )
    parser.add_argument(
        "--type", default="undirected", choices=["undirected", "directed"],
        help="Graph type: undirected or directed (default: undirected)"
    )
    parser.add_argument(
        "--pop", type=int, default=20,
        help="KMA population size (default: 20)"
    )
    parser.add_argument(
        "--gens", type=int, default=100,
        help="KMA maximum generations (default: 100)"
    )
    parser.add_argument(
        "--timeout", type=int, default=600,
        help="Hard wall-clock timeout in seconds for MA/KMA refinement (default: 600)"
    )
    parser.add_argument(
        "--earlystop", type=int, default=20,
        help="Patience / early-stopping generations without improvement (default: 20)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.65,
        help="GNN probability threshold for FVS candidate selection (default: 0.65, precision-first)"
    )
    parser.add_argument(
        "--gnn-hidden", type=int, default=None,
        help="Optional hidden dimension override for loading GNN weights (default: auto-detect)"
    )
    parser.add_argument(
        "--gnn-timeout", type=int, default=60,
        help="Hard wall-clock timeout in seconds for the GNN phase only (default: 60)"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Also run pure KMA for comparison (shows GNN benefit)"
    )
    parser.add_argument(
        "--mode",
        choices=["GNN-KMA", "GNN-KMA-2", "GNN-KMA-3"],
        default="GNN-KMA",
        help="Hybrid mode: legacy GNN-KMA, advanced GNN-KMA-2, or research-grade GNN-KMA-3",
    )

    args = parser.parse_args()

    if args.timeout <= 0:
        print("ERROR: --timeout must be a positive integer")
        sys.exit(1)
    if args.earlystop <= 0:
        print("ERROR: --earlystop must be a positive integer")
        sys.exit(1)
    if args.gnn_timeout <= 0:
        print("ERROR: --gnn-timeout must be a positive integer")
        sys.exit(1)

    # ── Local imports to avoid circular dependency ─────────────────────────────
    from experiments.benchmark_undirected import parse_graph_file, verify_fvs
    from experiments.benchmark_directed   import parse_directed_graph_file, verify_dfvs

    filepath = Path(args.graph)
    if not filepath.exists():
        print(f"ERROR: File not found: {args.graph}")
        sys.exit(1)

    # ── Parse graph ───────────────────────────────────────────────────────────
    try:
        if args.type == "undirected":
            n, edges = parse_graph_file(str(filepath))
        else:
            n, edges = parse_directed_graph_file(str(filepath))
    except Exception as ex:
        print(f"ERROR: Could not parse {filepath}: {ex}")
        sys.exit(1)

    print(f"\n{'─' * 60}")
    print(f"  File : {filepath.name}")
    print(f"  Type : {args.type}")
    print(f"  Graph: {n} vertices, {len(edges)} edges")
    print(f"{'─' * 60}")

    # ── GNN-KMA run ────────────────────────────────────────────────────────────
    print(f"\n  ── {args.mode} (GNN + KMA) ──")
    
    if not HAS_CPP_ENGINE:
        print("ERROR: cpp_engine not available. Please compile it:")
        print("  cd cpp_engine && mkdir -p build && cd build && cmake .. && make")
        sys.exit(1)
    
    start = time.perf_counter()

    if args.type == "undirected":
        if args.mode == "GNN-KMA-3":
            fvs = gnn_KMA3_solve_undirected(
                n, edges, args.pop, args.gens, args.threshold, args.gnn_hidden,
                gnn_timeout=args.gnn_timeout,
                max_time_seconds=args.timeout, early_stop=args.earlystop,
            )
        elif args.mode == "GNN-KMA-2":
            fvs = gnn_KMA2_solve_undirected(
                n, edges, args.pop, args.gens, args.threshold, args.gnn_hidden,
                gnn_timeout=args.gnn_timeout,
                max_time_seconds=args.timeout, early_stop=args.earlystop,
            )
        else:
            fvs = gnn_KMA_solve_undirected(
                n, edges, args.pop, args.gens, args.threshold, args.gnn_hidden,
                gnn_timeout=args.gnn_timeout,
                max_time_seconds=args.timeout, early_stop=args.earlystop,
            )
        valid = verify_fvs(n, edges, fvs)
    else:
        if args.mode == "GNN-KMA-3":
            fvs = gnn_KMA3_solve_directed(
                n, edges, args.pop, args.gens, args.threshold, args.gnn_hidden,
                gnn_timeout=args.gnn_timeout,
                max_time_seconds=args.timeout, early_stop=args.earlystop,
            )
        elif args.mode == "GNN-KMA-2":
            fvs = gnn_KMA2_solve_directed(
                n, edges, args.pop, args.gens, args.threshold, args.gnn_hidden,
                gnn_timeout=args.gnn_timeout,
                max_time_seconds=args.timeout, early_stop=args.earlystop,
            )
        else:
            fvs = gnn_KMA_solve_directed(
                n, edges, args.pop, args.gens, args.threshold, args.gnn_hidden,
                gnn_timeout=args.gnn_timeout,
                max_time_seconds=args.timeout, early_stop=args.earlystop,
            )
        valid = verify_dfvs(n, edges, fvs)

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    status     = "✓ VALID" if valid else "✗ INVALID"
    print(f"  [RESULT] FVS size = {len(fvs)}  |  Time = {elapsed_ms:.2f} ms  |  {status}")

    # ── Optional comparison with pure KMA ─────────────────────────────────────
    if args.compare:
        print(f"\n  ── Pure KMA (no GNN) ──")
        start = time.perf_counter()

        if args.type == "undirected":
            fvs_ma = kma_solve_undirected(
                n,
                edges,
                pop_size=args.pop,
                max_gens=args.gens,
                max_time_seconds=args.timeout,
                early_stop=args.earlystop,
            )
            valid_ma = verify_fvs(n, edges, fvs_ma)
        else:
            fvs_ma = kma_solve_directed(
                n,
                edges,
                pop_size=args.pop,
                max_gens=args.gens,
                max_time_seconds=args.timeout,
                early_stop=args.earlystop,
            )
            valid_ma = verify_dfvs(n, edges, fvs_ma)

        ms_ma   = (time.perf_counter() - start) * 1000.0
        status_ma = "✓ VALID" if valid_ma else "✗ INVALID"
        print(f"  [RESULT] FVS size = {len(fvs_ma)}  |  Time = {ms_ma:.2f} ms  |  {status_ma}")

        # Print comparison
        print(f"\n  ── Comparison ──")
        print(f"  {args.mode:8s}: {len(fvs):>4} vertices  ({elapsed_ms:.2f} ms)")
        print(f"  Pure KMA: {len(fvs_ma):>4} vertices  ({ms_ma:.2f} ms)")
        diff = len(fvs_ma) - len(fvs)
        if diff > 0:
            print(f"  GNN improvement: -{diff} vertices better than pure KMA ✓")
        elif diff == 0:
            print(f"  Same solution quality.")
        else:
            print(f"  Pure KMA was slightly better on this instance.")


if __name__ == "__main__":
    main()