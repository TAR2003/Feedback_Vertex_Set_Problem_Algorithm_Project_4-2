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