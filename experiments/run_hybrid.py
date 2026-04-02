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

def get_undirected_features(n, edges):
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
        d = degs.get(v, 0)
        features.append([
            d / max(n - 1, 1),
            clust.get(v, 0.0),
            math.log(d + 1) / math.log(n + 1)
        ])
    return features


def get_directed_features(n, edges):
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
        ind  = in_deg[v]
        outd = out_deg[v]
        features.append([
            ind  / max(n - 1, 1),
            outd / max(n - 1, 1),
            min(ind, outd) / max(n - 1, 1)
        ])
    return features


def get_undirected_features_v2(n, edges):
    """Advanced undirected structural features used by GNN-KMA-2."""
    if not HAS_NX:
        return get_undirected_features(n, edges)
    return compute_node_features_undirected_v2(n, edges)


def get_directed_features_v2(n, edges):
    """Advanced directed structural features used by GNN-KMA-2."""
    if not HAS_NX:
        return get_directed_features(n, edges)
    return compute_node_features_directed_v2(n, edges)


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
    threshold=0.2,
    min_fraction=0.01,
    max_fraction=0.15,
):
    """
    Select robust GNN candidates from per-vertex positive probabilities.

    Strategy:
      1) Primary: vertices with p >= threshold.
      2) Fallback: if empty, take top-k by probability (k ~= min_fraction * n).
      3) Safety cap: if too many pass, keep only top-k by probability
         (k ~= max_fraction * n).

    Returns:
      (candidate_set, mode_string)
    """
    torch = get_torch()
    if torch is None:
        return set(), "unavailable"

    n = int(positive_probs.numel())
    if n == 0:
        return set(), "empty-graph"

    min_k = max(1, int(math.ceil(n * max(0.0, min_fraction))))
    max_k = max(min_k, int(math.ceil(n * max(0.0, max_fraction))))

    selected = (positive_probs >= threshold).nonzero(as_tuple=True)[0]
    mode = "threshold"

    if selected.numel() == 0:
        k = min(min_k, n)
        selected = torch.topk(positive_probs, k).indices
        mode = "topk-fallback"
    elif selected.numel() > max_k:
        k = min(max_k, n)
        selected = torch.topk(positive_probs, k).indices
        mode = "threshold-capped"

    return set(selected.tolist()), mode


# ═══════════════════════════════════════════════════════════════════════════════
#  GNN Inference
# ═══════════════════════════════════════════════════════════════════════════════

def run_gnn_undirected(n, edges, threshold=0.2, hidden_dim=None):
    """
    Run undirected GNN. Returns set of predicted FVS vertex indices.
    Returns None if GNN is unavailable.
    """
    torch = get_torch()
    has_gnn, UNet, _ = get_gnn_models()
    weights_path = PROJECT_ROOT / "gnn_model" / "weights" / "undirected_fvs_gcn.pt"

    if not has_gnn or torch is None:
        print("  [GNN] PyTorch/GNN not available. Skipping GNN step.")
        return None

    if not weights_path.exists():
        print(f"  [GNN] Weights not found at {weights_path}. Skipping GNN step.")
        print("        Run: python gnn_model/train.py --type undirected")
        return None

    try:
        model, used_hidden = _load_model_with_checkpoint(
            UNet,
            weights_path,
            directed=False,
            hidden_dim_override=hidden_dim,
        )
        model.eval()

        feats = get_undirected_features(n, edges)
        x = torch.tensor(feats, dtype=torch.float)
        ei = make_edge_index(edges, n, bidirected=True)

        with torch.no_grad():
            logits = model(x, ei)
            pos_probs = logits.exp()[:, 1]
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


def run_gnn_directed(n, edges, threshold=0.2, hidden_dim=None):
    """
    Run directed GNN (DiGCN). Returns set of predicted DFVS vertex indices.
    Returns None if GNN is unavailable.
    """
    torch = get_torch()
    has_gnn, _, DNet = get_gnn_models()
    weights_path = PROJECT_ROOT / "gnn_model" / "weights" / "directed_fvs_gcn.pt"

    if not has_gnn or torch is None:
        print("  [GNN] PyTorch/GNN not available. Skipping GNN step.")
        return None

    if not weights_path.exists():
        print(f"  [GNN] Weights not found at {weights_path}. Skipping GNN step.")
        print("        Run: python gnn_model/train.py --type directed")
        return None

    try:
        model, used_hidden = _load_model_with_checkpoint(
            DNet,
            weights_path,
            directed=True,
            hidden_dim_override=hidden_dim,
        )
        model.eval()

        feats = get_directed_features(n, edges)
        x = torch.tensor(feats, dtype=torch.float)
        ei = make_edge_index(edges, n, bidirected=False)

        with torch.no_grad():
            logits = model(x, ei)
            pos_probs = logits.exp()[:, 1]
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


def run_gnn_undirected_v2(n, edges, threshold=0.2, hidden_dim=None):
    """Run undirected GNN-KMA-2 model with advanced structural features."""
    torch = get_torch()
    has_gnn_v2, UNetV2, _ = get_gnn_models_v2()
    weights_path = PROJECT_ROOT / "gnn_model" / "weights" / "undirected_fvs_gcn_v2.pt"

    if not has_gnn_v2 or torch is None:
        print("  [GNN-2] PyTorch/GNN not available. Skipping GNN step.")
        return None

    if not weights_path.exists():
        print(f"  [GNN-2] Weights not found at {weights_path}. Skipping GNN step.")
        print("          Run: python gnn_model/train.py --type undirected --variant v2")
        return None

    try:
        model, used_hidden = _load_model_with_checkpoint(
            UNetV2,
            weights_path,
            directed=False,
            hidden_dim_override=hidden_dim,
        )
        model.eval()

        feats = get_undirected_features_v2(n, edges)
        x = torch.tensor(feats, dtype=torch.float)
        ei = make_edge_index(edges, n, bidirected=True)

        with torch.no_grad():
            logits = model(x, ei)
            pos_probs = logits.exp()[:, 1]
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


def run_gnn_directed_v2(n, edges, threshold=0.2, hidden_dim=None):
    """Run directed GNN-KMA-2 model with advanced structural features."""
    torch = get_torch()
    has_gnn_v2, _, DNetV2 = get_gnn_models_v2()
    weights_path = PROJECT_ROOT / "gnn_model" / "weights" / "directed_fvs_gcn_v2.pt"

    if not has_gnn_v2 or torch is None:
        print("  [GNN-2] PyTorch/GNN not available. Skipping GNN step.")
        return None

    if not weights_path.exists():
        print(f"  [GNN-2] Weights not found at {weights_path}. Skipping GNN step.")
        print("          Run: python gnn_model/train.py --type directed --variant v2")
        return None

    try:
        model, used_hidden = _load_model_with_checkpoint(
            DNetV2,
            weights_path,
            directed=True,
            hidden_dim_override=hidden_dim,
        )
        model.eval()

        feats = get_directed_features_v2(n, edges)
        x = torch.tensor(feats, dtype=torch.float)
        ei = make_edge_index(edges, n, bidirected=False)

        with torch.no_grad():
            logits = model(x, ei)
            pos_probs = logits.exp()[:, 1]
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


# ═══════════════════════════════════════════════════════════════════════════════
#  GNN-KMA Solver
# ═══════════════════════════════════════════════════════════════════════════════

def gnn_KMA_solve_undirected(n, edges, pop_size=60, max_gens=300,
                            gnn_threshold=0.2, gnn_hidden_dim=None):
    """
    GNN-KMA: kernelize → GNN on kernel → KMA refinement for undirected FVS.

    GNN predictions are used as a warm-start hint to MA.  If GNN is
    unavailable, falls back to pure MA.
    """
    if not HAS_CPP_ENGINE:
        raise RuntimeError("cpp_engine not available. Please compile it first.")
    
    # Step 1: Kernelization first, then run GNN on the irreducible core.
    k_n, k_edges, forced, k_new_to_old = kernelize_undirected_graph(n, edges)

    if k_n == 0:
        return forced

    gnn_candidates = run_gnn_undirected(
        k_n, k_edges, threshold=gnn_threshold, hidden_dim=gnn_hidden_dim
    )

    if gnn_candidates is None:
        print(f"  [KMA] GNN unavailable, running pure KMA (pop={pop_size}, gens={max_gens})")
        fixed_kernel = set()
        reduced_n = k_n
        reduced_edges = k_edges
        reduced_to_kernel = list(range(k_n))
    else:
        fixed_kernel = {v for v in gnn_candidates if 0 <= v < k_n}
        if fixed_kernel:
            print(
                f"  [KMA] Using GNN-guided kernel core "
                f"(fixed={len(fixed_kernel)}, pop={pop_size}, gens={max_gens})"
            )
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
            print(f"  [KMA] GNN produced no fixed hints, running standard KMA (pop={pop_size}, gens={max_gens})")
            reduced_n = k_n
            reduced_edges = k_edges
            reduced_to_kernel = list(range(k_n))

    # Step 2: KMA refinement on the reduced kernel graph.
    if reduced_n > 0:
        if hasattr(cpp_engine, "solve_undirected_KMA"):
            reduced_fvs = cpp_engine.solve_undirected_KMA(reduced_n, reduced_edges, pop_size, max_gens)
        elif hasattr(cpp_engine, "solve_undirected_KMA"):
            reduced_fvs = cpp_engine.solve_undirected_KMA(reduced_n, reduced_edges, pop_size, max_gens)
        else:
            reduced_fvs = cpp_engine.solve_undirected_MA(reduced_n, reduced_edges, pop_size, max_gens)
    else:
        reduced_fvs = []

    kernel_fvs = set(fixed_kernel)
    kernel_fvs.update(
        reduced_to_kernel[v] for v in reduced_fvs if 0 <= v < len(reduced_to_kernel)
    )
    mapped = [k_new_to_old[v] for v in kernel_fvs if 0 <= v < len(k_new_to_old)]
    return sorted(set(forced).union(mapped))


def gnn_KMA_solve_directed(n, edges, pop_size=60, max_gens=300,
                          gnn_threshold=0.2, gnn_hidden_dim=None):
    """
    GNN-KMA: kernelize → GNN on kernel → KMA refinement for directed FVS.
    """
    if not HAS_CPP_ENGINE:
        raise RuntimeError("cpp_engine not available. Please compile it first.")
    
    k_n, k_edges, forced, k_new_to_old = kernelize_directed_graph(n, edges)

    if k_n == 0:
        return forced

    gnn_candidates = run_gnn_directed(
        k_n, k_edges, threshold=gnn_threshold, hidden_dim=gnn_hidden_dim
    )

    if gnn_candidates is None:
        print(f"  [KMA] GNN unavailable, running pure KMA (pop={pop_size}, gens={max_gens})")
        fixed_kernel = set()
        reduced_n = k_n
        reduced_edges = k_edges
        reduced_to_kernel = list(range(k_n))
    else:
        fixed_kernel = {v for v in gnn_candidates if 0 <= v < k_n}
        if fixed_kernel:
            print(
                f"  [KMA] Using GNN-guided kernel core "
                f"(fixed={len(fixed_kernel)}, pop={pop_size}, gens={max_gens})"
            )
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
            print(f"  [KMA] GNN produced no fixed hints, running standard KMA (pop={pop_size}, gens={max_gens})")
            reduced_n = k_n
            reduced_edges = k_edges
            reduced_to_kernel = list(range(k_n))

    if reduced_n > 0:
        if hasattr(cpp_engine, "solve_directed_KMA"):
            reduced_fvs = cpp_engine.solve_directed_KMA(reduced_n, reduced_edges, pop_size, max_gens)
        elif hasattr(cpp_engine, "solve_directed_KMA"):
            reduced_fvs = cpp_engine.solve_directed_KMA(reduced_n, reduced_edges, pop_size, max_gens)
        else:
            reduced_fvs = cpp_engine.solve_directed_MA(reduced_n, reduced_edges, pop_size, max_gens)
    else:
        reduced_fvs = []

    kernel_fvs = set(fixed_kernel)
    kernel_fvs.update(
        reduced_to_kernel[v] for v in reduced_fvs if 0 <= v < len(reduced_to_kernel)
    )
    mapped = [k_new_to_old[v] for v in kernel_fvs if 0 <= v < len(k_new_to_old)]
    return sorted(set(forced).union(mapped))


def gnn_KMA2_solve_undirected(n, edges, pop_size=60, max_gens=300,
                              gnn_threshold=0.2, gnn_hidden_dim=None):
    """
    GNN-KMA-2: kernelize -> GNN-v2 on kernel -> KMA refinement for undirected FVS.
    """
    if not HAS_CPP_ENGINE:
        raise RuntimeError("cpp_engine not available. Please compile it first.")

    k_n, k_edges, forced, k_new_to_old = kernelize_undirected_graph(n, edges)
    if k_n == 0:
        return forced

    gnn_candidates = run_gnn_undirected_v2(
        k_n, k_edges, threshold=gnn_threshold, hidden_dim=gnn_hidden_dim
    )

    if gnn_candidates is None:
        print(f"  [KMA-2] GNN unavailable, running pure KMA (pop={pop_size}, gens={max_gens})")
        fixed_kernel = set()
        reduced_n = k_n
        reduced_edges = k_edges
        reduced_to_kernel = list(range(k_n))
    else:
        fixed_kernel = {v for v in gnn_candidates if 0 <= v < k_n}
        if fixed_kernel:
            print(
                f"  [KMA-2] Using GNN-guided kernel core "
                f"(fixed={len(fixed_kernel)}, pop={pop_size}, gens={max_gens})"
            )
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
            print(f"  [KMA-2] GNN produced no fixed hints, running standard KMA (pop={pop_size}, gens={max_gens})")
            reduced_n = k_n
            reduced_edges = k_edges
            reduced_to_kernel = list(range(k_n))

    if reduced_n > 0:
        if hasattr(cpp_engine, "solve_undirected_KMA"):
            reduced_fvs = cpp_engine.solve_undirected_KMA(reduced_n, reduced_edges, pop_size, max_gens)
        elif hasattr(cpp_engine, "solve_undirected_KMA"):
            reduced_fvs = cpp_engine.solve_undirected_KMA(reduced_n, reduced_edges, pop_size, max_gens)
        else:
            reduced_fvs = cpp_engine.solve_undirected_MA(reduced_n, reduced_edges, pop_size, max_gens)
    else:
        reduced_fvs = []

    kernel_fvs = set(fixed_kernel)
    kernel_fvs.update(
        reduced_to_kernel[v] for v in reduced_fvs if 0 <= v < len(reduced_to_kernel)
    )
    mapped = [k_new_to_old[v] for v in kernel_fvs if 0 <= v < len(k_new_to_old)]
    return sorted(set(forced).union(mapped))


def gnn_KMA2_solve_directed(n, edges, pop_size=60, max_gens=300,
                            gnn_threshold=0.2, gnn_hidden_dim=None):
    """
    GNN-KMA-2: kernelize -> GNN-v2 on kernel -> KMA refinement for directed FVS.
    """
    if not HAS_CPP_ENGINE:
        raise RuntimeError("cpp_engine not available. Please compile it first.")

    k_n, k_edges, forced, k_new_to_old = kernelize_directed_graph(n, edges)
    if k_n == 0:
        return forced

    gnn_candidates = run_gnn_directed_v2(
        k_n, k_edges, threshold=gnn_threshold, hidden_dim=gnn_hidden_dim
    )

    if gnn_candidates is None:
        print(f"  [KMA-2] GNN unavailable, running pure KMA (pop={pop_size}, gens={max_gens})")
        fixed_kernel = set()
        reduced_n = k_n
        reduced_edges = k_edges
        reduced_to_kernel = list(range(k_n))
    else:
        fixed_kernel = {v for v in gnn_candidates if 0 <= v < k_n}
        if fixed_kernel:
            print(
                f"  [KMA-2] Using GNN-guided kernel core "
                f"(fixed={len(fixed_kernel)}, pop={pop_size}, gens={max_gens})"
            )
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
            print(f"  [KMA-2] GNN produced no fixed hints, running standard KMA (pop={pop_size}, gens={max_gens})")
            reduced_n = k_n
            reduced_edges = k_edges
            reduced_to_kernel = list(range(k_n))

    if reduced_n > 0:
        if hasattr(cpp_engine, "solve_directed_KMA"):
            reduced_fvs = cpp_engine.solve_directed_KMA(reduced_n, reduced_edges, pop_size, max_gens)
        elif hasattr(cpp_engine, "solve_directed_KMA"):
            reduced_fvs = cpp_engine.solve_directed_KMA(reduced_n, reduced_edges, pop_size, max_gens)
        else:
            reduced_fvs = cpp_engine.solve_directed_MA(reduced_n, reduced_edges, pop_size, max_gens)
    else:
        reduced_fvs = []

    kernel_fvs = set(fixed_kernel)
    kernel_fvs.update(
        reduced_to_kernel[v] for v in reduced_fvs if 0 <= v < len(reduced_to_kernel)
    )
    mapped = [k_new_to_old[v] for v in kernel_fvs if 0 <= v < len(k_new_to_old)]
    return sorted(set(forced).union(mapped))


# Backward-compatible alias names for legacy imports
# (existing code used gnn_kme_solve_* while implementation is gnn_KMA_solve_*)
gnn_kme_solve_undirected = gnn_KMA_solve_undirected
gnn_kme_solve_directed = gnn_KMA_solve_directed
gnn_kma2_solve_undirected = gnn_KMA2_solve_undirected
gnn_kma2_solve_directed = gnn_KMA2_solve_directed


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
        "--pop", type=int, default=60,
        help="KMA population size (default: 60)"
    )
    parser.add_argument(
        "--gens", type=int, default=300,
        help="KMA maximum generations (default: 300)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.2,
        help="GNN probability threshold for FVS candidate selection (default: 0.2)"
    )
    parser.add_argument(
        "--gnn-hidden", type=int, default=None,
        help="Optional hidden dimension override for loading GNN weights (default: auto-detect)"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Also run pure KMA for comparison (shows GNN benefit)"
    )
    parser.add_argument(
        "--mode",
        choices=["GNN-KMA", "GNN-KMA-2"],
        default="GNN-KMA",
        help="Hybrid mode: legacy GNN-KMA or advanced GNN-KMA-2",
    )

    args = parser.parse_args()

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
        if args.mode == "GNN-KMA-2":
            fvs = gnn_KMA2_solve_undirected(
                n, edges, args.pop, args.gens, args.threshold, args.gnn_hidden
            )
        else:
            fvs = gnn_KMA_solve_undirected(
                n, edges, args.pop, args.gens, args.threshold, args.gnn_hidden
            )
        valid = verify_fvs(n, edges, fvs)
    else:
        if args.mode == "GNN-KMA-2":
            fvs = gnn_KMA2_solve_directed(
                n, edges, args.pop, args.gens, args.threshold, args.gnn_hidden
            )
        else:
            fvs = gnn_KMA_solve_directed(
                n, edges, args.pop, args.gens, args.threshold, args.gnn_hidden
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
            if hasattr(cpp_engine, "solve_undirected_KMA"):
                fvs_ma = cpp_engine.solve_undirected_KMA(n, edges, args.pop, args.gens)
            elif hasattr(cpp_engine, "solve_undirected_KMA"):
                fvs_ma = cpp_engine.solve_undirected_KMA(n, edges, args.pop, args.gens)
            else:
                fvs_ma = cpp_engine.solve_undirected_MA(n, edges, args.pop, args.gens)
            valid_ma = verify_fvs(n, edges, fvs_ma)
        else:
            if hasattr(cpp_engine, "solve_directed_KMA"):
                fvs_ma = cpp_engine.solve_directed_KMA(n, edges, args.pop, args.gens)
            elif hasattr(cpp_engine, "solve_directed_KMA"):
                fvs_ma = cpp_engine.solve_directed_KMA(n, edges, args.pop, args.gens)
            else:
                fvs_ma = cpp_engine.solve_directed_MA(n, edges, args.pop, args.gens)
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