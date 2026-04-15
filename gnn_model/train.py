"""
train.py
========
Training script for GNN models (v1, v2, v3) for FVS prediction.

Usage:
  python gnn_model/train.py --type undirected --epochs 100 --lr 0.001
  python gnn_model/train.py --type directed   --epochs 100 --hidden 128
  python gnn_model/train.py --type both       --epochs 200 --batch_size 32
  python gnn_model/train.py --type directed   --variant v3 --epochs 200 --hidden 128
    python gnn_model/train.py --type directed   --take-exact 500 --take-heuristic 500

Output:
  gnn_model/weights/undirected_fvs_gcn.pt     (v1)
  gnn_model/weights/directed_fvs_gcn.pt       (v1)
  gnn_model/weights/undirected_fvs_gcn_v2.pt  (v2)
  gnn_model/weights/directed_fvs_gcn_v2.pt    (v2)
  gnn_model/weights/undirected_fvs_gcn_v3.pt  (v3)
  gnn_model/weights/directed_fvs_gcn_v3.pt    (v3)

Training protocol (v3):
  - AsymmetricFVSLoss (fp_gamma=2.0, fn_gamma=0.5) — penalizes FP more than FN
    because hard-fixed GNN false positives inflated FVS size catastrophically.
  - Warmup + cosine LR decay schedule (warmup 10 epochs, cosine to eta_min=1e-5)
  - Gradient clipping (max_norm=1.0) for stable 5-layer GAT training
  - Stratified train/val split by graph family/category
  - Primary validation metric: topk_precision@8% (not F1) for v3
  - Reproducible training via --seed (default: 42)

References:
  - Ben-Baruch et al. (2021) Asymmetric Loss for Multi-Label Classification
    Motivation: FP cost >> FN cost when false positives are hard-fixed
  - He et al. (2016) Deep Residual Learning (residual GATv2 training)
  - Loshchilov & Hutter (2017) SGDR: Warmup + cosine decay
"""

import argparse
import math
import random
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
    from torch_geometric.data import Data, DataLoader as PyGDataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("ERROR: PyTorch and torch_geometric are required for training.")
    print("Install with: pip install torch torch-geometric")
    sys.exit(1)

from gnn_model.model_undirected import UndirectedFVSNet, compute_class_weights
from gnn_model.model_directed   import DirectedFVSNet, compute_class_weights_directed
from gnn_model.model_undirected_v2 import UndirectedFVSNetV2, compute_class_weights_v2
from gnn_model.model_directed_v2 import DirectedFVSNetV2, compute_class_weights_directed_v2


def _log(msg: str) -> None:
    print(msg, flush=True)


def _validate_graph_data(data: Data) -> tuple[bool, str]:
    """Validate a loaded PyG graph and return (is_valid, reason_if_invalid)."""
    x = getattr(data, "x", None)
    y = getattr(data, "y", None)
    edge_index = getattr(data, "edge_index", None)

    if x is None or y is None or edge_index is None:
        return False, "missing x/y/edge_index"
    if not torch.is_tensor(x) or not torch.is_tensor(y) or not torch.is_tensor(edge_index):
        return False, "x/y/edge_index must be tensors"
    if x.dim() != 2:
        return False, "x must be 2D"
    if y.dim() != 1:
        return False, "y must be 1D"
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        return False, "edge_index must have shape [2, m]"

    num_nodes = x.size(0)
    if num_nodes <= 0:
        return False, "empty graph (num_nodes=0)"
    if y.size(0) != num_nodes:
        return False, "label length mismatch"
    if x.numel() == 0:
        return False, "empty x tensor"
    if not torch.isfinite(x).all():
        return False, "non-finite values in x"
    if edge_index.numel() > 0:
        if edge_index.min().item() < 0 or edge_index.max().item() >= num_nodes:
            return False, "edge_index out of bounds"
    if y.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool):
        return False, "y must be integer/bool class labels"

    return True, ""


# ═══════════════════════════════════════════════════════════════════════════════
#  Asymmetric Focal Loss
# ═══════════════════════════════════════════════════════════════════════════════

class AsymmetricFVSLoss(nn.Module):
    """
    Asymmetric focal loss for FVS vertex classification.

    Penalizes false positives (FP) more than false negatives (FN).
    This is motivated by the soft-hint coupling design: a vertex that is
    incorrectly hard-fixed (FP) will permanently inflate the FVS size,
    whereas a missed vertex (FN) can still be found by KMA.

    Loss formulation (per vertex v):
        L(v) = - alpha * (1 - p_t)^gamma_t * log(p_t + eps)
        where:
            p   = P(v ∈ FVS) from sigmoid(logit)
            p_t = p  if y=1 (true positive), (1-p) if y=0 (true negative)
            gamma_t = fn_gamma if y=1 (missed → FN cost)
                    = fp_gamma if y=0 (wrong predict → FP cost)

    Setting fp_gamma > fn_gamma suppresses false-positive predictions.

    Default: fp_gamma=2.0, fn_gamma=0.5
      - A perfectly predicted vertex (p_t → 1) contributes 0 to loss.
      - For a badly predicted vertex:
          FN (y=1, predicted 0): downweight by (1-p)^0.5 → easier case
          FP (y=0, predicted 1): downweight by (1-p)^2   → harder penalty

    Reference: Ben-Baruch et al. (2021) "Asymmetric Loss For Multi-Label
        Classification." ICCV 2021. https://arxiv.org/abs/2009.14119
    """

    def __init__(
        self,
        fp_gamma: float = 2.0,
        fn_gamma: float = 0.5,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.fp_gamma = fp_gamma
        self.fn_gamma = fn_gamma
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (n,) raw logits from model (before sigmoid)
            targets: (n,) binary labels {0, 1}
        Returns:
            Scalar loss value
        """
        probs = torch.sigmoid(logits)
        targets_f = targets.float()

        # p_t: probability of correct class
        p_t = probs * targets_f + (1.0 - probs) * (1.0 - targets_f)
        p_t = torch.clamp(p_t, min=self.eps, max=1.0 - self.eps)

        # Asymmetric focusing factor
        gamma = torch.where(
            targets_f == 1,
            torch.full_like(targets_f, self.fn_gamma),
            torch.full_like(targets_f, self.fp_gamma),
        )

        loss = -((1.0 - p_t) ** gamma) * torch.log(p_t)
        return loss.mean()


def asymmetric_loss_weight_fn(y: torch.Tensor) -> torch.Tensor:
    """Dummy weight fn for v3 (loss handles imbalance internally)."""
    return torch.ones(2)


# ═══════════════════════════════════════════════════════════════════════════════
#  TopK Precision Metric
# ═══════════════════════════════════════════════════════════════════════════════

def compute_topk_precision(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k_fraction: float = 0.08,
) -> float:
    """
    Compute precision among top-k% predicted vertices (by descending probability).

    This is the primary validation metric for v3; it directly measures the
    quality of hard-fixing decisions in the soft-hint coupling.

    Precision@k = |{v: top-k AND v ∈ FVS}| / k

    Args:
        logits:     (n,) raw logits or probabilities
        labels:     (n,) binary ground truth {0, 1}
        k_fraction: fraction of vertices to consider (default 8%)

    Returns:
        precision@k as float in [0, 1]
    """
    n = logits.numel()
    if n == 0:
        return 0.0

    k = max(1, int(math.ceil(n * k_fraction)))
    probs = torch.sigmoid(logits)
    topk_indices = torch.topk(probs, k).indices
    topk_labels = labels[topk_indices]
    precision = topk_labels.float().mean().item()
    return precision


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Loading + Stratified Split
# ═══════════════════════════════════════════════════════════════════════════════

def load_pt_dataset(data_dir: Path) -> list:
    """Load all .pt graph Data objects recursively from a directory."""
    files   = sorted(data_dir.rglob("*.pt"))
    dataset = []
    skipped = 0
    for file_path in files:
        try:
            data = torch.load(file_path, weights_only=False, map_location="cpu")
            ok, reason = _validate_graph_data(data)
            if not ok:
                skipped += 1
                _log(f"  [load-skip] {file_path.name}: {reason}")
                continue
            dataset.append(data)
        except Exception as e:
            skipped += 1
            _log(f"  [load-skip] {file_path.name}: read failure: {e}")

    _log(f"  Loaded {len(dataset)} graphs from {data_dir}")
    if skipped:
        _log(f"  Skipped {skipped} invalid graphs during load")
    return dataset


def _canonical_track_name(track: str) -> str:
    """Normalize track names to exact/heuristic/unknown buckets."""
    t = (track or "").strip().lower()
    if t in {"exact", "exact_track", "exact-track"}:
        return "exact"
    if t in {"heuristic", "heuristic_track", "heuristic-track"}:
        return "heuristic"
    return "unknown"


def sample_dataset_by_track(
    dataset: list,
    take_exact: int | None,
    take_heuristic: int | None,
    seed: int,
) -> list:
    """
    Randomly subsample loaded graphs by label track.

    If a requested take count is None, all graphs from that track are kept.
    Unknown-track graphs are excluded when any take-* flag is used.
    """
    if take_exact is None and take_heuristic is None:
        return dataset

    exact_graphs: list = []
    heuristic_graphs: list = []
    unknown_graphs: list = []

    for g in dataset:
        track = _canonical_track_name(str(getattr(g, "track", "unknown")))
        if track == "exact":
            exact_graphs.append(g)
        elif track == "heuristic":
            heuristic_graphs.append(g)
        else:
            unknown_graphs.append(g)

    rng = random.Random(seed)
    rng.shuffle(exact_graphs)
    rng.shuffle(heuristic_graphs)
    rng.shuffle(unknown_graphs)

    exact_take = len(exact_graphs) if take_exact is None else min(take_exact, len(exact_graphs))
    heur_take = len(heuristic_graphs) if take_heuristic is None else min(take_heuristic, len(heuristic_graphs))

    if take_exact is not None and take_exact > len(exact_graphs):
        _log(
            f"  [take-exact] requested {take_exact}, but only {len(exact_graphs)} exact-track graphs are available"
        )
    if take_heuristic is not None and take_heuristic > len(heuristic_graphs):
        _log(
            "  [take-heuristic] requested "
            f"{take_heuristic}, but only {len(heuristic_graphs)} heuristic-track graphs are available"
        )

    sampled = exact_graphs[:exact_take] + heuristic_graphs[:heur_take]
    rng.shuffle(sampled)

    if unknown_graphs:
        _log(f"  [track-sampling] excluded {len(unknown_graphs)} unknown-track graphs")

    _log(
        "  Applied track sampling: "
        f"exact={exact_take}/{len(exact_graphs)}, "
        f"heuristic={heur_take}/{len(heuristic_graphs)}, "
        f"total={len(sampled)}"
    )
    return sampled


def clean_pt_dataset(base_dir: Path) -> tuple[int, int]:
    """
    Remove unreadable/corrupted PT files and files with NaN/Inf features.

    Returns:
        (scanned_count, removed_count)
    """
    files = sorted(base_dir.rglob("*.pt"))
    removed_count = 0

    _log(f"\n[clean] Scanning {base_dir} for corrupted .pt files...")

    for file_path in files:
        should_remove = False
        reason = ""
        try:
            data = torch.load(file_path, weights_only=False, map_location="cpu")
            ok, invalid_reason = _validate_graph_data(data)
            if not ok:
                should_remove = True
                reason = invalid_reason
        except Exception as e:
            should_remove = True
            reason = f"read failure: {e}"

        if should_remove:
            try:
                file_path.unlink()
                removed_count += 1
                _log(f"[clean] Deleted: {file_path} ({reason})")
            except Exception as e:
                _log(f"[clean] Failed to delete {file_path}: {e}")

    _log(
        f"[clean] Done. Scanned {len(files)} files. Removed {removed_count} bad files."
    )
    return len(files), removed_count


def log_dataset_breakdown(dataset: list) -> None:
    """Print a concise track/category breakdown for loaded PT graphs."""
    if not dataset:
        return

    by_track: dict[str, int] = {}
    by_track_category: dict[tuple[str, str], int] = {}
    for g in dataset:
        track = str(getattr(g, "track", "unknown"))
        category = str(getattr(g, "category", "unknown"))
        by_track[track] = by_track.get(track, 0) + 1
        key = (track, category)
        by_track_category[key] = by_track_category.get(key, 0) + 1

    track_summary = ", ".join(f"{k}:{v}" for k, v in sorted(by_track.items()))
    _log(f"  Track mix: {track_summary}")

    top_pairs = sorted(by_track_category.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
    if top_pairs:
        pair_summary = ", ".join(f"{t}/{c}:{n}" for (t, c), n in top_pairs)
        _log(f"  Top track/category buckets: {pair_summary}")


def stratified_split(
    dataset: list,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list, list]:
    """
    Stratified train/val split by graph family/category.

    Each .pt Data object may have a 'family' attribute (string) set by
    dataset_gen.py indicating the graph category (e.g., "erdos_renyi",
    "barabasi_albert", "pace_benchmark"). Within each family, val_ratio
    fraction is held out for validation.

    Falls back to random split if no family attribute is found.

    Reference: Stratified splits prevent distribution mismatch where the
    validation set only contains easy/small graphs.
    """
    rng = random.Random(seed)

    # Group by family
    families: dict[str, list] = {}
    for g in dataset:
        fam = getattr(g, 'family', 'unknown')
        if fam not in families:
            families[fam] = []
        families[fam].append(g)

    if len(families) == 1 and 'unknown' in families:
        # Fallback: simple random split
        shuffled = list(dataset)
        rng.shuffle(shuffled)
        split = int(len(shuffled) * (1 - val_ratio))
        return shuffled[:split], shuffled[split:]

    train_set, val_set = [], []
    for fam, graphs in families.items():
        rng.shuffle(graphs)
        split = max(1, int(len(graphs) * (1 - val_ratio)))
        train_set.extend(graphs[:split])
        val_set.extend(graphs[split:])

    rng.shuffle(train_set)
    rng.shuffle(val_set)
    return train_set, val_set


def train_val_split(dataset: list, val_ratio: float = 0.2) -> tuple[list, list]:
    """Random train/val split (used for v1/v2 backward compatibility)."""
    return stratified_split(dataset, val_ratio=val_ratio, seed=42)


# ═══════════════════════════════════════════════════════════════════════════════
#  LR Scheduler
# ═══════════════════════════════════════════════════════════════════════════════

def get_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    eta_min_ratio: float = 0.01,
) -> LambdaLR:
    """
    Linear warmup followed by cosine annealing LR schedule.

    Phase 1 (epoch 0..warmup_epochs-1): lr = base_lr * (epoch / warmup_epochs)
    Phase 2 (epoch warmup_epochs..total_epochs): cosine decay to eta_min_ratio * base_lr

    Reference: Loshchilov & Hutter (2017) SGDR: Stochastic Gradient Descent
        with Warm Restarts. ICLR 2017.
    """
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(1, warmup_epochs)
        # Cosine decay from 1.0 to eta_min_ratio
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_min_ratio + (1.0 - eta_min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# ═══════════════════════════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(model, dataset, device, weight_fn, is_v3=False):
    """Compute loss, accuracy, precision, recall, F1, and (for v3) topk_precision."""
    model.eval()
    total_loss = 0.0
    tp = fp = fn = tn = 0
    topk_sum = 0.0
    topk_n = 0

    criterion_v3 = AsymmetricFVSLoss(fp_gamma=2.0, fn_gamma=0.5)

    with torch.no_grad():
        for data in dataset:
            x          = data.x.to(device)
            edge_index = data.edge_index.to(device)
            y          = data.y.to(device)

            if x.size(0) == 0 or y.numel() == 0:
                continue

            if is_v3:
                logits = model(x, edge_index)
                if not torch.isfinite(logits).all():
                    continue
                loss = criterion_v3(logits, y)
                if not torch.isfinite(loss):
                    continue
                total_loss += loss.item()
                preds = (torch.sigmoid(logits) >= 0.5).long()
                # TopK precision
                topk_sum += compute_topk_precision(logits, y, k_fraction=0.08)
                topk_n += 1
            else:
                logits  = model(x, edge_index)
                if not torch.isfinite(logits).all():
                    continue
                weights = weight_fn(y).to(device)
                loss    = nn.NLLLoss(weight=weights)(logits, y)
                if not torch.isfinite(loss):
                    continue
                total_loss += loss.item()
                preds = logits.argmax(dim=1)

            tp += ((preds == 1) & (y == 1)).sum().item()
            fp += ((preds == 1) & (y == 0)).sum().item()
            fn += ((preds == 0) & (y == 1)).sum().item()
            tn += ((preds == 0) & (y == 0)).sum().item()

    n       = len(dataset)
    avg_loss = total_loss / max(n, 1)
    prec     = tp / max(tp + fp, 1)
    recall   = tp / max(tp + fn, 1)
    f1       = 2 * prec * recall / max(prec + recall, 1e-8)
    acc      = (tp + tn) / max(tp + fp + fn + tn, 1)
    topk_prec = topk_sum / max(topk_n, 1)

    return {
        "loss": avg_loss,
        "acc": acc,
        "precision": prec,
        "recall": recall,
        "f1": f1,
        "topk_precision": topk_prec,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Training Loop
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(
    model,
    train_set,
    val_set,
    weight_fn,
    epochs: int,
    lr: float,
    device,
    save_path: Path,
    log_every: int,
    is_v3: bool = False,
    warmup_epochs: int = 10,
    max_grad_norm: float = 1.0,
    seed: int = 42,
):
    """
    Main training loop.

    For v3:
      - AsymmetricFVSLoss (fp_gamma=2.0, fn_gamma=0.5)
      - Warmup + cosine LR schedule
      - Gradient clipping (max_norm=1.0)
      - Checkpoint by topk_precision@8% (primary metric)

    For v1/v2:
      - Weighted NLL loss
      - CosineAnnealingLR
      - Checkpoint by validation F1
    """
    torch.manual_seed(seed)

    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    if is_v3:
        scheduler = get_warmup_cosine_scheduler(optimizer, warmup_epochs, epochs)
        criterion_v3 = AsymmetricFVSLoss(fp_gamma=2.0, fn_gamma=0.5)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        criterion_v3 = None

    best_metric   = -1.0
    patience      = 20
    patience_ctr  = 0
    save_path.parent.mkdir(parents=True, exist_ok=True)

    metric_name = "topk_precision" if is_v3 else "f1"

    _log(f"\n  Training: {len(train_set)} graphs  |  Val: {len(val_set)} graphs")
    _log(f"  Device  : {device}  |  Is_V3: {is_v3}")
    _log(f"  Epochs  : {epochs}  |  LR: {lr}  |  Warmup: {warmup_epochs if is_v3 else 0}")
    _log(f"  Primary metric: {metric_name}")
    _log(f"  {'Epoch':>6}  {'TrainLoss':>10}  {'ValLoss':>10}  "
         f"{'ValF1':>8}  {'topkPrec':>10}  {'ValAcc':>8}")
    _log("  " + "─" * 62)

    log_every = max(1, log_every)

    for epoch in range(1, epochs + 1):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        total_train_loss = 0.0
        valid_batches = 0
        skipped_batches = 0
        recovered_batches = 0

        for data in train_set:
            # Stabilize tensor dtypes/ranges before forward to avoid NaN/Inf cascades.
            x = data.x.float()
            x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
            x = torch.clamp(x, min=-1e4, max=1e4).to(device)

            edge_index = data.edge_index.long().to(device)
            y = data.y.view(-1).long().to(device)

            # Enforce binary labels for v3 loss; some generated graphs may carry
            # non-binary integer labels that destabilize asymmetric loss.
            if is_v3:
                y = (y > 0).long()

            if x.size(0) == 0 or y.numel() == 0:
                skipped_batches += 1
                continue

            optimizer.zero_grad()

            if is_v3:
                logits = model(x, edge_index)
                if not torch.isfinite(logits).all():
                    recovered_batches += 1
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
                logits = torch.clamp(logits, min=-20.0, max=20.0)
                loss   = criterion_v3(logits, y)
            else:
                logits  = model(x, edge_index)
                if not torch.isfinite(logits).all():
                    recovered_batches += 1
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
                logits = torch.clamp(logits, min=-20.0, max=20.0)
                weights = weight_fn(y).to(device)
                loss    = nn.NLLLoss(weight=weights)(logits, y)

            if not torch.isfinite(loss):
                recovered_batches += 1
                loss = torch.nan_to_num(loss, nan=0.0, posinf=1.0, neginf=1.0)

            loss.backward()

            # Guard against exploding/invalid gradients propagating NaNs.
            grad_is_finite = True
            for p in model.parameters():
                if p.grad is not None and (not torch.isfinite(p.grad).all()):
                    grad_is_finite = False
                    break
            if not grad_is_finite:
                recovered_batches += 1
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=1.0, neginf=-1.0)

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            optimizer.step()
            total_train_loss += loss.item()
            valid_batches += 1

        scheduler.step()

        if valid_batches == 0:
            _log("  ERROR: all training batches were invalid/non-finite this epoch; aborting early.")
            break

        # Heartbeat log
        if epoch % log_every == 0 or epoch == epochs:
            train_loss = total_train_loss / max(valid_batches, 1)
            pct = 100.0 * epoch / max(epochs, 1)
            cur_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else lr
            _log(f"  [progress] epoch {epoch}/{epochs} ({pct:.1f}%) "
                 f"train_loss={train_loss:.4f} lr={cur_lr:.7f} "
                  f"valid_batches={valid_batches} skipped={skipped_batches} "
                  f"recovered={recovered_batches}")

        # ── Validate every 5 epochs ───────────────────────────────────────────
        if epoch % 5 == 0 or epoch == epochs:
            train_loss = total_train_loss / max(valid_batches, 1)
            val_m      = compute_metrics(model, val_set, device, weight_fn, is_v3=is_v3)

            _log(f"  {epoch:>6}  {train_loss:>10.4f}  {val_m['loss']:>10.4f}"
                 f"  {val_m['f1']:>8.4f}  {val_m['topk_precision']:>10.4f}"
                 f"  {val_m['acc']:>8.4f}")

            # ── Save best model ───────────────────────────────────────────────
            current_metric = val_m[metric_name]
            if current_metric > best_metric:
                best_metric = current_metric
                torch.save(model.state_dict(), save_path)
                patience_ctr = 0
                _log(f"  ✓ Saved best model ({metric_name}={best_metric:.4f})")
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    _log(f"\n  Early stopping at epoch {epoch} "
                         f"(no improvement in {patience} checks)")
                    break

    _log(f"\n  Best Val {metric_name}: {best_metric:.4f}")
    _log(f"  Model saved: {save_path}")
    return best_metric


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train GNN for FVS vertex prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--type",    default="both", choices=["undirected", "directed", "both"])
    parser.add_argument("--epochs",  type=int,   default=100)
    parser.add_argument("--lr",      type=float, default=0.001)
    parser.add_argument("--hidden",  type=int,   default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--variant", choices=["v1", "v2", "v3"], default="v1",
        help="Model variant: v1 (base GCN), v2 (enriched features GCN), v3 (GAT+residual)"
    )
    parser.add_argument(
        "--v3", action="store_true",
        help="Shorthand for --variant v3: use research-grade GAT architecture"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (data split, model init)"
    )
    parser.add_argument(
        "--warmup-epochs", type=int, default=10,
        help="[v3 only] Linear LR warmup epochs before cosine decay"
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=1.0,
        help="[v3 only] Gradient clipping max norm"
    )
    parser.add_argument(
        "--data-root", type=str,
        default=str(PROJECT_ROOT / "gnn_model" / "datasets" / "pt"),
        help="Path to .pt dataset root directory"
    )
    parser.add_argument(
        "--take-exact", type=int, default=None,
        help="Randomly keep only N exact-track graphs from the loaded dataset"
    )
    parser.add_argument(
        "--take-heuristic", type=int, default=None,
        help="Randomly keep only M heuristic-track graphs from the loaded dataset"
    )
    args = parser.parse_args()

    # --v3 flag is shorthand for --variant v3
    if args.v3:
        args.variant = "v3"

    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be in (0, 1)")
    if args.take_exact is not None and args.take_exact < 0:
        raise ValueError("--take-exact must be >= 0")
    if args.take_heuristic is not None and args.take_heuristic < 0:
        raise ValueError("--take-heuristic must be >= 0")

    # Set global seeds for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_v3    = (args.variant == "v3")

    # Resolve data directory
    data_dir = Path(args.data_root)
    if args.data_root == str(PROJECT_ROOT / "gnn_model" / "datasets" / "pt"):
        if args.variant == "v2":
            data_dir = PROJECT_ROOT / "gnn_model" / "datasets" / "pt_v2"
        elif args.variant == "v3":
            data_dir = PROJECT_ROOT / "gnn_model" / "datasets" / "pt_v3"

    wt_dir = PROJECT_ROOT / "gnn_model" / "weights"

    # Always scrub dataset files before training to avoid NaN/Inf crashes.
    clean_pt_dataset(data_dir)

    # Model + weight file selection
    if args.variant == "v3":
        try:
            from gnn_model.model_directed_v3 import (
                DirectedFVSNetV3, UndirectedFVSNetV3, compute_class_weights_v3
            )
        except ImportError as e:
            _log(f"ERROR: Could not import v3 models: {e}")
            sys.exit(1)
        und_model_cls   = UndirectedFVSNetV3
        dir_model_cls   = DirectedFVSNetV3
        und_weight_fn   = asymmetric_loss_weight_fn
        dir_weight_fn   = asymmetric_loss_weight_fn
        und_weight_path = wt_dir / "undirected_fvs_gcn_v3.pt"
        dir_weight_path = wt_dir / "directed_fvs_gcn_v3.pt"
    elif args.variant == "v2":
        und_model_cls   = UndirectedFVSNetV2
        dir_model_cls   = DirectedFVSNetV2
        und_weight_fn   = compute_class_weights_v2
        dir_weight_fn   = compute_class_weights_directed_v2
        und_weight_path = wt_dir / "undirected_fvs_gcn_v2.pt"
        dir_weight_path = wt_dir / "directed_fvs_gcn_v2.pt"
    else:
        und_model_cls   = UndirectedFVSNet
        dir_model_cls   = DirectedFVSNet
        und_weight_fn   = compute_class_weights
        dir_weight_fn   = compute_class_weights_directed
        und_weight_path = wt_dir / "undirected_fvs_gcn.pt"
        dir_weight_path = wt_dir / "directed_fvs_gcn.pt"

    if args.type in ("undirected", "both"):
        _log("\n" + "═" * 60)
        _log(f"  Training UNDIRECTED FVS GCN ({args.variant})")
        _log("═" * 60)
        dataset = load_pt_dataset(data_dir / "undirected")
        if not dataset:
            _log("  No data found. Run dataset_gen.py first.")
        else:
            dataset = sample_dataset_by_track(
                dataset,
                take_exact=args.take_exact,
                take_heuristic=args.take_heuristic,
                seed=args.seed,
            )
            if not dataset:
                _log("  No data left after track sampling; skipping undirected training.")
            else:
                log_dataset_breakdown(dataset)
                train_set, val_set = stratified_split(
                    dataset, val_ratio=args.val_ratio, seed=args.seed
                )
                model = und_model_cls(
                    hidden_dim=args.hidden,
                    dropout=args.dropout,
                    **({"in_channels": 16} if is_v3 else {}),
                )
                train_model(
                    model, train_set, val_set, und_weight_fn,
                    epochs=args.epochs, lr=args.lr, device=device,
                    save_path=und_weight_path, log_every=args.log_every,
                    is_v3=is_v3,
                    warmup_epochs=args.warmup_epochs,
                    max_grad_norm=args.max_grad_norm,
                    seed=args.seed,
                )

    if args.type in ("directed", "both"):
        _log("\n" + "═" * 60)
        _log(f"  Training DIRECTED FVS DiGCN ({args.variant})")
        _log("═" * 60)
        dataset = load_pt_dataset(data_dir / "directed")
        if not dataset:
            _log("  No data found. Run dataset_gen.py first.")
        else:
            dataset = sample_dataset_by_track(
                dataset,
                take_exact=args.take_exact,
                take_heuristic=args.take_heuristic,
                seed=args.seed,
            )
            if not dataset:
                _log("  No data left after track sampling; skipping directed training.")
            else:
                log_dataset_breakdown(dataset)
                train_set, val_set = stratified_split(
                    dataset, val_ratio=args.val_ratio, seed=args.seed
                )
                model = dir_model_cls(
                    hidden_dim=args.hidden,
                    dropout=args.dropout,
                    **({"in_channels": 16} if is_v3 else {}),
                )
                train_model(
                    model, train_set, val_set, dir_weight_fn,
                    epochs=args.epochs, lr=args.lr, device=device,
                    save_path=dir_weight_path, log_every=args.log_every,
                    is_v3=is_v3,
                    warmup_epochs=args.warmup_epochs,
                    max_grad_norm=args.max_grad_norm,
                    seed=args.seed,
                )


if __name__ == "__main__":
    main()