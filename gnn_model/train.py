"""
train.py
========
Training script for both undirected and directed GNN models.

Usage:
  python gnn_model/train.py --type undirected --epochs 100 --lr 0.001
  python gnn_model/train.py --type directed   --epochs 100 --hidden 128
  python gnn_model/train.py --type both       --epochs 200 --batch_size 32

Output:
  gnn_model/weights/undirected_fvs_gcn.pt
  gnn_model/weights/directed_fvs_gcn.pt

Training protocol:
  - Adam optimizer with cosine LR schedule
  - Weighted cross-entropy loss (handles class imbalance)
  - 80/20 train/val split
  - Early stopping (patience = 20 epochs)
  - Save best model by validation F1 score on the FVS class
"""

import argparse
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    import torch.nn as nn
    from torch.optim import Adam
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch_geometric.data import Data, DataLoader as PyGDataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("ERROR: PyTorch and torch_geometric are required for training.")
    print("Install with: pip install torch torch-geometric")
    sys.exit(1)

from gnn_model.model_undirected import UndirectedFVSNet, compute_class_weights
from gnn_model.model_directed   import DirectedFVSNet, compute_class_weights_directed


def _log(msg: str) -> None:
    print(msg, flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_pt_dataset(data_dir: Path) -> list:
    """Load all .pt graph Data objects recursively from a directory."""
    files   = sorted(data_dir.rglob("*.pt"))
    dataset = [torch.load(f, weights_only=False) for f in files]
    _log(f"  Loaded {len(dataset)} graphs from {data_dir}")
    return dataset


def train_val_split(dataset: list, val_ratio: float = 0.2):
    """Randomly split dataset into train and validation sets."""
    import random
    random.shuffle(dataset)
    split = int(len(dataset) * (1 - val_ratio))
    return dataset[:split], dataset[split:]


# ═══════════════════════════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(model, dataset, device, weight_fn):
    """Compute loss, accuracy, precision, recall, and F1 on a dataset."""
    model.eval()
    total_loss = 0.0
    tp = fp = fn = tn = 0

    with torch.no_grad():
        for data in dataset:
            x          = data.x.to(device)
            edge_index = data.edge_index.to(device)
            y          = data.y.to(device)

            logits  = model(x, edge_index)
            weights = weight_fn(y).to(device)
            loss    = nn.NLLLoss(weight=weights)(logits, y)
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

    return {"loss": avg_loss, "acc": acc, "precision": prec, "recall": recall, "f1": f1}


# ═══════════════════════════════════════════════════════════════════════════════
#  Training Loop
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(model, train_set, val_set, weight_fn,
                epochs: int, lr: float, device, save_path: Path, log_every: int):
    """
    Main training loop.
    - Adam optimizer + cosine LR decay
    - Weighted NLL loss
    - Early stopping by validation F1
    """
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_f1   = -1.0
    patience      = 20
    patience_ctr  = 0
    save_path.parent.mkdir(parents=True, exist_ok=True)

    _log(f"\n  Training: {len(train_set)} graphs  |  Val: {len(val_set)} graphs")
    _log(f"  Device  : {device}")
    _log(f"  Epochs  : {epochs}  |  LR: {lr}")
    _log(f"  {'Epoch':>6}  {'TrainLoss':>10}  {'ValLoss':>10}  {'ValF1':>8}  {'ValAcc':>8}")
    _log("  " + "─" * 52)

    log_every = max(1, log_every)

    for epoch in range(1, epochs + 1):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        total_train_loss = 0.0

        for data in train_set:
            x          = data.x.to(device)
            edge_index = data.edge_index.to(device)
            y          = data.y.to(device)

            optimizer.zero_grad()
            logits  = model(x, edge_index)
            weights = weight_fn(y).to(device)
            loss    = nn.NLLLoss(weight=weights)(logits, y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        scheduler.step()

        # Heartbeat log so long runs always show live progress.
        if epoch % log_every == 0 or epoch == epochs:
            train_loss = total_train_loss / max(len(train_set), 1)
            pct = 100.0 * epoch / max(epochs, 1)
            cur_lr = scheduler.get_last_lr()[0]
            _log(f"  [progress] epoch {epoch}/{epochs} ({pct:.1f}%) train_loss={train_loss:.4f} lr={cur_lr:.6f}")

        # ── Validate every 5 epochs ───────────────────────────────────────────
        if epoch % 5 == 0 or epoch == epochs:
            train_loss = total_train_loss / max(len(train_set), 1)
            val_m      = compute_metrics(model, val_set, device, weight_fn)

            _log(f"  {epoch:>6}  {train_loss:>10.4f}  {val_m['loss']:>10.4f}"
                 f"  {val_m['f1']:>8.4f}  {val_m['acc']:>8.4f}")

            # ── Save best model ───────────────────────────────────────────────
            if val_m["f1"] > best_val_f1:
                best_val_f1 = val_m["f1"]
                torch.save(model.state_dict(), save_path)
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    _log(f"\n  Early stopping at epoch {epoch} (no improvement in {patience} checks)")
                    break

    _log(f"\n  Best Val F1: {best_val_f1:.4f}")
    _log(f"  Model saved: {save_path}")
    return best_val_f1


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train GNN for FVS prediction")
    parser.add_argument("--type",    default="both", choices=["undirected", "directed", "both"])
    parser.add_argument("--epochs",  type=int,   default=100)
    parser.add_argument("--lr",      type=float, default=0.001)
    parser.add_argument("--hidden",  type=int,   default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--data-root", type=str, default=str(PROJECT_ROOT / "gnn_model" / "datasets" / "pt"))
    args = parser.parse_args()

    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be in (0, 1)")

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_root)
    wt_dir   = PROJECT_ROOT / "gnn_model" / "weights"

    if args.type in ("undirected", "both"):
        _log("\n" + "═" * 60)
        _log("  Training UNDIRECTED FVS GCN")
        _log("═" * 60)
        dataset = load_pt_dataset(data_dir / "undirected")
        if not dataset:
            _log("  No data found. Run dataset_gen.py first.")
        else:
            train_set, val_set = train_val_split(dataset, val_ratio=args.val_ratio)
            model = UndirectedFVSNet(hidden_dim=args.hidden, dropout=args.dropout)
            train_model(
                model, train_set, val_set, compute_class_weights,
                epochs=args.epochs, lr=args.lr, device=device,
                save_path=wt_dir / "undirected_fvs_gcn.pt", log_every=args.log_every
            )

    if args.type in ("directed", "both"):
        _log("\n" + "═" * 60)
        _log("  Training DIRECTED FVS DiGCN")
        _log("═" * 60)
        dataset = load_pt_dataset(data_dir / "directed")
        if not dataset:
            _log("  No data found. Run dataset_gen.py first.")
        else:
            train_set, val_set = train_val_split(dataset, val_ratio=args.val_ratio)
            model = DirectedFVSNet(hidden_dim=args.hidden, dropout=args.dropout)
            train_model(
                model, train_set, val_set, compute_class_weights_directed,
                epochs=args.epochs, lr=args.lr, device=device,
                save_path=wt_dir / "directed_fvs_gcn.pt", log_every=args.log_every
            )


if __name__ == "__main__":
    main()