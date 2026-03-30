"""
model_undirected.py
===================
Graph Convolutional Network (GCN) for undirected FVS node classification.

Architecture:
  Input  → GCN layer 1 (hidden_dim) + ReLU + Dropout
         → GCN layer 2 (hidden_dim) + ReLU + Dropout
         → GCN layer 3 (hidden_dim)
         → MLP head (hidden_dim → 2) + Softmax
  Output → per-node binary classification (in FVS / not in FVS)

The GCN layers aggregate information from 1-hop neighborhoods, allowing
each vertex to "see" its local structure — exactly what determines whether
a vertex is on a cycle and should be in the FVS.

Loss: Weighted cross-entropy (FVS vertices are a minority class, so we
upweight them by the class imbalance ratio).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv, SAGEConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("WARNING: torch_geometric not found. Using manual GCN implementation.")


# ═══════════════════════════════════════════════════════════════════════════════
#  Manual GCN implementation (fallback if torch_geometric unavailable)
# ═══════════════════════════════════════════════════════════════════════════════

class ManualGCNConv(nn.Module):
    """
    Single GCN layer: H' = σ(D^{-1/2} A D^{-1/2} H W)
    where A = adjacency + self-loops, D = degree matrix.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.bias   = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        # Add self-loops
        self_loop = torch.arange(n, device=x.device)
        self_loop = torch.stack([self_loop, self_loop], dim=0)
        edge_index = torch.cat([edge_index, self_loop], dim=1)

        # Degree normalization
        row, col = edge_index
        deg = torch.zeros(n, device=x.device)
        deg.scatter_add_(0, row, torch.ones(row.size(0), device=x.device))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0

        # Normalized message passing: sum over neighbors
        norm  = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        h     = self.linear(x)
        aggr  = torch.zeros_like(h)
        aggr.scatter_add_(0, col.unsqueeze(1).expand_as(h[row]), h[row] * norm.unsqueeze(1))
        return aggr + self.bias


# ═══════════════════════════════════════════════════════════════════════════════
#  Main GCN Model
# ═══════════════════════════════════════════════════════════════════════════════

class UndirectedFVSNet(nn.Module):
    """
    3-layer GCN for undirected FVS vertex prediction.

    Input features (per vertex):
      - degree (normalized)
      - clustering coefficient
      - log-degree (normalized)

    Output: 2-class softmax per vertex (not-in-FVS / in-FVS)
    """

    def __init__(self, in_channels: int = 3, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout

        if HAS_PYG:
            # Use GraphSAGE convolutions — more stable than GCN for FVS
            self.conv1 = SAGEConv(in_channels, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim,  hidden_dim)
            self.conv3 = SAGEConv(hidden_dim,  hidden_dim)
        else:
            self.conv1 = ManualGCNConv(in_channels, hidden_dim)
            self.conv2 = ManualGCNConv(hidden_dim,  hidden_dim)
            self.conv3 = ManualGCNConv(hidden_dim,  hidden_dim)

        # MLP classification head
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)   # 2 classes: not-in-FVS, in-FVS
        )

        # Batch normalization for training stability
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x          : Node feature matrix [num_nodes, in_channels]
            edge_index : COO edge index       [2, num_edges]
        Returns:
            logits     : [num_nodes, 2]  (log-softmax output)
        """
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv3(h, edge_index)
        h = self.bn3(h)
        h = F.relu(h)

        out = self.mlp(h)
        return F.log_softmax(out, dim=1)

    def predict_fvs(self, x: torch.Tensor, edge_index: torch.Tensor,
                    threshold: float = 0.5) -> list:
        """
        Run inference and return predicted FVS vertex list.
        Args:
            threshold : probability threshold for including a vertex
        Returns:
            List of vertex indices predicted to be in the FVS
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, edge_index)
            probs  = logits.exp()
            preds  = (probs[:, 1] > threshold).nonzero(as_tuple=True)[0]
        return preds.tolist()


# ═══════════════════════════════════════════════════════════════════════════════
#  Loss helper
# ═══════════════════════════════════════════════════════════════════════════════

def compute_class_weights(y: torch.Tensor) -> torch.Tensor:
    """
    Compute class weights to handle FVS label imbalance.
    FVS vertices are typically < 20% of all vertices.
    Weight = n_samples / (n_classes * class_count[class])
    """
    n       = y.size(0)
    n_fvs   = y.sum().item()
    n_other = n - n_fvs

    if n_fvs == 0 or n_other == 0:
        return torch.ones(2)

    w_other = n / (2 * n_other)
    w_fvs   = n / (2 * n_fvs)
    return torch.tensor([w_other, w_fvs], dtype=torch.float)