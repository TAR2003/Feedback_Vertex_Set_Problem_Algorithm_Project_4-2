"""
model_directed.py
=================
Directed Graph Convolutional Network (DiGCN) for directed FVS node classification.

Key difference from undirected GCN:
  In directed graphs, edges have direction.  A standard GCN symmetrizes the
  adjacency matrix (treats all edges as undirected), which loses directionality
  information crucial for detecting directed cycles.

  Our DiGCN uses SEPARATE aggregation for incoming and outgoing edges:
    h_in[v]  = aggregate(predecessors of v)  → "what flows INTO v"
    h_out[v] = aggregate(successors of v)    → "what flows OUT OF v"
    h[v]     = concat(h_in[v], h_out[v], x[v])  → full directed context

  The concatenated representation captures the key signal:
  If v has many predecessors AND many successors, it's likely on a directed cycle.

Architecture:
  Input(3) → DiGCN Layer 1(128) → DiGCN Layer 2(128) → DiGCN Layer 3(64)
           → MLP Head → 2-class log-softmax

Input features (3 per vertex):
  [0] in-degree (normalized)
  [1] out-degree (normalized)
  [2] min(in, out) / (n-1)  ← proxy for cycle participation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


# ═══════════════════════════════════════════════════════════════════════════════
#  Directed Convolutional Layer
# ═══════════════════════════════════════════════════════════════════════════════

class DirectedConvLayer(nn.Module):
    """
    Directed graph convolutional layer.

    Aggregates separately over:
      - Incoming edges (predecessors): captures "what leads to v"
      - Outgoing edges (successors):   captures "where v leads"

    Output: h'[v] = σ(W_in · mean(h[predecessors]) +
                      W_out· mean(h[successors]) +
                      W_self· h[v] + b)
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Separate weight matrices for in/out/self
        self.W_in   = nn.Linear(in_channels, out_channels, bias=False)
        self.W_out  = nn.Linear(in_channels, out_channels, bias=False)
        self.W_self = nn.Linear(in_channels, out_channels, bias=False)
        self.bias   = nn.Parameter(torch.zeros(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_in.weight)
        nn.init.xavier_uniform_(self.W_out.weight)
        nn.init.xavier_uniform_(self.W_self.weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x          : [N, in_channels]
            edge_index : [2, E]  directed edges (row=src, col=dst)
        Returns:
            h'         : [N, out_channels]
        """
        N   = x.size(0)
        src = edge_index[0]   # source vertices
        dst = edge_index[1]   # destination vertices

        # ── Incoming aggregation (mean over predecessors) ────────────────────
        # For vertex v: mean of x[u] for all u → v
        in_agg = torch.zeros(N, x.size(1), device=x.device)
        in_cnt = torch.zeros(N, 1,         device=x.device)
        in_agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, x.size(1)), x[src])
        in_cnt.scatter_add_(0, dst.unsqueeze(1), torch.ones(dst.size(0), 1, device=x.device))
        in_cnt = in_cnt.clamp(min=1)
        in_agg = in_agg / in_cnt

        # ── Outgoing aggregation (mean over successors) ──────────────────────
        # For vertex v: mean of x[w] for all v → w
        out_agg = torch.zeros(N, x.size(1), device=x.device)
        out_cnt = torch.zeros(N, 1,         device=x.device)
        out_agg.scatter_add_(0, src.unsqueeze(1).expand(-1, x.size(1)), x[dst])
        out_cnt.scatter_add_(0, src.unsqueeze(1), torch.ones(src.size(0), 1, device=x.device))
        out_cnt = out_cnt.clamp(min=1)
        out_agg = out_agg / out_cnt

        # ── Combine ──────────────────────────────────────────────────────────
        return self.W_in(in_agg) + self.W_out(out_agg) + self.W_self(x) + self.bias


# ═══════════════════════════════════════════════════════════════════════════════
#  Main DiGCN Model
# ═══════════════════════════════════════════════════════════════════════════════

class DirectedFVSNet(nn.Module):
    """
    3-layer Directed GCN for directed FVS vertex prediction.
    """

    def __init__(self, in_channels: int = 3, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout

        # Three directed conv layers
        self.conv1 = DirectedConvLayer(in_channels, hidden_dim)
        self.conv2 = DirectedConvLayer(hidden_dim,  hidden_dim)
        self.conv3 = DirectedConvLayer(hidden_dim,  hidden_dim)

        # Batch norm
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # Classification head
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x          : [N, in_channels]
            edge_index : [2, E]  (directed edges, row=src, col=dst)
        Returns:
            logits     : [N, 2]  log-softmax
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

    def predict_dfvs(self, x: torch.Tensor, edge_index: torch.Tensor,
                     threshold: float = 0.5) -> list:
        """
        Predict DFVS vertices.
        Returns list of vertex indices with P(in-FVS) > threshold.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, edge_index)
            probs  = logits.exp()
            preds  = (probs[:, 1] > threshold).nonzero(as_tuple=True)[0]
        return preds.tolist()


def compute_class_weights_directed(y: torch.Tensor) -> torch.Tensor:
    """Class weights for directed FVS (same logic as undirected)."""
    n       = y.size(0)
    n_fvs   = y.sum().item()
    n_other = n - n_fvs
    if n_fvs == 0 or n_other == 0:
        return torch.ones(2)
    return torch.tensor([n / (2 * n_other), n / (2 * n_fvs)], dtype=torch.float)