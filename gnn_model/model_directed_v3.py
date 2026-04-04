"""
model_directed_v3.py
====================
Research-grade GNN models for FVS prediction.

Architecture: GAT (v2) + residual connections + 5-layer depth + global readout.

Key design decisions vs v1/v2:
  - GATv2Conv replaces hand-rolled aggregation: learns attention weights over
    neighbors rather than averaging them uniformly. High-degree cycle vertices
    get stronger signals from their cycle partners.
  - Residual connections (skip connections) prevent oversmoothing in deep GNNs.
    Without them, 5-layer GraphSAGE converges to uniform node embeddings.
  - Separate forward/reverse aggregation in DirectedFVSNetV3 captures both
    in-flow and out-flow structural roles, which are asymmetric in DFVS.
  - Global readout (mean pooling → MLP) appended to node features gives each
    vertex graph-level context, enabling relative-difficulty assessment.
  - Output: single logit per node → sigmoid → BCEWithLogitsLoss.
    (v1/v2 used 2-class log-softmax + NLLLoss; this is more numerically stable)

References:
  - Veličković et al. (2018) Graph Attention Networks, ICLR 2018.
  - Brody et al. (2022) How Attentive are Graph Attention Networks? (GATv2)
  - He et al. (2016) Deep Residual Learning for Image Recognition.
  - Rampasek et al. (2022) Recipe for a General, Powerful, Scalable Graph
    Transformer. NeurIPS 2022.
  - Khalil et al. (2017) Learning Combinatorial Optimization Algorithms over
    Graphs (S2V-DQN).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATv2Conv, BatchNorm
    from torch_geometric.nn import global_mean_pool
    HAS_GATV2 = True
except ImportError:
    GATv2Conv = None
    BatchNorm = None
    global_mean_pool = None
    HAS_GATV2 = False


# ─── Fallback BatchNorm for when torch_geometric is not installed ────────────
if not HAS_GATV2:
    class BatchNorm(nn.Module):  # type: ignore[no-redef]
        def __init__(self, dim: int):
            super().__init__()
            self.bn = nn.BatchNorm1d(dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.bn(x)


class ResidualGATBlock(nn.Module):
    """
    Single GATv2 layer with residual connection + batch normalization.

    Residual connection: output = ReLU(BN(GAT(x))) + proj(x)
    where proj is identity if in_channels == out_channels, else a linear map.

    References:
      - He et al. (2016) Deep Residual Learning for Image Recognition.
      - Brody et al. (2022) How Attentive are Graph Attention Networks?
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        dropout: float = 0.3,
        concat: bool = False,
    ):
        super().__init__()
        self.dropout_p = dropout

        if HAS_GATV2:
            self.conv = GATv2Conv(
                in_channels,
                out_channels,
                heads=heads,
                concat=concat,     # concat=False → output is out_channels
                dropout=dropout,
                add_self_loops=True,
            )
        else:
            # Fallback: plain linear transform (no attention)
            self.conv = None
            self.lin = nn.Linear(in_channels, out_channels)

        self.norm = BatchNorm(out_channels)

        # Residual projection when dimensions differ
        self.residual_proj: nn.Module | None
        if in_channels != out_channels:
            self.residual_proj = nn.Linear(in_channels, out_channels, bias=False)
        else:
            self.residual_proj = None

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        # Residual path
        identity = x
        if self.residual_proj is not None:
            identity = self.residual_proj(x)

        # Main path
        if self.conv is not None:
            out = self.conv(x, edge_index)
        else:
            out = self.lin(x)

        out = self.norm(out)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout_p, training=self.training)

        return out + identity   # residual add


class DirectedFVSNetV3(nn.Module):
    """
    Research-grade directed FVS prediction network.

    Architecture:
      - Separate forward/reverse GAT aggregation (captures directed cycle roles)
      - 5 ResidualGATBlock layers (wider receptive field, no oversmoothing)
      - Global mean-pool readout concatenated to per-node features
      - 3-layer bottleneck MLP → single logit
      - Output: (n,) logits, use sigmoid + BCEWithLogitsLoss

    Input features: 16-channel vector from feature_engineering_v3.py

    References:
      - Veličković et al. (2018) GAT
      - Rampasek et al. (2022) GPS: global readout design
      - Khalil et al. (2017) S2V-DQN: global embedding concatenation
    """

    def __init__(
        self,
        in_channels: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 5,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout_p = dropout

        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.input_norm = BatchNorm(hidden_dim)

        # Per-layer forward + reverse GAT + fusion
        self.forward_layers = nn.ModuleList()
        self.reverse_layers = nn.ModuleList()
        self.fusion_norms = nn.ModuleList()
        self.fusion_projs = nn.ModuleList()

        for _ in range(num_layers):
            self.forward_layers.append(
                ResidualGATBlock(hidden_dim, hidden_dim, heads=heads,
                                 dropout=dropout, concat=False)
            )
            self.reverse_layers.append(
                ResidualGATBlock(hidden_dim, hidden_dim, heads=heads,
                                 dropout=dropout, concat=False)
            )
            # Fuse: forward + reverse + identity skip
            self.fusion_projs.append(nn.Linear(3 * hidden_dim, hidden_dim))
            self.fusion_norms.append(BatchNorm(hidden_dim))

        # Global readout projection (graph → per-node context)
        self.global_readout = nn.Linear(hidden_dim, hidden_dim // 4)

        # Classification MLP
        mlp_in = hidden_dim + hidden_dim // 4
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            # NOTE: no sigmoid here — use BCEWithLogitsLoss
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:          Node features (n × in_channels)
            edge_index: Directed edges (2 × m)
            batch:      Batch vector for global readout (None = single graph)
        Returns:
            logits:     (n,) — raw logits for FVS membership (apply sigmoid)
        """
        # Reverse edges for backward aggregation
        reverse_edge_index = edge_index.flip(0)

        # Input projection
        h = F.relu(self.input_norm(self.input_proj(x)))
        h = F.dropout(h, p=self.dropout_p, training=self.training)

        # GNN message passing
        for i in range(self.num_layers):
            h_fwd = self.forward_layers[i](h, edge_index)
            h_rev = self.reverse_layers[i](h, reverse_edge_index)

            # Fuse forward + backward + skip
            h_fused = torch.cat([h_fwd, h_rev, h], dim=-1)
            h = F.relu(self.fusion_norms[i](self.fusion_projs[i](h_fused)))

        # Global context: mean readout → expand to per-node
        if batch is not None and global_mean_pool is not None:
            g = global_mean_pool(h, batch)         # (batch_size, hidden_dim)
            g_node = g[batch]                       # (n, hidden_dim)
        else:
            g_node = h.mean(dim=0, keepdim=True).expand(h.size(0), -1)

        g_context = F.relu(self.global_readout(g_node))  # (n, hidden//4)

        # Concatenate and classify
        h_final = torch.cat([h, g_context], dim=-1)
        logits = self.mlp(h_final).squeeze(-1)           # (n,)
        return logits


class UndirectedFVSNetV3(nn.Module):
    """
    Research-grade undirected FVS prediction network.

    Same architecture as DirectedFVSNetV3 but without direction-specific
    forward/reverse separation — uses a single ResidualGATBlock per layer.

    Input features: 16-channel vector from feature_engineering_v3.py
    Output: (n,) logits — apply sigmoid + BCEWithLogitsLoss
    """

    def __init__(
        self,
        in_channels: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 5,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout_p = dropout

        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.input_norm = BatchNorm(hidden_dim)

        self.layers = nn.ModuleList([
            ResidualGATBlock(hidden_dim, hidden_dim, heads=heads,
                             dropout=dropout, concat=False)
            for _ in range(num_layers)
        ])

        self.global_readout = nn.Linear(hidden_dim, hidden_dim // 4)

        mlp_in = hidden_dim + hidden_dim // 4
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = F.relu(self.input_norm(self.input_proj(x)))
        h = F.dropout(h, p=self.dropout_p, training=self.training)

        for layer in self.layers:
            h = layer(h, edge_index)

        if batch is not None and global_mean_pool is not None:
            g = global_mean_pool(h, batch)
            g_node = g[batch]
        else:
            g_node = h.mean(dim=0, keepdim=True).expand(h.size(0), -1)

        g_context = F.relu(self.global_readout(g_node))
        h_final = torch.cat([h, g_context], dim=-1)
        logits = self.mlp(h_final).squeeze(-1)
        return logits


def compute_class_weights_v3(y: torch.Tensor) -> torch.Tensor:
    """Compute class balance weight for v3 BCEWithLogitsLoss pos_weight."""
    n_pos = y.sum().float()
    n_neg = (y == 0).sum().float()
    if n_pos == 0 or n_neg == 0:
        return torch.ones(1)
    return (n_neg / n_pos).unsqueeze(0)


# Alias for compatibility
DirectedFVSNetV3 = DirectedFVSNetV3
UndirectedFVSNetV3 = UndirectedFVSNetV3
