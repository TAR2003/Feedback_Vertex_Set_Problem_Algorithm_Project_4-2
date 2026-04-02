"""Undirected model for GNN-KMA-2 with enriched structural features."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class ManualGCNConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        self_loop = torch.arange(n, device=x.device)
        self_loop = torch.stack([self_loop, self_loop], dim=0)
        edge_index = torch.cat([edge_index, self_loop], dim=1)

        row, col = edge_index
        deg = torch.zeros(n, device=x.device)
        deg.scatter_add_(0, row, torch.ones(row.size(0), device=x.device))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        h = self.linear(x)
        aggr = torch.zeros_like(h)
        aggr.scatter_add_(0, col.unsqueeze(1).expand_as(h[row]), h[row] * norm.unsqueeze(1))
        return aggr + self.bias


class UndirectedFVSNetV2(nn.Module):
    def __init__(self, in_channels: int = 11, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout

        if HAS_PYG:
            self.conv1 = SAGEConv(in_channels, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)
            self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        else:
            self.conv1 = ManualGCNConv(in_channels, hidden_dim)
            self.conv2 = ManualGCNConv(hidden_dim, hidden_dim)
            self.conv3 = ManualGCNConv(hidden_dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
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


def compute_class_weights_v2(y: torch.Tensor) -> torch.Tensor:
    n = y.size(0)
    n_fvs = y.sum().item()
    n_other = n - n_fvs
    if n_fvs == 0 or n_other == 0:
        return torch.ones(2)
    return torch.tensor([n / (2 * n_other), n / (2 * n_fvs)], dtype=torch.float)
