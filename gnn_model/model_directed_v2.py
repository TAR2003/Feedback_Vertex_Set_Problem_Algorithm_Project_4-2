"""Directed model for GNN-KMA-2 with enriched structural features."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectedConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.W_in = nn.Linear(in_channels, out_channels, bias=False)
        self.W_out = nn.Linear(in_channels, out_channels, bias=False)
        self.W_self = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_in.weight)
        nn.init.xavier_uniform_(self.W_out.weight)
        nn.init.xavier_uniform_(self.W_self.weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        src = edge_index[0]
        dst = edge_index[1]

        in_agg = torch.zeros(n, x.size(1), device=x.device)
        in_cnt = torch.zeros(n, 1, device=x.device)
        in_agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, x.size(1)), x[src])
        in_cnt.scatter_add_(0, dst.unsqueeze(1), torch.ones(dst.size(0), 1, device=x.device))
        in_cnt = in_cnt.clamp(min=1)
        in_agg = in_agg / in_cnt

        out_agg = torch.zeros(n, x.size(1), device=x.device)
        out_cnt = torch.zeros(n, 1, device=x.device)
        out_agg.scatter_add_(0, src.unsqueeze(1).expand(-1, x.size(1)), x[dst])
        out_cnt.scatter_add_(0, src.unsqueeze(1), torch.ones(src.size(0), 1, device=x.device))
        out_cnt = out_cnt.clamp(min=1)
        out_agg = out_agg / out_cnt

        return self.W_in(in_agg) + self.W_out(out_agg) + self.W_self(x) + self.bias


class DirectedFVSNetV2(nn.Module):
    def __init__(self, in_channels: int = 11, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout

        self.conv1 = DirectedConvLayer(in_channels, hidden_dim)
        self.conv2 = DirectedConvLayer(hidden_dim, hidden_dim)
        self.conv3 = DirectedConvLayer(hidden_dim, hidden_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

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


def compute_class_weights_directed_v2(y: torch.Tensor) -> torch.Tensor:
    n = y.size(0)
    n_fvs = y.sum().item()
    n_other = n - n_fvs
    if n_fvs == 0 or n_other == 0:
        return torch.ones(2)
    return torch.tensor([n / (2 * n_other), n / (2 * n_fvs)], dtype=torch.float)
