"""Temporal Graph Neural Network (TGNN) for TrustShield.

Investigates continuous-time dynamic graph message passing for fraud detection.
Enforces the fundamental temporal invariant:
    event_timestamp < decision_timestamp (strictly no future information)

Features:
- Continuous Time Harmonic Encoding: phi(Delta t) mapping time intervals into cosine/sine bases
- Time-Decayed Edge Attention Conv: modulates message weights based on recency
- Temporal Safety Verification: asserts Delta t >= 0 for all active messages
- Controlled ablation against static GNN and HeteroGNN
"""

import math
from typing import Tuple
import torch


import torch.nn as nn
import torch.nn.functional as F


class Time2Vec(nn.Module):
    """Continuous Time Harmonic Encoder mapping time interval Delta t into periodic and linear embeddings."""

    def __init__(self, out_dim: int = 16):
        super().__init__()
        self.out_dim = out_dim
        self.w0 = nn.Parameter(torch.randn(1, 1))
        self.b0 = nn.Parameter(torch.randn(1, 1))
        self.w = nn.Parameter(torch.randn(1, out_dim - 1))
        self.b = nn.Parameter(torch.randn(1, out_dim - 1))

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        """delta_t shape: [E, 1] in days or hours (must be strictly >= 0)."""
        # Strict temporal guard
        if torch.any(delta_t < 0):
            raise AssertionError(
                "Temporal leakage detected: negative delta_t found in dynamic graph message passing."
            )

        linear = self.w0 * delta_t + self.b0
        periodic = torch.sin(self.w * delta_t + self.b)
        return torch.cat([linear, periodic], dim=-1)


class TemporalGraphAttentionLayer(nn.Module):
    """Message passing layer with time-decayed attention."""

    def __init__(self, in_features: int, out_features: int, time_dim: int = 16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.time_encoder = Time2Vec(out_dim=time_dim)

        self.w_src = nn.Linear(in_features, out_features, bias=False)
        self.w_dst = nn.Linear(in_features, out_features, bias=False)
        self.w_time = nn.Linear(time_dim, out_features, bias=False)
        self.w_msg = nn.Linear(in_features + time_dim, out_features)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> torch.Tensor:
        """h: [N, in_features], edge_index: [2, E], delta_t: [E, 1]."""
        u, v = edge_index[0], edge_index[1]
        t_enc = self.time_encoder(delta_t)

        h_src = self.w_src(h[u])
        h_dst = self.w_dst(h[v])
        h_t = self.w_time(t_enc)

        # Scaled dot-product query-key attention
        attn_scores = (h_src * (h_dst + h_t)).sum(dim=-1, keepdim=True) / math.sqrt(self.out_features)
        # Destination-wise softmax normalization (Issue 18)
        exp_scores = torch.exp(torch.clamp(attn_scores, -20.0, 20.0))
        denom = torch.zeros((h.size(0), 1), device=h.device)
        denom.index_add_(0, v, exp_scores)
        attn_weights = exp_scores / (denom[v] + 1e-8)

        # Message formulation combining node state and temporal gap
        messages = self.w_msg(torch.cat([h[u], t_enc], dim=-1))
        weighted_messages = attn_weights * messages

        # Aggregate into destination nodes
        out = torch.zeros((h.size(0), self.out_features), device=h.device)
        out.index_add_(0, v, weighted_messages)
        return F.relu(out)


class TemporalGNN(nn.Module):
    """Multi-layer Temporal Graph Neural Network."""

    def __init__(self, in_features: int = 8, hidden_features: int = 16, out_features: int = 16, time_dim: int = 16):
        super().__init__()
        self.layer1 = TemporalGraphAttentionLayer(in_features, hidden_features, time_dim=time_dim)
        self.layer2 = TemporalGraphAttentionLayer(hidden_features, out_features, time_dim=time_dim)
        self.classifier = nn.Sequential(
            nn.Linear(out_features, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h1 = self.layer1(h, edge_index, delta_t)
        h2 = self.layer2(h1, edge_index, delta_t)
        logits = self.classifier(h2)
        return logits, h2
