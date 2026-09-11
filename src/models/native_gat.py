"""
PRISM — Native PyTorch Dynamic Graph Attention Network (NativeBipartiteGAT)

A pure PyTorch multi-head edge-aware Graph Attention layer that extracts
topological structural embeddings from IP interaction graphs with ZERO
external C++ dependencies (no torch_geometric required).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class NativeGraphAttentionLayer(nn.Module):
    """
    Multi-Head Graph Attention Layer operating on dense or batched interaction graphs.
    h_i' = sigma( sum_{j in N(i)} alpha_{ij} W_v h_j )
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        alpha: float = 0.2,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.d_head = out_features // n_heads

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a_src = nn.Parameter(torch.empty(size=(1, n_heads, self.d_head)))
        self.a_dst = nn.Parameter(torch.empty(size=(1, n_heads, self.d_head)))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_features)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(
        self,
        h: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        h   : (B, N, in_features) node embeddings
        adj : (B, N, N) adjacency matrix (1.0 where connection exists, 0.0 otherwise)
        """
        B, N, _ = h.shape

        # Linear projection: (B, N, n_heads, d_head)
        Wh = self.W(h).view(B, N, self.n_heads, self.d_head)

        # Attention logits: (B, n_heads, N, 1) + (B, n_heads, 1, N)
        Wh_t = Wh.transpose(1, 2)  # (B, n_heads, N, d_head)
        f_src = (Wh_t * self.a_src.unsqueeze(2)).sum(dim=-1, keepdim=True)  # (B, n_heads, N, 1)
        f_dst = (Wh_t * self.a_dst.unsqueeze(2)).sum(dim=-1).unsqueeze(-2)  # (B, n_heads, 1, N)

        logits = self.leakyrelu(f_src + f_dst)  # (B, n_heads, N, N)

        if adj is not None:
            # Mask disconnected nodes with large negative value
            adj_mask = (adj.unsqueeze(1) <= 0)
            logits = logits.masked_fill(adj_mask, -1e9)

        attn = F.softmax(logits, dim=-1)
        attn = self.dropout(attn)

        # Weighted message aggregation: (B, n_heads, N, d_head)
        h_prime = torch.matmul(attn, Wh_t)

        # Transpose back and concat heads: (B, N, out_features)
        h_prime = h_prime.transpose(1, 2).contiguous().view(B, N, self.out_features)
        return self.norm(h_prime + self.W(h))


class DynamicGraphEncoder(nn.Module):
    """
    Two-layer Dynamic Graph Attention Encoder.
    Processes dynamic IP node topologies and outputs a fixed-size graph embedding.
    """

    def __init__(
        self,
        d_node: int = 16,
        d_hidden: int = 64,
        d_graph: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_graph = d_graph

        self.in_proj = nn.Sequential(
            nn.Linear(d_node, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
        )

        self.gat1 = NativeGraphAttentionLayer(d_hidden, d_hidden, n_heads=n_heads, dropout=dropout)
        self.gat2 = NativeGraphAttentionLayer(d_hidden, d_graph, n_heads=n_heads, dropout=dropout)

        self.readout = nn.Sequential(
            nn.Linear(d_graph * 2, d_graph),
            nn.LayerNorm(d_graph),
            nn.GELU(),
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        node_feats: (B, N, d_node)
        adj       : (B, N, N)
        Returns   : (B, d_graph)
        """
        h = self.in_proj(node_feats)
        h = self.gat1(h, adj)
        h = F.gelu(h)
        h = self.gat2(h, adj)

        # Readout: MeanPool || MaxPool
        h_mean = h.mean(dim=1)
        h_max, _ = h.max(dim=1)
        graph_emb = self.readout(torch.cat([h_mean, h_max], dim=-1))
        return graph_emb


class GraphTopologyExtractor(nn.Module):
    """
    Fast neural topological projection for aggregated window states.
    Projects explicit graph invariants (degrees, reciprocity, port entropy,
    directionality) into a rich 64-d topological manifold.
    """

    def __init__(
        self,
        d_in: int = 8,
        d_out: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, d_out),
            nn.LayerNorm(d_out),
            nn.GELU(),
        )

    def forward(self, graph_invariants: torch.Tensor) -> torch.Tensor:
        """graph_invariants: (B, 8) -> (B, 64)"""
        return self.net(graph_invariants)
