"""
PRISM - Graph Neural Network World Model
GNN encoder over per-window network graphs + Temporal Transformer over graph
embeddings. Operates on PyTorch Geometric Data objects.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components import (
    LearnablePositionalEncoding,
    StatePredictionHead,
    ClassificationHead,
    generate_causal_mask,
)
from src.utils.constants import NUM_MITRE_STAGES

logger = logging.getLogger("prism.models.gnn_model")

# ---------------------------------------------------------------------------
# Try importing PyTorch Geometric; fall back gracefully if not installed
# ---------------------------------------------------------------------------
try:
    from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    _PYG_AVAILABLE = True
except ImportError:
    _PYG_AVAILABLE = False
    logger.warning(
        "torch_geometric not installed. GraphWorldModel will use a "
        "fallback MLP-based graph encoder."
    )


# ---------------------------------------------------------------------------
# GNN Encoder
# ---------------------------------------------------------------------------
class GraphSAGEEncoder(nn.Module):
    """Two-layer GraphSAGE encoder -> graph-level embedding via mean pooling."""

    def __init__(
        self,
        d_node: int,
        d_edge: int,
        d_graph: int,
        gnn_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not _PYG_AVAILABLE:
            raise RuntimeError("torch_geometric is required for GraphSAGEEncoder")

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        in_channels = d_node

        for i in range(gnn_layers):
            out_channels = d_graph if i == gnn_layers - 1 else d_graph
            self.convs.append(SAGEConv(in_channels, out_channels))
            self.norms.append(nn.LayerNorm(out_channels))
            in_channels = out_channels

        self.dropout = nn.Dropout(dropout)
        self.d_graph = d_graph

    def forward(self, x, edge_index, batch):
        """
        x          : (N_nodes, d_node)
        edge_index : (2, N_edges)
        batch      : (N_nodes,) — node-to-graph mapping
        Returns    : (N_graphs, d_graph)
        """
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.gelu(x)
            x = self.dropout(x)

        graph_emb = global_mean_pool(x, batch)  # (N_graphs, d_graph)
        return graph_emb


class GATEncoder(nn.Module):
    """Two-layer Graph Attention Network encoder."""

    def __init__(
        self,
        d_node: int,
        d_edge: int,
        d_graph: int,
        gnn_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not _PYG_AVAILABLE:
            raise RuntimeError("torch_geometric is required for GATEncoder")

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        in_channels = d_node

        for i in range(gnn_layers):
            is_last = i == gnn_layers - 1
            out_ch = d_graph // heads if not is_last else d_graph
            n_heads = heads if not is_last else 1
            self.convs.append(
                GATConv(in_channels, out_ch, heads=n_heads,
                        dropout=dropout, concat=not is_last)
            )
            self.norms.append(nn.LayerNorm(out_ch * n_heads if not is_last else d_graph))
            in_channels = out_ch * n_heads if not is_last else d_graph

        self.dropout = nn.Dropout(dropout)
        self.d_graph = d_graph

    def forward(self, x, edge_index, batch):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.gelu(x)
            x = self.dropout(x)
        return global_mean_pool(x, batch)


class FallbackMLPEncoder(nn.Module):
    """Fallback when PyG is not available: treat node feature matrix as flat input."""

    def __init__(self, d_node: int, d_graph: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_node, d_graph),
            nn.LayerNorm(d_graph),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_graph, d_graph),
        )
        self.d_graph = d_graph

    def forward(self, x, edge_index=None, batch=None):
        """x: (N, d_node). Mean-pool nodes, then MLP."""
        if batch is not None:
            from torch_scatter import scatter_mean
            x_agg = scatter_mean(x, batch, dim=0)
        else:
            x_agg = x.mean(dim=0, keepdim=True)
        return self.mlp(x_agg)


# ---------------------------------------------------------------------------
# Graph World Model
# ---------------------------------------------------------------------------
class GraphWorldModel(nn.Module):
    """
    World Model using GNN encoder + Temporal Transformer.

    Each time window → graph G_t → GNN embedding z_t.
    Sequence [z_{t-L+1}, ..., z_t] → Transformer → predictions.
    """

    def __init__(
        self,
        d_node: int = 8,
        d_edge: int = 8,
        d_graph: int = 128,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        gnn_type: str = "graphsage",
        gnn_layers: int = 2,
        lookback: int = 20,
        dropout: float = 0.1,
        head_dropout: float = 0.3,
        num_mitre_stages: int = NUM_MITRE_STAGES,
    ):
        super().__init__()
        self.d_graph = d_graph
        self.d_model = d_model
        self.lookback = lookback

        # GNN Encoder
        if not _PYG_AVAILABLE:
            self.gnn = FallbackMLPEncoder(d_node, d_graph, dropout)
        elif gnn_type == "gat":
            self.gnn = GATEncoder(d_node, d_edge, d_graph, gnn_layers, dropout=dropout)
        else:
            self.gnn = GraphSAGEEncoder(d_node, d_edge, d_graph, gnn_layers, dropout)

        # Project graph embedding to transformer dimension
        self.graph_proj = nn.Sequential(
            nn.Linear(d_graph, d_model),
            nn.LayerNorm(d_model),
        )

        # Positional encoding
        self.pos_enc = LearnablePositionalEncoding(lookback + 64, d_model)

        # Temporal Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        # Output heads
        self.state_head = StatePredictionHead(d_model, d_graph, head_dropout)
        self.infiltration_head = ClassificationHead(d_model, 2, head_dropout)
        self.mitre_head = ClassificationHead(d_model, num_mitre_stages, head_dropout)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "GraphWorldModel: gnn=%s, d_graph=%d, d_model=%d, params=%.2fM",
            gnn_type, d_graph, d_model, n_params / 1e6,
        )

    def encode_graph_sequence(
        self, graph_seq: list
    ) -> torch.Tensor:
        """
        Encode a sequence of graphs into a tensor of embeddings.

        Parameters
        ----------
        graph_seq : list of dicts with keys node_features, edge_index, edge_features
                    OR list of PyG Data objects

        Returns
        -------
        torch.Tensor (B, L, d_model) — or (1, L, d_model) for single sequence
        """
        embeddings = []
        for g in graph_seq:
            if _PYG_AVAILABLE and isinstance(g, Data):
                z = self.gnn(g.x, g.edge_index, g.batch)
            elif isinstance(g, dict):
                x = torch.tensor(g["node_features"], dtype=torch.float32)
                ei = torch.tensor(g["edge_index"], dtype=torch.long)
                batch = torch.zeros(x.size(0), dtype=torch.long)
                if _PYG_AVAILABLE:
                    z = self.gnn(x, ei, batch)
                else:
                    z = self.gnn(x)
            else:
                raise TypeError(f"Unknown graph format: {type(g)}")
            embeddings.append(z)

        seq = torch.cat(embeddings, dim=0)            # (L, d_graph)
        seq = self.graph_proj(seq)                    # (L, d_model)
        return seq.unsqueeze(0)                       # (1, L, d_model)

    def forward(
        self,
        graph_emb_seq: torch.Tensor,
        return_attention: bool = False,
    ) -> dict:
        """
        Parameters
        ----------
        graph_emb_seq : (B, L, d_graph) — pre-encoded graph embeddings

        Returns
        -------
        Same dict contract as StateTransformerWorldModel.forward()
        """
        B, L, _ = graph_emb_seq.shape
        device = graph_emb_seq.device

        x = self.graph_proj(graph_emb_seq)    # (B, L, d_model)
        x = self.pos_enc(x)

        causal_mask = generate_causal_mask(L, device)
        hidden_seq = self.transformer(x, mask=causal_mask, is_causal=True)

        h_t = hidden_seq[:, -1, :]

        pred_mean, pred_logvar = self.state_head(h_t)
        pred_binary = self.infiltration_head(h_t)
        pred_mitre = self.mitre_head(h_t)

        return {
            "pred_state_mean": pred_mean,
            "pred_state_logvar": pred_logvar,
            "pred_binary": pred_binary,
            "pred_mitre": pred_mitre,
            "hidden": h_t,
            "hidden_seq": hidden_seq,
            "attention_weights": None,
        }

    def predict_infiltration_prob(self, graph_emb_seq: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.forward(graph_emb_seq)
        return torch.softmax(out["pred_binary"], dim=-1)[:, 1]
