"""Attention-weight visualization (encoder self-attention, decoder self-attention,
and encoder-decoder cross-attention) as interactive Plotly heatmaps.
"""

from __future__ import annotations

from typing import List, Sequence

import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots


def _to_numpy(attn_weights: torch.Tensor, layer: int, head: int) -> "list[list[float]]":
    """attn_weights: list-of-layers, each (batch, num_heads, seq_q, seq_k).
    Returns a single (seq_q, seq_k) matrix for the requested layer/head,
    using batch index 0.
    """
    weights = attn_weights[layer][0, head].detach().cpu().numpy()
    return weights


def plot_attention_heatmap(
    attn_weights: Sequence[torch.Tensor],
    query_tokens: Sequence[str],
    key_tokens: Sequence[str],
    layer: int = 0,
    head: int = 0,
    title: str = "Attention weights",
) -> go.Figure:
    """Single heatmap for one (layer, head) pair.

    Parameters
    ----------
    attn_weights:
        List of per-layer attention tensors, each shaped
        (batch, num_heads, seq_q, seq_k) — as returned by
        `Summarizer.get_attention_maps`.
    query_tokens, key_tokens:
        Token strings labelling the heatmap axes.
    """
    matrix = _to_numpy(attn_weights, layer, head)

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=list(key_tokens),
            y=list(query_tokens),
            colorscale="Viridis",
            colorbar=dict(title="weight"),
        )
    )
    fig.update_layout(
        title=f"{title} (layer {layer}, head {head})",
        xaxis_title="Key tokens",
        yaxis_title="Query tokens",
        yaxis=dict(autorange="reversed"),
        height=max(400, 24 * len(query_tokens)),
    )
    return fig


def plot_attention_grid(
    attn_weights: Sequence[torch.Tensor],
    query_tokens: Sequence[str],
    key_tokens: Sequence[str],
    layer: int = 0,
    num_heads: int | None = None,
    title: str = "Attention heads",
) -> go.Figure:
    """Grid of small heatmaps, one per attention head, for a given layer."""
    n_heads = num_heads or attn_weights[layer].shape[1]
    cols = min(4, n_heads)
    rows = (n_heads + cols - 1) // cols

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"head {h}" for h in range(n_heads)])

    for h in range(n_heads):
        matrix = _to_numpy(attn_weights, layer, h)
        r, c = h // cols + 1, h % cols + 1
        fig.add_trace(
            go.Heatmap(z=matrix, x=list(key_tokens), y=list(query_tokens), colorscale="Viridis", showscale=False),
            row=r,
            col=c,
        )

    fig.update_layout(title=f"{title} — layer {layer}", height=300 * rows)
    fig.update_yaxes(autorange="reversed")
    return fig


def average_heads(attn_weights: Sequence[torch.Tensor], layer: int = 0) -> List[List[float]]:
    """Average attention weights across all heads for a given layer -> (seq_q, seq_k)."""
    return attn_weights[layer][0].mean(dim=0).detach().cpu().numpy().tolist()
