"""Scaled dot-product attention — the core primitive of the Transformer."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes Attention(Q, K, V) = softmax(QK^T / sqrt(d_k) + mask) V.

    Shapes
    ------
    query: (..., seq_len_q, d_k)
    key:   (..., seq_len_k, d_k)
    value: (..., seq_len_k, d_v)
    mask:  broadcastable to (..., seq_len_q, seq_len_k); positions with
           value 0/False are masked out (set to -inf before softmax).

    Returns
    -------
    output: (..., seq_len_q, d_v)
    attn_weights: (..., seq_len_q, seq_len_k) — useful for visualization.
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)

    # Guard against rows that were fully masked (e.g. padding queries),
    # which would otherwise produce NaNs from softmax over all -inf.
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=training)

    output = torch.matmul(attn_weights, value)
    return output, attn_weights
