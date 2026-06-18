"""Multi-head attention module."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.model.attention import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    """Runs `num_heads` scaled-dot-product-attention operations in parallel
    over learned linear projections of Q, K, V, then concatenates and
    projects the result back to `d_model`.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout_p = dropout
        self.attn_weights: Optional[torch.Tensor] = None  # cached for visualization

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)"""
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model)"""
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """query/key/value: (batch, seq_len, d_model)
        mask: broadcastable to (batch, 1, seq_len_q, seq_len_k)
        """
        q = self._split_heads(self.w_q(query))
        k = self._split_heads(self.w_k(key))
        v = self._split_heads(self.w_v(value))

        attn_output, attn_weights = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout_p=self.dropout_p, training=self.training
        )

        self.attn_weights = attn_weights.detach()  # (batch, num_heads, seq_q, seq_k)

        output = self._combine_heads(attn_output)
        return self.w_o(output), attn_weights
