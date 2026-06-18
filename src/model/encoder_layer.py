"""Single Transformer encoder layer: self-attention + feed-forward."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.model.multi_head_attention import MultiHeadAttention


class PositionwiseFeedForward(nn.Module):
    """Two-layer MLP applied independently to each position: FFN(x) = max(0, xW1+b1)W2+b2."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    """One encoder block:

    x -> [Multi-Head Self-Attention -> Dropout -> Add & Norm] ->
         [Feed-Forward -> Dropout -> Add & Norm] -> out
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (batch, src_len, d_model); src_mask masks out padding positions."""
        attn_out, attn_weights = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + self.dropout1(attn_out))

        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_out))

        return x, attn_weights
