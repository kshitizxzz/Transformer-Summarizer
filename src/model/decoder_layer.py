"""Single Transformer decoder layer: masked self-attention + cross-attention + feed-forward."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.model.encoder_layer import PositionwiseFeedForward
from src.model.multi_head_attention import MultiHeadAttention


class DecoderLayer(nn.Module):
    """One decoder block:

    x -> [Masked Multi-Head Self-Attention -> Dropout -> Add & Norm] ->
         [Multi-Head Cross-Attention over encoder output -> Dropout -> Add & Norm] ->
         [Feed-Forward -> Dropout -> Add & Norm] -> out
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (batch, tgt_len, d_model) — decoder input (target embeddings so far)
        memory: (batch, src_len, d_model) — encoder output
        tgt_mask: causal + padding mask for decoder self-attention
        memory_mask: padding mask over encoder positions for cross-attention
        """
        self_attn_out, self_attn_weights = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_out))

        cross_attn_out, cross_attn_weights = self.cross_attn(x, memory, memory, mask=memory_mask)
        x = self.norm2(x + self.dropout2(cross_attn_out))

        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_out))

        return x, self_attn_weights, cross_attn_weights
