"""Transformer encoder: embedding + positional encoding + stacked encoder layers."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from src.model.embeddings import TokenEmbedding
from src.model.encoder_layer import EncoderLayer
from src.model.positional_encoding import PositionalEncoding


class Encoder(nn.Module):
    """Stack of `num_layers` `EncoderLayer`s, fed by an embedded + positionally-encoded input."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = TokenEmbedding(vocab_size, d_model, padding_idx=padding_idx)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """src: (batch, src_len) token ids -> (batch, src_len, d_model)"""
        x = self.embedding(src)
        x = self.positional_encoding(x)

        attn_weights_per_layer: List[torch.Tensor] = []
        for layer in self.layers:
            x, attn_weights = layer(x, src_mask)
            attn_weights_per_layer.append(attn_weights)

        return self.norm(x), attn_weights_per_layer
