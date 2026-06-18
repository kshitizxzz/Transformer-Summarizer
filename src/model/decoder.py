"""Transformer decoder: embedding + positional encoding + stacked decoder layers."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from src.model.decoder_layer import DecoderLayer
from src.model.embeddings import TokenEmbedding
from src.model.positional_encoding import PositionalEncoding


class Decoder(nn.Module):
    """Stack of `num_layers` `DecoderLayer`s, fed by an embedded + positionally-encoded target."""

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
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        tgt: (batch, tgt_len) token ids
        memory: (batch, src_len, d_model) encoder output
        """
        x = self.embedding(tgt)
        x = self.positional_encoding(x)

        self_attn_per_layer: List[torch.Tensor] = []
        cross_attn_per_layer: List[torch.Tensor] = []
        for layer in self.layers:
            x, self_attn, cross_attn = layer(x, memory, tgt_mask, memory_mask)
            self_attn_per_layer.append(self_attn)
            cross_attn_per_layer.append(cross_attn)

        return self.norm(x), self_attn_per_layer, cross_attn_per_layer
