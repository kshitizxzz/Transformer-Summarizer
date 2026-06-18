"""Token embedding layer."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Learned token embedding, scaled by sqrt(d_model) as in the original paper.

    Scaling keeps the embedding magnitude comparable to the sinusoidal
    positional encoding it gets added to.
    """

    def __init__(self, vocab_size: int, d_model: int, padding_idx: int = 0) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: (batch, seq_len) -> (batch, seq_len, d_model)"""
        return self.embedding(token_ids) * math.sqrt(self.d_model)
