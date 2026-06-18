"""Perplexity computation for the Transformer summarizer."""

from __future__ import annotations

import math
from typing import Optional, Union

import torch
from torch.utils.data import DataLoader

from src.model.transformer import Transformer
from src.training.trainer import LabelSmoothingLoss


def perplexity_from_loss(loss: float) -> float:
    """Perplexity = exp(average per-token cross-entropy loss)."""
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


@torch.no_grad()
def compute_perplexity(
    model: Transformer,
    data_loader: DataLoader,
    pad_id: int,
    device: Optional[Union[str, torch.device]] = None,
) -> float:
    """Run the model over `data_loader` and return corpus-level perplexity.

    Uses plain (non-label-smoothed) cross-entropy, since perplexity is only
    well defined with respect to the true token-probability distribution.
    """
    device = torch.device(device) if device else next(model.parameters()).device
    model.eval()

    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")
    total_loss = 0.0
    total_tokens = 0

    for batch in data_loader:
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)

        decoder_input = tgt[:, :-1]
        gold = tgt[:, 1:]

        logits = model(src, decoder_input)
        loss = criterion(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))

        num_tokens = (gold != pad_id).sum().item()
        total_loss += loss.item()
        total_tokens += num_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    return perplexity_from_loss(avg_loss)


__all__ = ["perplexity_from_loss", "compute_perplexity", "LabelSmoothingLoss"]
