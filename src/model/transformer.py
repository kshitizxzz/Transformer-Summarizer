"""Full encoder-decoder Transformer for sequence-to-sequence summarization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from src.model.decoder import Decoder
from src.model.encoder import Encoder


@dataclass
class TransformerConfig:
    """Hyperparameters for `Transformer`. Mirrors the base config in
    "Attention Is All You Need", scaled down for a summarization task
    trained on a single GPU/CPU.
    """

    src_vocab_size: int
    tgt_vocab_size: int
    d_model: int = 256
    num_layers: int = 3
    num_heads: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    max_len: int = 512
    pad_id: int = 0


def make_src_mask(src: torch.Tensor, pad_id: int) -> torch.Tensor:
    """(batch, src_len) -> (batch, 1, 1, src_len) boolean mask, True = attend."""
    return (src != pad_id).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Combines padding mask with a causal (no-peek-ahead) mask.

    Returns (batch, 1, tgt_len, tgt_len) boolean mask, True = attend.
    """
    batch_size, tgt_len = tgt.shape
    pad_mask = (tgt != pad_id).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, tgt_len)

    causal_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device, dtype=torch.bool))
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, tgt_len, tgt_len)

    return pad_mask & causal_mask


class Transformer(nn.Module):
    """Encoder-decoder Transformer with a final linear "generator" projecting
    decoder hidden states to vocabulary logits.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.pad_id = config.pad_id

        self.encoder = Encoder(
            vocab_size=config.src_vocab_size,
            d_model=config.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            max_len=config.max_len,
            padding_idx=config.pad_id,
        )
        self.decoder = Decoder(
            vocab_size=config.tgt_vocab_size,
            d_model=config.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            max_len=config.max_len,
            padding_idx=config.pad_id,
        )
        self.generator = nn.Linear(config.d_model, config.tgt_vocab_size)

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        src_mask = make_src_mask(src, self.pad_id)
        memory, enc_attn = self.encoder(src, src_mask)
        return memory, src_mask, enc_attn

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        tgt_mask = make_tgt_mask(tgt, self.pad_id)
        out, self_attn, cross_attn = self.decoder(tgt, memory, tgt_mask, src_mask)
        return out, self_attn, cross_attn

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """src: (batch, src_len), tgt: (batch, tgt_len) -> logits (batch, tgt_len, tgt_vocab_size)

        `tgt` should be the decoder input (i.e. target sequence shifted right,
        teacher-forced during training).
        """
        memory, src_mask, _ = self.encode(src)
        decoder_out, _, _ = self.decode(tgt, memory, src_mask)
        return self.generator(decoder_out)

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        sos_id: int,
        eos_id: int,
        max_len: int = 100,
    ) -> torch.Tensor:
        """Autoregressively generate output ids one token at a time, always
        picking the highest-probability next token. src: (batch, src_len).
        Returns (batch, generated_len) token ids (including <sos>, excluding
        anything past the first <eos>).
        """
        self.eval()
        device = src.device
        batch_size = src.size(0)

        memory, src_mask, _ = self.encode(src)

        ys = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            decoder_out, _, _ = self.decode(ys, memory, src_mask)
            logits = self.generator(decoder_out[:, -1, :])  # (batch, vocab)
            next_token = logits.argmax(dim=-1, keepdim=True)  # (batch, 1)

            ys = torch.cat([ys, next_token], dim=1)
            finished = finished | (next_token.squeeze(1) == eos_id)
            if finished.all():
                break

        return ys

    @torch.no_grad()
    def beam_search_decode(
        self,
        src: torch.Tensor,
        sos_id: int,
        eos_id: int,
        max_len: int = 100,
        beam_size: int = 4,
        length_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Beam search decoding for a single example (batch size 1).

        Returns the best hypothesis as a (1, gen_len) LongTensor.
        """
        self.eval()
        device = src.device
        if src.size(0) != 1:
            raise ValueError("beam_search_decode currently supports batch size 1")

        memory, src_mask, _ = self.encode(src)

        # Each beam: (tokens tensor, cumulative log-prob)
        beams: List[Tuple[torch.Tensor, float]] = [(torch.tensor([[sos_id]], device=device), 0.0)]
        completed: List[Tuple[torch.Tensor, float]] = []

        for _ in range(max_len - 1):
            candidates: List[Tuple[torch.Tensor, float]] = []
            for tokens, score in beams:
                if tokens[0, -1].item() == eos_id:
                    completed.append((tokens, score))
                    continue

                decoder_out, _, _ = self.decode(tokens, memory, src_mask)
                logits = self.generator(decoder_out[:, -1, :])
                log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)  # (vocab,)

                topk_log_probs, topk_ids = log_probs.topk(beam_size)
                for lp, idx in zip(topk_log_probs.tolist(), topk_ids.tolist()):
                    new_tokens = torch.cat([tokens, torch.tensor([[idx]], device=device)], dim=1)
                    candidates.append((new_tokens, score + lp))

            if not candidates:
                break  # all beams already completed

            # Keep top `beam_size` candidates by length-normalized score.
            candidates.sort(key=lambda c: c[1] / (c[0].size(1) ** length_penalty), reverse=True)
            beams = candidates[:beam_size]

            if all(tokens[0, -1].item() == eos_id for tokens, _ in beams):
                completed.extend(beams)
                break

        completed.extend(beams)
        completed.sort(key=lambda c: c[1] / (c[0].size(1) ** length_penalty), reverse=True)
        return completed[0][0]
