"""Bi-LSTM Encoder + Bahdanau (Additive) Attention Decoder.

This module implements a sequence-to-sequence model using:
  - Bidirectional LSTM encoder  (deep.pdf: Bidirectional LSTM)
  - Bahdanau additive attention  (deep.pdf: Attention Mechanism / Bahdanau 2014)
  - Unidirectional LSTM decoder

This serves as a baseline to compare against the Transformer. The comparison
shows the evolution: RNN -> Attention -> Transformer (self-attention).

Architecture
------------
Encoder:
    Embedding -> Bidirectional LSTM
    Output: encoder_outputs (seq, batch, 2*hidden), hidden state

Bahdanau Attention (Additive):
    score(s_t, h_i) = v^T * tanh(W_a * h_i + U_a * s_t)
    alpha_t = softmax(score)
    context_t = sum(alpha_t * h_i)

Decoder:
    Embedding + context_t -> LSTM cell -> Linear -> vocab logits
    (Teacher forcing during training)

Usage
-----
    from src.models.bilstm_bahdanau import BiLSTMBahdanauSeq2Seq, BiLSTMConfig

    config = BiLSTMConfig(vocab_size=8000, embed_dim=128, hidden_dim=256)
    model = BiLSTMBahdanauSeq2Seq(config)

    # Training step
    logits = model(src, tgt, teacher_forcing_ratio=0.5)

    # Inference
    output_ids = model.generate(src, sos_id=2, eos_id=3, max_len=50)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class BiLSTMConfig:
    vocab_size: int
    embed_dim: int = 128
    hidden_dim: int = 256      # per direction in encoder; decoder uses 2*hidden_dim
    num_encoder_layers: int = 2
    num_decoder_layers: int = 1
    dropout: float = 0.3
    pad_id: int = 0


# ---------------------------------------------------------------------------
# Bahdanau (Additive) Attention
# ---------------------------------------------------------------------------

class BahdanauAttention(nn.Module):
    """Additive attention as described in Bahdanau et al. (2014).

    score(s, h) = v^T * tanh(W_a * h + U_a * s)
    alpha       = softmax(score)
    context     = alpha @ encoder_outputs

    Parameters
    ----------
    encoder_dim : int -- encoder output dimension (2*hidden for BiLSTM)
    decoder_dim : int -- decoder hidden state dimension
    attn_dim    : int -- internal projection dimension
    """

    def __init__(self, encoder_dim: int, decoder_dim: int, attn_dim: int = 256) -> None:
        super().__init__()
        # W_a projects encoder outputs
        self.W_a = nn.Linear(encoder_dim, attn_dim, bias=False)
        # U_a projects decoder hidden state
        self.U_a = nn.Linear(decoder_dim, attn_dim, bias=False)
        # v^T: maps to scalar score
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,       # (batch, decoder_dim)
        encoder_outputs: torch.Tensor,      # (batch, src_len, encoder_dim)
        src_mask: Optional[torch.Tensor] = None,  # (batch, src_len) bool, True=valid
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        context : (batch, encoder_dim)  -- weighted sum of encoder outputs
        alpha   : (batch, src_len)      -- attention weights
        """
        src_len = encoder_outputs.size(1)

        # Expand decoder hidden: (batch, 1, decoder_dim) -> broadcast over src_len
        dec_h = decoder_hidden.unsqueeze(1).expand(-1, src_len, -1)  # (batch, src_len, dec_dim)

        # Bahdanau score: tanh(W_a*h + U_a*s)
        energy = self.v(
            torch.tanh(self.W_a(encoder_outputs) + self.U_a(dec_h))
        ).squeeze(-1)  # (batch, src_len)

        # Mask padding positions with -inf before softmax
        if src_mask is not None:
            energy = energy.masked_fill(~src_mask, float("-inf"))

        alpha = F.softmax(energy, dim=-1)  # (batch, src_len)

        # Context vector
        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch, enc_dim)

        return context, alpha


# ---------------------------------------------------------------------------
# Encoder: Bidirectional LSTM
# ---------------------------------------------------------------------------

class BiLSTMEncoder(nn.Module):
    """Bidirectional LSTM encoder.

    Encodes the source sequence into a sequence of hidden states by running
    an LSTM in both directions and concatenating the outputs.

    Output dimension per timestep: 2 * hidden_dim
    """

    def __init__(self, config: BiLSTMConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            config.vocab_size, config.embed_dim, padding_idx=config.pad_id
        )
        self.lstm = nn.LSTM(
            input_size=config.embed_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_encoder_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.num_encoder_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(config.dropout)

        # Project concatenated bidirectional hidden/cell to decoder size
        self.hidden_proj = nn.Linear(2 * config.hidden_dim, 2 * config.hidden_dim)
        self.cell_proj   = nn.Linear(2 * config.hidden_dim, 2 * config.hidden_dim)

    def forward(
        self, src: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        src : (batch, src_len)

        Returns
        -------
        encoder_outputs : (batch, src_len, 2*hidden_dim)
        (h_n, c_n)      : each (num_dec_layers, batch, 2*hidden_dim) -- for decoder init
        """
        embedded = self.dropout(self.embedding(src))          # (batch, src_len, embed_dim)
        outputs, (h_n, c_n) = self.lstm(embedded)             # outputs: (batch, src_len, 2*hid)

        # Concatenate forward and backward last hidden/cell states
        # h_n shape: (num_layers*2, batch, hidden_dim)
        # Take last layer forward + backward
        h_fwd = h_n[-2]   # (batch, hidden_dim)
        h_bwd = h_n[-1]   # (batch, hidden_dim)
        c_fwd = c_n[-2]
        c_bwd = c_n[-1]

        h_combined = torch.tanh(self.hidden_proj(torch.cat([h_fwd, h_bwd], dim=-1)))  # (batch, 2*hid)
        c_combined = torch.tanh(self.cell_proj(torch.cat([c_fwd, c_bwd], dim=-1)))

        # Unsqueeze to match num_layers=1 decoder
        return outputs, (h_combined.unsqueeze(0), c_combined.unsqueeze(0))


# ---------------------------------------------------------------------------
# Decoder: LSTM + Bahdanau Attention
# ---------------------------------------------------------------------------

class BahdanauDecoder(nn.Module):
    """LSTM decoder with Bahdanau attention over encoder outputs.

    At each step t:
        1. Embed target token
        2. Compute Bahdanau attention over encoder_outputs using prev hidden state
        3. Concatenate embedding + context vector -> LSTM input
        4. LSTM step -> new hidden state
        5. Linear projection -> vocab logits
    """

    def __init__(self, config: BiLSTMConfig) -> None:
        super().__init__()
        encoder_dim = 2 * config.hidden_dim
        decoder_dim = 2 * config.hidden_dim

        self.embedding = nn.Embedding(
            config.vocab_size, config.embed_dim, padding_idx=config.pad_id
        )
        self.attention = BahdanauAttention(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            attn_dim=encoder_dim,
        )
        self.lstm = nn.LSTM(
            input_size=config.embed_dim + encoder_dim,
            hidden_size=decoder_dim,
            num_layers=config.num_decoder_layers,
            batch_first=True,
        )
        self.fc_out = nn.Linear(decoder_dim + encoder_dim + config.embed_dim, config.vocab_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward_step(
        self,
        tgt_token: torch.Tensor,                        # (batch,)
        prev_hidden: Tuple[torch.Tensor, torch.Tensor], # LSTM (h, c)
        encoder_outputs: torch.Tensor,                  # (batch, src_len, enc_dim)
        src_mask: Optional[torch.Tensor],               # (batch, src_len)
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Single decoder step.

        Returns
        -------
        logits      : (batch, vocab_size)
        new_hidden  : updated LSTM (h, c)
        alpha       : (batch, src_len) attention weights
        """
        embedded = self.dropout(self.embedding(tgt_token.unsqueeze(1)))  # (batch, 1, embed_dim)

        # Bahdanau attention using top-layer hidden state
        h_top = prev_hidden[0][-1]  # (batch, dec_dim)
        context, alpha = self.attention(h_top, encoder_outputs, src_mask)  # (batch, enc_dim)

        # LSTM input: [embedding || context]
        lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)  # (batch, 1, embed+enc)
        lstm_out, new_hidden = self.lstm(lstm_input, prev_hidden)          # (batch, 1, dec_dim)
        lstm_out = lstm_out.squeeze(1)                                      # (batch, dec_dim)

        # Predict next token
        pred_input = torch.cat([lstm_out, context, embedded.squeeze(1)], dim=-1)
        logits = self.fc_out(self.dropout(pred_input))                     # (batch, vocab_size)

        return logits, new_hidden, alpha


# ---------------------------------------------------------------------------
# Full Seq2Seq Model
# ---------------------------------------------------------------------------

class BiLSTMBahdanauSeq2Seq(nn.Module):
    """Full encoder-decoder seq2seq with Bidirectional LSTM + Bahdanau attention.

    This is the baseline model against which the Transformer is compared.
    """

    def __init__(self, config: BiLSTMConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = BiLSTMEncoder(config)
        self.decoder = BahdanauDecoder(config)
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier initialization for linear layers (same as Transformer baseline)."""
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self,
        src: torch.Tensor,                              # (batch, src_len)
        tgt: torch.Tensor,                              # (batch, tgt_len)
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        src : source token ids
        tgt : target token ids (first token = <sos>)
        teacher_forcing_ratio : probability of using ground truth at each step

        Returns
        -------
        logits : (batch, tgt_len-1, vocab_size)
        """
        batch_size, tgt_len = tgt.shape
        vocab_size = self.config.vocab_size

        src_mask = (src != self.config.pad_id)  # (batch, src_len)

        encoder_outputs, hidden = self.encoder(src)  # enc_out: (batch, src_len, 2*hid)

        # Decode step by step
        dec_input = tgt[:, 0]  # <sos> token
        all_logits = []

        for t in range(1, tgt_len):
            logits, hidden, _ = self.decoder.forward_step(
                dec_input, hidden, encoder_outputs, src_mask
            )
            all_logits.append(logits)

            # Teacher forcing: use ground truth token or model's prediction
            use_teacher = torch.rand(1).item() < teacher_forcing_ratio
            if use_teacher:
                dec_input = tgt[:, t]
            else:
                dec_input = logits.argmax(dim=-1)

        return torch.stack(all_logits, dim=1)  # (batch, tgt_len-1, vocab_size)

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,          # (batch, src_len)
        sos_id: int,
        eos_id: int,
        max_len: int = 100,
    ) -> torch.Tensor:
        """Greedy decoding for inference.

        Returns
        -------
        output_ids : (batch, generated_len)
        """
        batch_size = src.size(0)
        src_mask = (src != self.config.pad_id)

        encoder_outputs, hidden = self.encoder(src)

        dec_input = torch.full((batch_size,), sos_id, dtype=torch.long, device=src.device)
        generated = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)

        for _ in range(max_len):
            logits, hidden, _ = self.decoder.forward_step(
                dec_input, hidden, encoder_outputs, src_mask
            )
            next_token = logits.argmax(dim=-1)
            generated.append(next_token)
            finished |= (next_token == eos_id)
            if finished.all():
                break
            dec_input = next_token

        return torch.stack(generated, dim=1)  # (batch, gen_len)

    def count_parameters(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = BiLSTMConfig(vocab_size=8000, embed_dim=128, hidden_dim=256)
    model = BiLSTMBahdanauSeq2Seq(cfg)
    print(f"BiLSTM+Bahdanau parameters: {model.count_parameters():,}")

    # Dummy forward
    src = torch.randint(1, 100, (2, 20))
    tgt = torch.randint(1, 100, (2, 10))
    logits = model(src, tgt, teacher_forcing_ratio=1.0)
    print(f"Logits shape: {logits.shape}")  # (2, 9, 8000)

    # Dummy inference
    out = model.generate(src, sos_id=2, eos_id=3, max_len=15)
    print(f"Generated shape: {out.shape}")
