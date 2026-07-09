"""Training loop, loss, LR scheduling, logging, and checkpointing for the Transformer."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model.transformer import Transformer


class LabelSmoothingLoss(nn.Module):
    """Cross-entropy with label smoothing (Szegedy et al.) over a padded
    vocabulary, ignoring `<pad>` positions in the loss.

    Implemented "from scratch" via KL-divergence against a smoothed target
    distribution, matching the original Transformer paper's recipe.
    """

    def __init__(self, vocab_size: int, pad_id: int = 0, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.criterion = nn.KLDivLoss(reduction="sum")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """logits: (N, vocab_size) raw scores; target: (N,) gold ids."""
        log_probs = torch.log_softmax(logits, dim=-1)

        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (self.vocab_size - 1))  # exclude pad + gold slot
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.pad_id] = 0.0

        pad_mask = target == self.pad_id
        true_dist.masked_fill_(pad_mask.unsqueeze(1), 0.0)

        num_tokens = (~pad_mask).sum().clamp(min=1)
        return self.criterion(log_probs, true_dist) / num_tokens


class NoamScheduler:
    """The learning-rate schedule from "Attention Is All You Need":

        lr = d_model^-0.5 * min(step^-0.5, step * warmup_steps^-1.5)

    i.e. linear warmup followed by inverse-square-root decay.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int, warmup_steps: int = 4000) -> None:
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self) -> float:
        self.step_num += 1
        lr = (self.d_model ** -0.5) * min(self.step_num ** -0.5, self.step_num * self.warmup_steps ** -1.5)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def state_dict(self) -> dict:
        return {"step_num": self.step_num, "warmup_steps": self.warmup_steps, "d_model": self.d_model}

    def load_state_dict(self, state: dict) -> None:
        self.step_num = state["step_num"]
        self.warmup_steps = state["warmup_steps"]
        self.d_model = state["d_model"]


class Trainer:
    """Encapsulates the training/validation loop, checkpointing, and
    structured logging (consumed later by `src.visualization.training_plots`).
    """

    def __init__(
        self,
        model: Transformer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        pad_id: int,
        device: Optional[Union[str, torch.device]] = None,
        lr: float = 1e-4,
        warmup_steps: int = 4000,
        label_smoothing: float = 0.1,
        checkpoint_dir: Union[str, Path] = "checkpoints",
        log_dir: Union[str, Path] = "logs",
        grad_clip: float = 1.0,
    ) -> None:
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.pad_id = pad_id
        self.grad_clip = grad_clip

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
        self.scheduler = NoamScheduler(self.optimizer, model.config.d_model, warmup_steps)
        self.criterion = LabelSmoothingLoss(model.config.tgt_vocab_size, pad_id, label_smoothing)

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[dict] = []

        self.global_step = 0
        self.best_val_loss = float("inf")

    # ------------------------------------------------------------------ #
    def _run_batch(self, batch: dict) -> torch.Tensor:
        src = batch["src"].to(self.device)
        tgt = batch["tgt"].to(self.device)

        decoder_input = tgt[:, :-1]   # teacher forcing input (shifted right)
        gold = tgt[:, 1:]             # what the model should predict

        logits = self.model(src, decoder_input)  # (batch, tgt_len-1, vocab)
        loss = self.criterion(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))
        return loss

    def train_epoch(self, epoch: int, log_every: int = 50) -> float:
        self.model.train()
        total_loss, total_batches = 0.0, 0

        for i, batch in enumerate(self.train_loader, start=1):
            self.optimizer.zero_grad()
            loss = self._run_batch(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            total_batches += 1
            self.global_step += 1

            if i % log_every == 0:
                print(f"epoch {epoch} | step {i}/{len(self.train_loader)} | loss {loss.item():.4f}")

        return total_loss / max(total_batches, 1)

    @torch.no_grad()
    def validate(self) -> float:
        if self.val_loader is None:
            return float("nan")

        self.model.eval()
        total_loss, total_batches = 0.0, 0
        for batch in self.val_loader:
            loss = self._run_batch(batch)
            total_loss += loss.item()
            total_batches += 1

        return total_loss / max(total_batches, 1)

    def fit(self, num_epochs: int) -> list[dict]:
        for epoch in range(1, num_epochs + 1):
            start = time.time()
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            elapsed = time.time() - start

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": self.optimizer.param_groups[0]["lr"],
                "elapsed_sec": elapsed,
            }
            self.history.append(record)
            self._write_log()
            print(f"[epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} ({elapsed:.1f}s)")

            self.save_checkpoint(self.checkpoint_dir / f"epoch_{epoch}.pt")

            # Without a val_loader, val_loss is nan; fall back to train_loss so
            # best.pt still gets written (the UI and inference CLI default to it).
            tracked_loss = train_loss if math.isnan(val_loss) else val_loss
            if not (tracked_loss != tracked_loss) and tracked_loss < self.best_val_loss:
                self.best_val_loss = tracked_loss
                self.save_checkpoint(self.checkpoint_dir / "best.pt")

        return self.history

    # ------------------------------------------------------------------ #
    def _write_log(self) -> None:
        with open(self.log_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "config": self.model.config,
                "global_step": self.global_step,
                "best_val_loss": self.best_val_loss,
            },
            path,
        )

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        # weights_only=False: checkpoints store a TransformerConfig object, not just tensors.
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
