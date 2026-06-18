"""Plots training/validation loss curves and learning-rate schedule from
the JSON history written by `src.training.trainer.Trainer`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Union

import plotly.graph_objects as go


def load_history(log_dir: Union[str, Path] = "logs", filename: str = "training_history.json") -> List[dict]:
    """Load the list of per-epoch records written during training."""
    path = Path(log_dir) / filename
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plot_loss_curves(history: List[dict], title: str = "Training & validation loss") -> go.Figure:
    """Line chart of train_loss and val_loss vs. epoch."""
    epochs = [r["epoch"] for r in history]
    train_loss = [r.get("train_loss") for r in history]
    val_loss = [r.get("val_loss") for r in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_loss, mode="lines+markers", name="train loss"))
    fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode="lines+markers", name="val loss"))
    fig.update_layout(title=title, xaxis_title="Epoch", yaxis_title="Loss")
    return fig


def plot_learning_rate(history: List[dict], title: str = "Learning rate schedule") -> go.Figure:
    """Line chart of the (Noam-scheduled) learning rate vs. epoch."""
    epochs = [r["epoch"] for r in history]
    lr = [r.get("lr") for r in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=lr, mode="lines+markers", name="learning rate"))
    fig.update_layout(title=title, xaxis_title="Epoch", yaxis_title="LR")
    return fig


def plot_rouge_scores(rouge_history: List[dict], title: str = "ROUGE over epochs") -> go.Figure:
    """Line chart of rouge-1/2/l F1 vs. epoch, if such a log is available.

    Expects each record to look like:
        {"epoch": int, "rouge-1": float, "rouge-2": float, "rouge-l": float}
    """
    epochs = [r["epoch"] for r in rouge_history]

    fig = go.Figure()
    for metric in ("rouge-1", "rouge-2", "rouge-l"):
        if any(metric in r for r in rouge_history):
            fig.add_trace(
                go.Scatter(x=epochs, y=[r.get(metric) for r in rouge_history], mode="lines+markers", name=metric)
            )
    fig.update_layout(title=title, xaxis_title="Epoch", yaxis_title="F1 score")
    return fig
