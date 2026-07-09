"""Training Analytics: loss curves and learning-rate schedule from training logs."""

import sys
import math
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.training_plots import load_history, plot_learning_rate, plot_loss_curves

st.set_page_config(page_title="Training Analytics", page_icon="📈", layout="wide")
st.title("📈 Training Analytics")

log_dir = PROJECT_ROOT / "logs"
history = load_history(log_dir)

if not history:
    st.info(
        "No training history found yet. Train the model first:\n\n"
        "`python -m src.training.train --train_path data/train.csv --val_path data/val.csv`"
    )
else:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_loss_curves(history), use_container_width=True)
    with col2:
        st.plotly_chart(plot_learning_rate(history), use_container_width=True)

    valid_perp = [r for r in history if isinstance(r.get("val_loss"), float) and not math.isnan(r["val_loss"])]
    if valid_perp:
        st.subheader("Perplexity over epochs")
        fig_perp = go.Figure()
        fig_perp.add_trace(go.Scatter(
            x=[r["epoch"] for r in valid_perp],
            y=[math.exp(min(r["val_loss"], 10)) for r in valid_perp],
            mode="lines+markers",
            name="perplexity",
            line=dict(color="orange"),
        ))
        fig_perp.update_layout(
            title="Validation perplexity over epochs",
            xaxis_title="Epoch",
            yaxis_title="Perplexity"
        )
        st.plotly_chart(fig_perp, use_container_width=True)

    st.subheader("Raw history")
    st.dataframe(history, use_container_width=True)
