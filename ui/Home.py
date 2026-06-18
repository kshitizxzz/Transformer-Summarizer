"""Landing page for the Transformer Summarizer Streamlit app."""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Transformer Summarizer", page_icon="📝", layout="wide")

st.title("📝 Transformer Summarizer")
st.markdown(
    """
A from-scratch implementation of the Transformer architecture
(*Attention Is All You Need*, Vaswani et al., 2017) for abstractive
text summarization — encoder, decoder, multi-head attention,
positional encodings, training loop, and evaluation, all built without
relying on a pre-built Transformer library.
"""
)

col1, col2, col3 = st.columns(3)

checkpoint_path = PROJECT_ROOT / "checkpoints" / "best.pt"
vocab_path = PROJECT_ROOT / "data" / "vocab.json"
history_path = PROJECT_ROOT / "logs" / "training_history.json"

with col1:
    st.metric("Trained checkpoint", "Found" if checkpoint_path.exists() else "Not found")
with col2:
    st.metric("Vocabulary", "Built" if vocab_path.exists() else "Not built")
with col3:
    st.metric("Training logs", "Available" if history_path.exists() else "None yet")

st.divider()

st.subheader("Get started")
st.markdown(
    """
Use the sidebar to navigate:

- **Summarize** — generate a summary for your own text using a trained checkpoint.
- **Dashboard** — high-level overview of data, model configuration, and status.
- **Attention Visualizer** — inspect encoder/decoder attention heatmaps.
- **Training Analytics** — loss curves and learning-rate schedule.
- **Results** — ROUGE / perplexity scores and example outputs.
- **About** — architecture details and references.
"""
)

if not checkpoint_path.exists():
    st.info(
        "No trained checkpoint found yet. Train one with:\n\n"
        "`python -m src.training.train --train_path data/train.csv --val_path data/val.csv`"
    )
