"""About page: architecture overview, tech stack, and references."""

import streamlit as st

st.set_page_config(page_title="About", page_icon="ℹ️")
st.title("ℹ️ About")

st.markdown(
    """
### Transformer Summarizer

This project implements the Transformer encoder-decoder architecture
from scratch in PyTorch and applies it to abstractive text
summarization (article → short summary).

#### Architecture

- **Embeddings** — learned token embeddings scaled by `sqrt(d_model)`.
- **Positional encoding** — fixed sinusoidal position signals.
- **Scaled dot-product attention** — the core `softmax(QK^T / sqrt(d_k))V` primitive.
- **Multi-head attention** — `h` parallel attention heads, concatenated and projected.
- **Encoder** — a stack of self-attention + feed-forward blocks (with residual
  connections and layer normalization) that turns the source article into a
  contextualized representation.
- **Decoder** — a stack of masked self-attention + cross-attention (over the
  encoder output) + feed-forward blocks that autoregressively generates the summary.
- **Generator** — a final linear + softmax layer projecting decoder states to
  vocabulary logits.

#### Training

- Label-smoothed cross-entropy loss.
- Noam learning-rate schedule (linear warmup, then inverse-square-root decay).
- Teacher forcing during training; greedy or beam search at inference time.

#### Evaluation

- ROUGE-1 / ROUGE-2 / ROUGE-L (implemented from scratch).
- Perplexity.

#### Tech stack

PyTorch · Streamlit · Plotly · NumPy / Pandas

#### Reference

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
Kaiser, Ł., & Polosukhin, I. (2017). *Attention Is All You Need*. NeurIPS.
"""
)
