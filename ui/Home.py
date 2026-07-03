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
text summarization on CNN/DailyMail. Built without any pre-built
Transformer library — encoder, decoder, multi-head attention,
positional encodings, and all evaluation metrics implemented from scratch.

**Also includes:** TF-IDF extractive baseline, Bi-LSTM + Bahdanau attention baseline,
Logistic Regression sentiment analysis, Random Forest topic classifier, and NER preservation analysis.
"""
)

col1, col2, col3, col4 = st.columns(4)

checkpoint_path = PROJECT_ROOT / "checkpoints" / "best.pt"
vocab_path      = PROJECT_ROOT / "data" / "vocab.json"
history_path    = PROJECT_ROOT / "logs" / "training_history.json"
results_path    = PROJECT_ROOT / "logs" / "eval_results.json"

with col1:
    st.metric("Transformer Checkpoint", "Found ✓" if checkpoint_path.exists() else "Not found")
with col2:
    st.metric("Vocabulary (8K tokens)", "Built ✓" if vocab_path.exists() else "Not built")
with col3:
    st.metric("Training Logs", "Available ✓" if history_path.exists() else "None yet")
with col4:
    st.metric("Eval Results", "Available ✓" if results_path.exists() else "None yet")

st.divider()

st.subheader("Project Architecture")
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
**Main Model: Transformer**
- 11.68M parameters, from scratch in PyTorch
- d_model=256, 4 heads, d_ff=1024
- 3 encoder + 3 decoder layers
- Sinusoidal positional encoding
- Xavier initialization, Adam optimizer
- Gradient clipping, Dropout
- Teacher forcing during training
""")
with col2:
    st.markdown("""
**Baselines & Analysis**
- TF-IDF extractive baseline (classical NLP)
- Bi-LSTM + Bahdanau attention baseline (RNN-era)
- Logistic Regression sentiment consistency
- Random Forest topic classifier (5-fold CV)
- NER entity preservation analysis
- N-gram overlap evaluation (unigram/bigram/trigram F1)
""")

st.divider()

st.subheader("Navigate")
st.markdown(
    """
- **Summarize** — generate a summary with the trained Transformer
- **Dashboard** — data, vocab, training status overview
- **Attention Visualizer** — encoder/decoder attention heatmaps
- **Training Analytics** — loss curves per epoch
- **Results** — N-gram overlap metrics, baseline comparison, examples
- **About** — architecture and references
- **Baseline Comparison** — TF-IDF vs Bi-LSTM vs Transformer
- **Sentiment Analysis** — Logistic Regression sentiment consistency
- **Topic Classifier** — Random Forest news topic classification
- **NER Analysis** — named entity preservation analysis
"""
)

if not checkpoint_path.exists():
    st.info(
        "No trained checkpoint found. Train with:\n\n"
        "`python -m src.training.train --train_path data/train.csv --val_path data/val.csv`"
    )
