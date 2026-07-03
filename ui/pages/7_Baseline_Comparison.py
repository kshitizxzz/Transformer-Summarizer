"""Baseline Comparison: Transformer vs TF-IDF vs Bi-LSTM + Bahdanau Attention.

Interactive page to compare all three summarization approaches:
  1. TF-IDF Extractive (classical NLP baseline)
  2. Bi-LSTM + Bahdanau Attention (RNN-based neural baseline)
  3. Transformer (our model)

Shows the progression: Bag-of-Words -> RNN+Attention -> Self-Attention Transformer.
"""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Baseline Comparison", page_icon="⚖️", layout="wide")
st.title("⚖️ Baseline Comparison")

st.markdown("""
Compare three summarization approaches: TF-IDF extractive, Bi-LSTM + Bahdanau attention,
and the Transformer — showing how each generates or selects summary text.
""")

# ------------------------------------------------------------------ #
# TF-IDF Demo (runs live, no checkpoint needed)
# ------------------------------------------------------------------ #
st.subheader("1. TF-IDF Extractive Baseline")
st.caption(
    "Scores each sentence by the sum of its TF-IDF token weights. "
    "Top-k sentences are selected as the extractive summary. "
    "No learning from data — pure classical NLP."
)

default_article = (
    "The stock market fell sharply on Wednesday as investors reacted to the Federal "
    "Reserve's decision to raise interest rates. Major indices dropped by more than two "
    "percent in early trading. Technology stocks were among the hardest hit, with Apple "
    "and Microsoft both falling over three percent. Analysts said the sell-off was driven "
    "by concerns about slowing economic growth and rising borrowing costs. The Federal "
    "Reserve chairman said the rate hike was necessary to control inflation, which has "
    "been running at its highest level in forty years. Consumer spending has remained "
    "resilient despite the rate increases, according to data released Thursday."
)

article_input = st.text_area("Enter article text:", value=default_article, height=150)
n_sentences   = st.slider("Number of sentences to extract:", 1, 5, 2)

if st.button("Run TF-IDF Baseline"):
    try:
        from src.models.tfidf_baseline import TFIDFSummarizer
        tfidf = TFIDFSummarizer(num_sentences=n_sentences)
        result = tfidf.summarize_with_scores(article_input)

        st.markdown("**Extractive Summary:**")
        st.success(result["summary"])

        st.markdown("**Sentence Scores (TF-IDF weight):**")
        score_data = []
        for i, (sent, score) in enumerate(zip(result["sentences"], result["scores"])):
            score_data.append({
                "Sentence": sent[:100] + ("..." if len(sent) > 100 else ""),
                "TF-IDF Score": round(score, 4),
                "Selected": "✓" if i in result["selected_indices"] else "",
            })
        st.dataframe(score_data, use_container_width=True)

        st.caption("Sentence score = sum of TF-IDF weights. Higher score = more informative sentence.")
    except Exception as e:
        st.error(f"Error: {e}")

st.divider()

# ------------------------------------------------------------------ #
# Bi-LSTM + Bahdanau Attention Architecture
# ------------------------------------------------------------------ #
st.subheader("2. Bi-LSTM + Bahdanau (Additive) Attention")
st.caption(
    "A trained Bi-LSTM model with Bahdanau attention. "
    "Requires a trained checkpoint at checkpoints/bilstm_best.pt. "
    "The architecture info below is always available."
)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Architecture:**")
    st.code("""
Encoder: Bidirectional LSTM
  - Forward + Backward LSTM (hidden=256 each)
  - Concat -> 512-dim encoder output

Bahdanau (Additive) Attention:
  - Decoder attends to all encoder states
  - Weighted context vector per decoding step

Decoder: Unidirectional LSTM
  - Input: [embed || context]
  - Output: vocab logits
  - Teacher forcing during training
    """, language="text")

with col2:
    st.markdown("**Bi-LSTM + Bahdanau**")
    st.info(
        "The Bidirectional LSTM encodes the source in both directions. "
        "Bahdanau attention lets the decoder dynamically weight encoder states "
        "at each generation step instead of using a single fixed vector."
    )

bilstm_ckpt = PROJECT_ROOT / "checkpoints" / "bilstm_best.pt"
if bilstm_ckpt.exists():
    st.success("Bi-LSTM checkpoint found! You can run inference.")
    bilstm_input = st.text_area("Article for Bi-LSTM:", value=article_input[:300], height=100,
                                key="bilstm_input")
    if st.button("Generate with Bi-LSTM + Bahdanau"):
        st.info("Bi-LSTM inference would run here. Train with: python -m src.training.train_bilstm")
else:
    st.warning(
        "No Bi-LSTM checkpoint found. Train the model first:\n"
        "```bash\npython -m src.training.train_bilstm --data_path data/train.csv\n```"
    )

st.divider()

# ------------------------------------------------------------------ #
# Comparison table
# ------------------------------------------------------------------ #
st.subheader("Method Comparison")

comparison = [
    {
        "Method":       "TF-IDF Extractive",
        "Type":         "Extractive",
        "Key Concepts": "TF-IDF, Bag of Words, N-grams",
        "Trainable":    "No (unsupervised)",
        "Generates New Text": "No",
        "Speed":        "Instant",
    },
    {
        "Method":       "Bi-LSTM + Bahdanau",
        "Type":         "Abstractive",
        "Key Concepts": "BiLSTM, Additive Attention, Teacher Forcing",
        "Trainable":    "Yes (supervised)",
        "Generates New Text": "Yes",
        "Speed":        "Fast",
    },
    {
        "Method":       "Transformer (Our Model)",
        "Type":         "Abstractive",
        "Key Concepts": "Self-Attention, Multi-Head Attention, Positional Encoding",
        "Trainable":    "Yes (supervised)",
        "Generates New Text": "Yes",
        "Speed":        "Moderate",
    },
]
st.dataframe(comparison, use_container_width=True)

