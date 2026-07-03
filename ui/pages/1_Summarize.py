"""Summarize page: paste text in, get an abstractive summary out."""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.inference import Summarizer  # noqa: E402

st.set_page_config(page_title="Summarize", page_icon="✍️")
st.title("✍️ Summarize")

checkpoint_path = PROJECT_ROOT / "checkpoints" / "best.pt"
vocab_path = PROJECT_ROOT / "data" / "vocab.json"


@st.cache_resource(show_spinner="Loading model...")
def load_summarizer(checkpoint: str, vocab: str):
    return Summarizer(checkpoint, vocab)


if not checkpoint_path.exists() or not vocab_path.exists():
    st.warning(
        "No trained checkpoint / vocabulary found yet. Train the model first:\n\n"
        "`python -m src.training.train --train_path data/train.csv`"
    )
else:
    with st.sidebar:
        st.subheader("Decoding options")
        method = st.radio("Method", ["greedy", "beam"], index=0)
        beam_size = st.slider("Beam size", 2, 10, 4, disabled=(method != "beam"))
        max_summary_len = st.slider("Max summary length", 20, 200, 100)
        no_repeat_ngram_size = st.slider(
            "Block repeated n-grams of size",
            0,
            6,
            3,
            help="Prevents the model from looping on the same phrase (e.g. "
            "'ever ever ever...'). 0 disables this guard.",
        )

    text = st.text_area("Paste an article to summarize", height=300, placeholder="Paste article text here...")

    if st.button("Generate summary", type="primary", disabled=not text.strip()):
        summarizer = load_summarizer(str(checkpoint_path), str(vocab_path))
        with st.spinner("Generating..."):
            summary = summarizer.summarize(
                text,
                method=method,
                beam_size=beam_size,
                max_summary_len=max_summary_len,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
        st.subheader("Summary")
        st.success(summary)
        input_words = len(text.split())
        summary_words = len(summary.split())
        ratio = round(input_words / max(summary_words, 1), 1)
        col1, col2, col3 = st.columns(3)
        col1.metric("Input words", input_words)
        col2.metric("Summary words", summary_words)
        col3.metric("Compression ratio", f"{ratio}x")
