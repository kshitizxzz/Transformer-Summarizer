"""Attention Visualizer: inspect encoder/decoder attention heatmaps for a given input."""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.inference import Summarizer  # noqa: E402
from src.visualization.attention_visualizer import plot_attention_grid, plot_attention_heatmap  # noqa: E402

st.set_page_config(page_title="Attention Visualizer", page_icon="🔍", layout="wide")
st.title("🔍 Attention Visualizer")

checkpoint_path = PROJECT_ROOT / "checkpoints" / "best.pt"
vocab_path = PROJECT_ROOT / "data" / "vocab.json"


@st.cache_resource(show_spinner="Loading model...")
def load_summarizer(checkpoint: str, vocab: str):
    return Summarizer(checkpoint, vocab)


if not checkpoint_path.exists() or not vocab_path.exists():
    st.warning("No trained checkpoint / vocabulary found yet. Train the model first.")
else:
    summarizer = load_summarizer(str(checkpoint_path), str(vocab_path))

    text = st.text_area(
        "Text to analyze", height=200, placeholder="Paste a short passage to keep the heatmap readable..."
    )

    if st.button("Run model & visualize attention", type="primary", disabled=not text.strip()):
        with st.spinner("Running encoder/decoder..."):
            maps = summarizer.get_attention_maps(text)
        st.session_state["attn_maps"] = maps

    maps = st.session_state.get("attn_maps")
    if maps:
        num_layers = len(maps["encoder_self_attention"])
        num_heads = maps["encoder_self_attention"][0].shape[1]

        attn_type = st.selectbox(
            "Attention type",
            ["Encoder self-attention", "Decoder self-attention", "Decoder cross-attention"],
        )
        layer = st.slider("Layer", 0, num_layers - 1, 0)
        view = st.radio("View", ["Single head", "All heads"], horizontal=True)

        type_map = {
            "Encoder self-attention": ("encoder_self_attention", maps["src_tokens"], maps["src_tokens"]),
            "Decoder self-attention": ("decoder_self_attention", maps["tgt_tokens"], maps["tgt_tokens"]),
            "Decoder cross-attention": ("decoder_cross_attention", maps["tgt_tokens"], maps["src_tokens"]),
        }
        key, query_tokens, key_tokens = type_map[attn_type]
        attn_weights = maps[key]

        if view == "Single head":
            head = st.slider("Head", 0, num_heads - 1, 0)
            fig = plot_attention_heatmap(attn_weights, query_tokens, key_tokens, layer=layer, head=head, title=attn_type)
        else:
            fig = plot_attention_grid(attn_weights, query_tokens, key_tokens, layer=layer, title=attn_type)

        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Generated summary tokens: {' '.join(maps['tgt_tokens'])}")
    else:
        st.info("Enter text and click **Run model & visualize attention** to see heatmaps.")
