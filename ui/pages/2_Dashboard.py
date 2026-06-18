"""Dashboard: at-a-glance view of data, model configuration, and training status."""

import json
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
st.title("📊 Dashboard")

data_dir = PROJECT_ROOT / "data"
checkpoint_dir = PROJECT_ROOT / "checkpoints"
log_dir = PROJECT_ROOT / "logs"
vocab_path = data_dir / "vocab.json"
history_path = log_dir / "training_history.json"

st.subheader("Data")
data_files = [f for f in data_dir.glob("*") if f.is_file() and f.name != ".gitkeep"]
if data_files:
    for f in data_files:
        st.write(f"- `{f.name}` — {f.stat().st_size / 1024:.1f} KB")
else:
    st.caption("No data files found in `data/` yet.")

st.subheader("Vocabulary")
if vocab_path.exists():
    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)
    st.write(f"Vocabulary size: **{len(vocab):,}** tokens")
else:
    st.caption("Vocabulary not built yet — run training to generate `data/vocab.json`.")

st.subheader("Checkpoints")
checkpoints = sorted(checkpoint_dir.glob("*.pt"))
if checkpoints:
    for ckpt in checkpoints:
        st.write(f"- `{ckpt.name}` — {ckpt.stat().st_size / (1024 * 1024):.1f} MB")
else:
    st.caption("No checkpoints saved yet.")

st.subheader("Training status")
if history_path.exists():
    with open(history_path, encoding="utf-8") as f:
        history = json.load(f)
    if history:
        latest = history[-1]
        val_loss = latest.get("val_loss")
        val_loss_display = f"{val_loss:.4f}" if isinstance(val_loss, (int, float)) and val_loss == val_loss else "n/a"

        col1, col2, col3 = st.columns(3)
        col1.metric("Epochs completed", latest["epoch"])
        col2.metric("Train loss", f"{latest['train_loss']:.4f}")
        col3.metric("Val loss", val_loss_display)
        st.dataframe(history, use_container_width=True)
else:
    st.caption("No training history yet — see the **Training Analytics** page once you've trained a model.")
