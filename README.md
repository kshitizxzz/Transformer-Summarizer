# Transformer Summarizer

A from-scratch implementation of the Transformer architecture ([Vaswani et al., 2017 — *Attention Is All You Need*](https://arxiv.org/abs/1706.03762)) in PyTorch, applied to abstractive text summarization, with a Streamlit dashboard for interactive use, attention visualization, and training analytics.

## Architecture

- **Embeddings** (`src/model/embeddings.py`) — learned token embeddings scaled by `sqrt(d_model)`.
- **Positional encoding** (`src/model/positional_encoding.py`) — fixed sinusoidal position signals.
- **Scaled dot-product attention** (`src/model/attention.py`) — `softmax(QK^T / sqrt(d_k)) V`.
- **Multi-head attention** (`src/model/multi_head_attention.py`) — parallel attention heads, concatenated and projected.
- **Encoder / decoder layers** (`src/model/encoder_layer.py`, `decoder_layer.py`) — self-attention, cross-attention, and position-wise feed-forward sublayers with residual connections and layer norm.
- **Encoder / decoder stacks** (`src/model/encoder.py`, `decoder.py`) — `N` stacked layers.
- **Transformer** (`src/model/transformer.py`) — full encoder-decoder model, mask construction, and greedy / beam-search decoding.

## Project structure

```
Transformer-Summarizer/
├── data/                      # datasets + generated vocabulary
├── checkpoints/                # saved model weights
├── logs/                       # training history, eval results
├── src/
│   ├── preprocessing/          # tokenizer, vocabulary, dataset/dataloader
│   ├── model/                  # Transformer building blocks
│   ├── training/                # loss, LR schedule, training loop
│   ├── evaluation/              # ROUGE, perplexity, inference
│   └── visualization/           # attention heatmaps, training plots
├── ui/                          # Streamlit multipage app
│   ├── Home.py
│   └── pages/
├── notebooks/exploration.ipynb  # dataset exploration
└── requirements.txt
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

## Data

Place a CSV or JSON/JSONL file under `data/` with two text columns (defaults match the CNN/DailyMail convention: `article`, `highlights`). Any dataset with a long-text column and a short-summary column works — pass `--article_col` / `--summary_col` to override the names.

## Training

```bash
python -m src.training.train \
    --train_path data/train.csv \
    --val_path data/val.csv \
    --epochs 10 \
    --batch_size 32
```

This builds (or reuses) a vocabulary at `data/vocab.json`, trains the Transformer, and writes:

- per-epoch checkpoints + a `best.pt` to `checkpoints/`
- a JSON training history to `logs/training_history.json`

Key hyperparameters (`--d_model`, `--num_layers`, `--num_heads`, `--d_ff`, `--dropout`, `--lr`, `--warmup_steps`, `--label_smoothing`) are all CLI flags — see `python -m src.training.train --help`.

## Inference

```bash
python -m src.evaluation.inference \
    --checkpoint checkpoints/best.pt \
    --vocab_path data/vocab.json \
    --text "Your article text here..." \
    --method beam --beam_size 4
```

## Evaluation

`src/evaluation/rouge.py` and `src/evaluation/perplexity.py` provide dependency-free ROUGE-1/2/L and perplexity. Save aggregate results + example outputs to `logs/eval_results.json` to populate the **Results** page in the UI:

```json
{
  "rouge": {"rouge-1": {"precision": 0.0, "recall": 0.0, "f1": 0.0}, "rouge-2": {...}, "rouge-l": {...}},
  "perplexity": 23.4,
  "examples": [{"article": "...", "reference": "...", "generated": "..."}]
}
```

## Streamlit UI

```bash
streamlit run ui/Home.py
```

Pages: **Summarize** (generate a summary for pasted text), **Dashboard** (data/model/training status at a glance), **Attention Visualizer** (encoder self-attention, decoder self-attention, and cross-attention heatmaps per layer/head), **Training Analytics** (loss curves, LR schedule), **Results** (ROUGE/perplexity + example outputs), **About** (architecture details).

## Notebook

`notebooks/exploration.ipynb` walks through loading a sample of the dataset, tokenizing it, inspecting sequence-length distributions, and sanity-checking the vocabulary's encode/decode round trip.

## Reference

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention Is All You Need*. NeurIPS.
