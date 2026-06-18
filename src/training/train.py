"""CLI entry point for training the Transformer summarizer.

Example
-------
    python -m src.training.train \\
        --train_path data/train.csv \\
        --val_path data/val.csv \\
        --epochs 10 --batch_size 32
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.model.transformer import Transformer, TransformerConfig
from src.preprocessing.dataset import SummarizationDataset, get_dataloader
from src.preprocessing.tokenizer import SimpleTokenizer
from src.preprocessing.vocabulary import Vocabulary
from src.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Transformer summarization model")

    parser.add_argument("--train_path", type=str, required=True, help="Path to training data (csv/json/jsonl)")
    parser.add_argument("--val_path", type=str, default=None, help="Path to validation data")
    parser.add_argument("--article_col", type=str, default="article")
    parser.add_argument("--summary_col", type=str, default="highlights")

    parser.add_argument("--vocab_path", type=str, default="data/vocab.json")
    parser.add_argument("--max_vocab_size", type=int, default=8000)
    parser.add_argument("--min_freq", type=int, default=2)

    parser.add_argument("--max_src_len", type=int, default=400)
    parser.add_argument("--max_tgt_len", type=int, default=100)

    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--label_smoothing", type=float, default=0.1)

    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--device", type=str, default=None)

    return parser.parse_args()


def build_or_load_vocab(args: argparse.Namespace, tokenizer: SimpleTokenizer) -> Vocabulary:
    vocab_path = Path(args.vocab_path)
    if vocab_path.exists():
        print(f"Loading existing vocabulary from {vocab_path}")
        return Vocabulary.load(vocab_path)

    print("Building vocabulary from training data...")
    raw_texts = SummarizationDataset.texts(args.train_path, args.article_col, args.summary_col)
    tokenized = (tokenizer.tokenize(t) for t in raw_texts)
    vocab = Vocabulary.build(tokenized, min_freq=args.min_freq, max_size=args.max_vocab_size)
    vocab.save(vocab_path)
    print(f"Vocabulary of size {len(vocab)} saved to {vocab_path}")
    return vocab


def main() -> None:
    args = parse_args()
    tokenizer = SimpleTokenizer()
    vocab = build_or_load_vocab(args, tokenizer)

    train_dataset = SummarizationDataset(
        args.train_path,
        vocab,
        tokenizer,
        article_col=args.article_col,
        summary_col=args.summary_col,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
    )
    train_loader = get_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_loader = None
    if args.val_path:
        val_dataset = SummarizationDataset(
            args.val_path,
            vocab,
            tokenizer,
            article_col=args.article_col,
            summary_col=args.summary_col,
            max_src_len=args.max_src_len,
            max_tgt_len=args.max_tgt_len,
        )
        val_loader = get_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)

    config = TransformerConfig(
        src_vocab_size=len(vocab),
        tgt_vocab_size=len(vocab),
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        pad_id=vocab.pad_id,
    )
    model = Transformer(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        pad_id=vocab.pad_id,
        device=args.device,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        label_smoothing=args.label_smoothing,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
    )
    trainer.fit(args.epochs)


if __name__ == "__main__":
    main()
