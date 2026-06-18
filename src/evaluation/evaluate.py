"""Evaluation CLI: ROUGE + perplexity over a held-out split, plus a handful
of qualitative examples, written to `logs/eval_results.json`. This is the
file the Streamlit "Results" page (`ui/pages/5_Results.py`) reads.

Perplexity runs over the *entire* file (cheap: batched forward passes only).
Generation (greedy/beam) + ROUGE run over a capped subset (`--max_eval_examples`),
since autoregressive decoding per example is the slow part on CPU.

Example
-------
    python -m src.evaluation.evaluate \\
        --data_path data/test.csv \\
        --checkpoint checkpoints/best.pt \\
        --vocab_path data/vocab.json \\
        --max_eval_examples 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.inference import Summarizer
from src.evaluation.perplexity import compute_perplexity
from src.evaluation.rouge import compute_rouge_corpus
from src.preprocessing.dataset import SummarizationDataset, get_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint: ROUGE + perplexity")
    parser.add_argument("--data_path", type=str, required=True, help="Held-out CSV/JSON(L) with article/summary columns")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--vocab_path", type=str, default="data/vocab.json")
    parser.add_argument("--article_col", type=str, default="article")
    parser.add_argument("--summary_col", type=str, default="highlights")

    parser.add_argument("--max_src_len", type=int, default=400)
    parser.add_argument("--max_tgt_len", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for the perplexity pass")

    parser.add_argument("--method", type=str, default="greedy", choices=["greedy", "beam"])
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--max_summary_len", type=int, default=100)

    parser.add_argument(
        "--max_eval_examples",
        type=int,
        default=200,
        help="Cap on how many examples get a generated summary + ROUGE score "
        "(generation is the slow part; perplexity still runs over the full file). "
        "Use 0 for no cap.",
    )
    parser.add_argument(
        "--num_qualitative_examples",
        type=int,
        default=5,
        help="How many (article, reference, generated) triples to save for the Results page",
    )

    parser.add_argument("--output_path", type=str, default="logs/eval_results.json")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summarizer = Summarizer(args.checkpoint, args.vocab_path, device=args.device)
    vocab, tokenizer = summarizer.vocab, summarizer.tokenizer

    # --- perplexity over the full split (cheap: forward passes only) ---
    dataset = SummarizationDataset(
        args.data_path,
        vocab,
        tokenizer,
        article_col=args.article_col,
        summary_col=args.summary_col,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
    )
    data_loader = get_dataloader(dataset, batch_size=args.batch_size, shuffle=False)
    perplexity = compute_perplexity(summarizer.model, data_loader, pad_id=vocab.pad_id, device=summarizer.device)
    print(f"Perplexity over {len(dataset)} examples: {perplexity:.3f}")

    # --- generation + ROUGE over a (possibly capped) subset ---
    records = dataset.records
    if args.max_eval_examples:
        records = records[: args.max_eval_examples]

    references, hypotheses, qualitative = [], [], []
    for i, (article, reference) in enumerate(records):
        generated = summarizer.summarize(
            article,
            max_src_len=args.max_src_len,
            max_summary_len=args.max_summary_len,
            method=args.method,
            beam_size=args.beam_size,
        )
        references.append(tokenizer.tokenize(reference))
        hypotheses.append(tokenizer.tokenize(generated))

        if i < args.num_qualitative_examples:
            qualitative.append({"article": article, "reference": reference, "generated": generated})

        if (i + 1) % 20 == 0:
            print(f"  generated {i + 1}/{len(records)} summaries...")

    rouge = compute_rouge_corpus(references, hypotheses)
    print(
        f"ROUGE-1 F1: {rouge['rouge-1']['f1']:.3f}  "
        f"ROUGE-2 F1: {rouge['rouge-2']['f1']:.3f}  "
        f"ROUGE-L F1: {rouge['rouge-l']['f1']:.3f}"
    )

    results = {"rouge": rouge, "perplexity": perplexity, "examples": qualitative}

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
