"""End-to-end inference: raw text -> trained model -> generated summary.

Used by both the CLI (`python -m src.evaluation.inference`) and the
Streamlit UI (`ui/pages/1_Summarize.py`).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Union

import torch

from src.model.transformer import Transformer, TransformerConfig
from src.preprocessing.tokenizer import SimpleTokenizer
from src.preprocessing.vocabulary import Vocabulary


class Summarizer:
    """Loads a trained checkpoint + vocabulary and exposes `.summarize(text)`."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        vocab_path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = SimpleTokenizer()
        self.vocab = Vocabulary.load(vocab_path)

        # weights_only=False: checkpoints store a TransformerConfig object, not just tensors.
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        config = checkpoint["config"]
        if isinstance(config, dict):
            config = TransformerConfig(**config)

        self.model = Transformer(config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def summarize(
        self,
        text: str,
        max_src_len: int = 400,
        max_summary_len: int = 100,
        method: str = "greedy",
        beam_size: int = 4,
    ) -> str:
        """Generate an abstractive summary for a single piece of text."""
        src_tokens = self.tokenizer.tokenize(text)[: max_src_len - 2]
        src_ids = self.vocab.encode(src_tokens, add_special_tokens=True)
        src = torch.tensor([src_ids], dtype=torch.long, device=self.device)

        if method == "beam":
            out_ids = self.model.beam_search_decode(
                src, self.vocab.sos_id, self.vocab.eos_id, max_len=max_summary_len, beam_size=beam_size
            )
        else:
            out_ids = self.model.greedy_decode(src, self.vocab.sos_id, self.vocab.eos_id, max_len=max_summary_len)

        out_tokens = self.vocab.decode(out_ids[0].tolist(), strip_special=True)
        return self.tokenizer.detokenize(out_tokens)

    @torch.no_grad()
    def get_attention_maps(self, text: str, max_src_len: int = 400, max_summary_len: int = 100) -> dict:
        """Generate a summary while also returning attention weights, for
        visualization (see `src.visualization.attention_visualizer`).
        """
        src_tokens = self.tokenizer.tokenize(text)[: max_src_len - 2]
        src_ids = self.vocab.encode(src_tokens, add_special_tokens=True)
        src = torch.tensor([src_ids], dtype=torch.long, device=self.device)

        memory, src_mask, enc_attn = self.model.encode(src)
        out_ids = self.model.greedy_decode(src, self.vocab.sos_id, self.vocab.eos_id, max_len=max_summary_len)
        _, dec_self_attn, dec_cross_attn = self.model.decode(out_ids, memory, src_mask)

        return {
            "src_tokens": ["<sos>"] + src_tokens + ["<eos>"],
            "tgt_tokens": self.vocab.decode(out_ids[0].tolist(), strip_special=False),
            "encoder_self_attention": enc_attn,
            "decoder_self_attention": dec_self_attn,
            "decoder_cross_attention": dec_cross_attn,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize text with a trained Transformer checkpoint")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--vocab_path", type=str, default="data/vocab.json")
    parser.add_argument("--text", type=str, required=True, help="Raw article text to summarize")
    parser.add_argument("--method", type=str, default="greedy", choices=["greedy", "beam"])
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--max_summary_len", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summarizer = Summarizer(args.checkpoint, args.vocab_path)
    summary = summarizer.summarize(
        args.text, method=args.method, beam_size=args.beam_size, max_summary_len=args.max_summary_len
    )
    print(summary)


if __name__ == "__main__":
    main()
