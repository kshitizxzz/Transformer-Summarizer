"""Lightweight tokenizer for the Transformer summarizer.

This module implements a dependency-free word-level tokenizer. It is
intentionally simple (regex based) so the rest of the pipeline has no
hard requirement on third-party tokenizer libraries, while still being
easy to swap out for a subword tokenizer (e.g. HuggingFace `tokenizers`
or `sentencepiece`) later without changing any other module: every other
component only relies on `tokenize()` / `detokenize()`.
"""

from __future__ import annotations

import re
from typing import List

# Matches words (including contractions like "it's"), standalone punctuation,
# and numbers as separate tokens.
_TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[^\sA-Za-z0-9]")


class SimpleTokenizer:
    """A small, fast, regex-based word/punctuation tokenizer.

    Parameters
    ----------
    lowercase:
        Whether to lowercase text before tokenizing.
    """

    def __init__(self, lowercase: bool = True) -> None:
        self.lowercase = lowercase

    def tokenize(self, text: str) -> List[str]:
        """Split raw text into a list of tokens."""
        if text is None:
            return []
        text = text.strip()
        if self.lowercase:
            text = text.lower()
        return _TOKEN_PATTERN.findall(text)

    def detokenize(self, tokens: List[str]) -> str:
        """Join tokens back into a readable string.

        Adds a leading space before each token unless it is punctuation
        that conventionally hugs the previous word (e.g. ``.``, ``,``).
        """
        no_space_before = {".", ",", "!", "?", ";", ":", "'", "n't", "%", ")", "]", "}"}
        no_space_after = {"(", "[", "{"}

        pieces: List[str] = []
        prev = None
        for tok in tokens:
            if pieces and tok not in no_space_before and prev not in no_space_after:
                pieces.append(" ")
            pieces.append(tok)
            prev = tok
        return "".join(pieces).strip()

    def __call__(self, text: str) -> List[str]:
        return self.tokenize(text)
