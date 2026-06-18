"""Vocabulary construction, token <-> id mapping, and (de)serialization."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence, Union

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]


class Vocabulary:
    """Maps tokens to integer ids and back.

    Reserves indices 0-3 for ``<pad>``, ``<unk>``, ``<sos>``, ``<eos>``.
    """

    def __init__(self) -> None:
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self._freeze = False
        for tok in SPECIAL_TOKENS:
            self._add(tok)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def _add(self, token: str) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]
        idx = len(self.token_to_id)
        self.token_to_id[token] = idx
        self.id_to_token[idx] = token
        return idx

    @classmethod
    def build(
        cls,
        tokenized_texts: Iterable[Sequence[str]],
        min_freq: int = 2,
        max_size: int | None = 30000,
    ) -> "Vocabulary":
        """Build a vocabulary from an iterable of pre-tokenized sequences.

        Parameters
        ----------
        tokenized_texts:
            Iterable where each element is a list of tokens (one per document).
        min_freq:
            Minimum token frequency required for inclusion.
        max_size:
            Maximum number of tokens to keep (special tokens excluded from
            the count, always kept). ``None`` means unlimited.
        """
        vocab = cls()
        counter: Counter[str] = Counter()
        for tokens in tokenized_texts:
            counter.update(tokens)

        most_common = counter.most_common()
        budget = None if max_size is None else max(0, max_size - len(SPECIAL_TOKENS))

        added = 0
        for token, freq in most_common:
            if freq < min_freq:
                continue
            if budget is not None and added >= budget:
                break
            vocab._add(token)
            added += 1
        return vocab

    # ------------------------------------------------------------------ #
    # Lookup
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    @property
    def sos_id(self) -> int:
        return self.token_to_id[SOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS_TOKEN]

    def encode(self, tokens: Sequence[str], add_special_tokens: bool = True) -> List[int]:
        """Convert tokens to ids, optionally wrapping with <sos>/<eos>."""
        ids = [self.token_to_id.get(tok, self.unk_id) for tok in tokens]
        if add_special_tokens:
            ids = [self.sos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: Sequence[int], strip_special: bool = True) -> List[str]:
        """Convert ids back to tokens, optionally dropping special tokens."""
        tokens = [self.id_to_token.get(int(i), UNK_TOKEN) for i in ids]
        if strip_special:
            tokens = [t for t in tokens if t not in SPECIAL_TOKENS]
        return tokens

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Vocabulary":
        with open(path, "r", encoding="utf-8") as f:
            token_to_id = json.load(f)
        vocab = cls.__new__(cls)
        vocab.token_to_id = {tok: int(idx) for tok, idx in token_to_id.items()}
        vocab.id_to_token = {idx: tok for tok, idx in vocab.token_to_id.items()}
        return vocab

    def __repr__(self) -> str:
        return f"Vocabulary(size={len(self)})"
