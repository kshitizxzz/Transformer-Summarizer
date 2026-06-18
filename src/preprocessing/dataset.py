"""Dataset and DataLoader utilities for article/summary pairs.

Supports loading from CSV or JSON-lines files containing two text
columns (configurable, defaults to the CNN/DailyMail convention of
``article`` and ``highlights``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.preprocessing.tokenizer import SimpleTokenizer
from src.preprocessing.vocabulary import Vocabulary


def _read_records(path: Union[str, Path], article_col: str, summary_col: str) -> List[Tuple[str, str]]:
    """Read (article, summary) pairs from a .csv or .jsonl/.json file."""
    path = Path(path)
    records: List[Tuple[str, str]] = []

    if path.suffix.lower() == ".csv":
        import csv

        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append((row[article_col], row[summary_col]))
    elif path.suffix.lower() in {".jsonl", ".json"}:
        with open(path, encoding="utf-8") as f:
            if path.suffix.lower() == ".jsonl":
                rows = (json.loads(line) for line in f if line.strip())
            else:
                rows = iter(json.load(f))
            for row in rows:
                records.append((row[article_col], row[summary_col]))
    else:
        raise ValueError(f"Unsupported dataset file type: {path.suffix}")

    return records


class SummarizationDataset(Dataset):
    """Tokenizes and encodes (article, summary) pairs for the Transformer.

    Each item returns a dict of LongTensors: ``src`` (encoder input,
    article tokens wrapped in <sos>/<eos>) and ``tgt`` (decoder
    target, summary tokens wrapped in <sos>/<eos>).
    """

    def __init__(
        self,
        path: Union[str, Path],
        vocab: Vocabulary,
        tokenizer: Optional[SimpleTokenizer] = None,
        article_col: str = "article",
        summary_col: str = "highlights",
        max_src_len: int = 400,
        max_tgt_len: int = 100,
    ) -> None:
        self.vocab = vocab
        self.tokenizer = tokenizer or SimpleTokenizer()
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.records = _read_records(path, article_col, summary_col)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        article, summary = self.records[idx]

        src_tokens = self.tokenizer.tokenize(article)[: self.max_src_len - 2]
        tgt_tokens = self.tokenizer.tokenize(summary)[: self.max_tgt_len - 2]

        src_ids = self.vocab.encode(src_tokens, add_special_tokens=True)
        tgt_ids = self.vocab.encode(tgt_tokens, add_special_tokens=True)

        return {
            "src": torch.tensor(src_ids, dtype=torch.long),
            "tgt": torch.tensor(tgt_ids, dtype=torch.long),
        }

    @staticmethod
    def texts(path: Union[str, Path], article_col: str = "article", summary_col: str = "highlights") -> List[str]:
        """Convenience helper: flatten all articles+summaries into raw strings.

        Useful for building a `Vocabulary` before constructing the dataset.
        """
        records = _read_records(path, article_col, summary_col)
        out: List[str] = []
        for article, summary in records:
            out.append(article)
            out.append(summary)
        return out


def make_collate_fn(pad_id: int):
    """Returns a collate function that pads `src`/`tgt` batches to equal length."""

    def collate_fn(batch: Sequence[dict]) -> dict:
        src = [item["src"] for item in batch]
        tgt = [item["tgt"] for item in batch]

        src_padded = pad_sequence(src, batch_first=True, padding_value=pad_id)
        tgt_padded = pad_sequence(tgt, batch_first=True, padding_value=pad_id)

        return {
            "src": src_padded,
            "tgt": tgt_padded,
            "src_lengths": torch.tensor([len(s) for s in src], dtype=torch.long),
            "tgt_lengths": torch.tensor([len(t) for t in tgt], dtype=torch.long),
        }

    return collate_fn


def get_dataloader(
    dataset: SummarizationDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader with dynamic padding for a `SummarizationDataset`."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=make_collate_fn(dataset.vocab.pad_id),
    )
