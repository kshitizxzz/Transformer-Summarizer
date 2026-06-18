"""Dependency-free ROUGE-1, ROUGE-2, and ROUGE-L implementation.

These are standard summarization metrics. Implemented from scratch (no
external `rouge`/`rouge-score` package required) so the project has no
hidden dependency for its headline evaluation numbers.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _prf1(overlap: int, ref_count: int, hyp_count: int) -> Dict[str, float]:
    precision = overlap / hyp_count if hyp_count else 0.0
    recall = overlap / ref_count if ref_count else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def rouge_n(reference: Sequence[str], hypothesis: Sequence[str], n: int = 1) -> Dict[str, float]:
    """ROUGE-N: n-gram overlap between reference and hypothesis token lists."""
    ref_ngrams = _ngrams(reference, n)
    hyp_ngrams = _ngrams(hypothesis, n)

    overlap = sum((ref_ngrams & hyp_ngrams).values())
    ref_count = sum(ref_ngrams.values())
    hyp_count = sum(hyp_ngrams.values())

    return _prf1(overlap, ref_count, hyp_count)


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Longest common subsequence length via standard O(len(a)*len(b)) DP."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]


def rouge_l(reference: Sequence[str], hypothesis: Sequence[str]) -> Dict[str, float]:
    """ROUGE-L: longest-common-subsequence based F-measure."""
    lcs = _lcs_length(reference, hypothesis)
    return _prf1(lcs, len(reference), len(hypothesis))


def compute_rouge(reference_tokens: Sequence[str], hypothesis_tokens: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L for a single (reference, hypothesis) pair."""
    return {
        "rouge-1": rouge_n(reference_tokens, hypothesis_tokens, 1),
        "rouge-2": rouge_n(reference_tokens, hypothesis_tokens, 2),
        "rouge-l": rouge_l(reference_tokens, hypothesis_tokens),
    }


def compute_rouge_corpus(
    references: List[Sequence[str]], hypotheses: List[Sequence[str]]
) -> Dict[str, Dict[str, float]]:
    """Average ROUGE-1/2/L F1/precision/recall across a corpus of summaries."""
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")

    totals = {
        "rouge-1": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        "rouge-2": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        "rouge-l": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
    }

    n = len(references)
    if n == 0:
        return totals

    for ref, hyp in zip(references, hypotheses):
        scores = compute_rouge(ref, hyp)
        for metric, values in scores.items():
            for key, val in values.items():
                totals[metric][key] += val

    for metric in totals:
        for key in totals[metric]:
            totals[metric][key] /= n

    return totals
