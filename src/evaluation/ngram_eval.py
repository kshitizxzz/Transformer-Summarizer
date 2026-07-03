"""N-gram overlap metrics for evaluating summarization quality.

Computes unigram, bigram, trigram precision/recall/F1 and LCS F1
between a generated summary and a reference summary.

Usage:
    from src.evaluation.ngram_eval import compute_ngram_scores, compute_ngram_corpus

    scores = compute_ngram_scores(ref_tokens, hyp_tokens)
    # scores["unigram"]["f1"], scores["bigram"]["f1"], ...
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1))


def _prf1(overlap: int, ref_count: int, hyp_count: int) -> Dict[str, float]:
    precision = overlap / hyp_count if hyp_count else 0.0
    recall    = overlap / ref_count  if ref_count  else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def ngram_overlap(reference: Sequence[str], hypothesis: Sequence[str], n: int) -> Dict[str, float]:
    ref_ng  = _ngrams(reference,  n)
    hyp_ng  = _ngrams(hypothesis, n)
    overlap = sum((ref_ng & hyp_ng).values())
    return _prf1(overlap, sum(ref_ng.values()), sum(hyp_ng.values()))


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]


def lcs_overlap(reference: Sequence[str], hypothesis: Sequence[str]) -> Dict[str, float]:
    lcs = _lcs_length(reference, hypothesis)
    return _prf1(lcs, len(reference), len(hypothesis))


def compute_ngram_scores(
    reference: Sequence[str], hypothesis: Sequence[str]
) -> Dict[str, Dict[str, float]]:
    """Compute unigram, bigram, trigram, and LCS overlap for one pair."""
    return {
        "unigram": ngram_overlap(reference, hypothesis, 1),
        "bigram":  ngram_overlap(reference, hypothesis, 2),
        "trigram": ngram_overlap(reference, hypothesis, 3),
        "lcs":     lcs_overlap(reference, hypothesis),
    }


def compute_ngram_corpus(
    references: List[Sequence[str]],
    hypotheses: List[Sequence[str]],
) -> Dict[str, Dict[str, float]]:
    """Average n-gram scores across a corpus."""
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")

    totals: Dict[str, Dict[str, float]] = {
        k: {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        for k in ("unigram", "bigram", "trigram", "lcs")
    }
    n = len(references)
    if n == 0:
        return totals

    for ref, hyp in zip(references, hypotheses):
        for metric, scores in compute_ngram_scores(ref, hyp).items():
            for stat, val in scores.items():
                totals[metric][stat] += val

    for metric in totals:
        for stat in totals[metric]:
            totals[metric][stat] /= n
    return totals


def print_ngram_report(scores: Dict[str, Dict[str, float]]) -> None:
    print("\n--- N-gram Overlap ---")
    for name in ("unigram", "bigram", "trigram", "lcs"):
        s = scores[name]
        label = "LCS     " if name == "lcs" else f"{name.capitalize():8s}"
        print(f"  {label}  P={s['precision']:.3f}  R={s['recall']:.3f}  F1={s['f1']:.3f}")
    print()
