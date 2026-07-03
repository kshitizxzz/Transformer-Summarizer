"""TF-IDF Extractive Summarization Baseline.

Implements a classical NLP extractive summarizer using TF-IDF sentence
scoring. Each sentence in the article gets a score = sum of TF-IDF weights
of its tokens. The top-k sentences (by score) are returned as the summary.

Usage
-----
    from src.models.tfidf_baseline import TFIDFSummarizer

    summarizer = TFIDFSummarizer(num_sentences=3)
    summarizer.fit(corpus_texts)          # build IDF from training corpus
    summary = summarizer.summarize(article_text)

    # Or use without fitting (uses within-document TF-IDF):
    summarizer = TFIDFSummarizer(num_sentences=3)
    summary = summarizer.summarize(article_text)
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter, defaultdict
from typing import List, Optional


# ---------------------------------------------------------------------------
# Simple preprocessing (stopwords + tokenization)
# ---------------------------------------------------------------------------

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "and", "but", "or", "nor", "so", "yet", "both", "either", "neither",
    "not", "no", "nor", "only", "own", "same", "than", "too", "very",
    "just", "that", "this", "these", "those", "it", "its", "he", "she",
    "they", "we", "i", "me", "him", "her", "them", "us", "my", "our",
    "their", "his", "her", "your", "what", "which", "who", "whom", "how",
    "when", "where", "why", "all", "each", "every", "most", "more", "other",
    "some", "such", "up", "out", "about", "also", "said", "says", "say",
}


def _tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """Lowercase, strip punctuation, optionally remove stopwords."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


def _split_sentences(text: str) -> List[str]:
    """Simple sentence splitter on . ! ? boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


# ---------------------------------------------------------------------------
# TF-IDF implementation (from scratch, no sklearn)
# ---------------------------------------------------------------------------

class TFIDFSummarizer:
    """Extractive summarizer using TF-IDF sentence scoring.

    Parameters
    ----------
    num_sentences : int
        Number of top-scoring sentences to include in the summary.
    use_corpus_idf : bool
        If True, IDF is computed over a training corpus (call .fit() first).
        If False, IDF is computed within the document only.
    remove_stopwords : bool
        Whether to remove stopwords before scoring.
    """

    def __init__(
        self,
        num_sentences: int = 3,
        use_corpus_idf: bool = False,
        remove_stopwords: bool = True,
    ) -> None:
        self.num_sentences = num_sentences
        self.use_corpus_idf = use_corpus_idf
        self.remove_stopwords = remove_stopwords

        # Corpus-level IDF (populated by .fit())
        self._corpus_idf: dict = {}
        self._corpus_size: int = 0

    # ------------------------------------------------------------------
    # Corpus fitting
    # ------------------------------------------------------------------

    def fit(self, texts: List[str]) -> "TFIDFSummarizer":
        """Build IDF table from a list of documents.

        IDF(term) = log((1 + N) / (1 + df(term))) + 1   [sklearn-style]
        """
        N = len(texts)
        df: Counter = Counter()
        for text in texts:
            tokens = set(_tokenize(text, self.remove_stopwords))
            df.update(tokens)

        self._corpus_idf = {
            term: math.log((1 + N) / (1 + count)) + 1
            for term, count in df.items()
        }
        self._corpus_size = N
        return self

    # ------------------------------------------------------------------
    # Within-document IDF (fallback when no corpus fitted)
    # ------------------------------------------------------------------

    @staticmethod
    def _doc_idf(sentences_tokens: List[List[str]]) -> dict:
        """Compute within-document IDF over sentence-as-documents."""
        N = len(sentences_tokens)
        df: Counter = Counter()
        for tokens in sentences_tokens:
            df.update(set(tokens))
        return {
            term: math.log((1 + N) / (1 + count)) + 1
            for term, count in df.items()
        }

    # ------------------------------------------------------------------
    # TF computation
    # ------------------------------------------------------------------

    @staticmethod
    def _tf(tokens: List[str]) -> dict:
        """Raw term frequency (count / total tokens)."""
        total = len(tokens)
        if total == 0:
            return {}
        counts = Counter(tokens)
        return {term: count / total for term, count in counts.items()}

    # ------------------------------------------------------------------
    # Scoring & summarization
    # ------------------------------------------------------------------

    def _score_sentence(self, tokens: List[str], idf: dict) -> float:
        """TF-IDF score for a sentence = sum of TF-IDF weights of its tokens."""
        tf = self._tf(tokens)
        return sum(tf.get(t, 0) * idf.get(t, 1.0) for t in tf)

    def summarize(self, text: str) -> str:
        """Return an extractive summary of `text`.

        Sentences are ranked by their TF-IDF score; the top
        `num_sentences` are returned in their original order.

        Returns
        -------
        str : the extractive summary
        """
        sentences = _split_sentences(text)
        if len(sentences) <= self.num_sentences:
            return text

        sentences_tokens = [
            _tokenize(s, self.remove_stopwords) for s in sentences
        ]

        # Choose IDF source
        if self.use_corpus_idf and self._corpus_idf:
            idf = self._corpus_idf
        else:
            idf = self._doc_idf(sentences_tokens)

        # Score each sentence
        scores = [
            self._score_sentence(tokens, idf)
            for tokens in sentences_tokens
        ]

        # Pick top-k by score, preserve original order
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_indices = sorted(ranked[: self.num_sentences])
        return " ".join(sentences[i] for i in top_indices)

    def summarize_with_scores(self, text: str) -> dict:
        """Return summary plus per-sentence scores (useful for visualization)."""
        sentences = _split_sentences(text)
        if not sentences:
            return {"summary": "", "sentences": [], "scores": []}

        sentences_tokens = [
            _tokenize(s, self.remove_stopwords) for s in sentences
        ]

        idf = self._corpus_idf if (self.use_corpus_idf and self._corpus_idf) \
              else self._doc_idf(sentences_tokens)

        scores = [self._score_sentence(t, idf) for t in sentences_tokens]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_indices = sorted(ranked[: self.num_sentences])
        summary = " ".join(sentences[i] for i in top_indices)

        return {
            "summary": summary,
            "sentences": sentences,
            "scores": scores,
            "selected_indices": top_indices,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "Machine learning is a branch of artificial intelligence. "
        "It focuses on building systems that learn from data. "
        "These systems improve their performance over time without being explicitly programmed. "
        "Deep learning is a subset of machine learning that uses neural networks. "
        "Neural networks are inspired by the human brain's structure. "
        "They consist of layers of interconnected nodes called neurons."
    )
    s = TFIDFSummarizer(num_sentences=2)
    result = s.summarize_with_scores(text)
    print("Summary:", result["summary"])
    print("\nSentence scores:")
    for sent, score in zip(result["sentences"], result["scores"]):
        mark = " <-- selected" if result["sentences"].index(sent) in result["selected_indices"] else ""
        print(f"  [{score:.4f}] {sent[:80]}{mark}")
