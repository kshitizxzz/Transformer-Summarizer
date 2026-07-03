"""Sentiment consistency analysis for summarization evaluation.

Trains a Logistic Regression classifier on TF-IDF features to predict
positive/negative sentiment, then checks whether generated summaries
preserve the sentiment of their source articles.

Usage:
    analyzer = SentimentConsistencyAnalyzer()
    analyzer.fit_with_lexicon(train_texts)
    results = analyzer.evaluate_consistency(articles, summaries)
    print(results["consistency_rate"])
"""

from __future__ import annotations

import json
import math
import string
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "and", "but", "or", "not", "no", "it", "its", "he", "she",
    "they", "we", "i", "that", "this", "which", "who", "what", "also",
}

POSITIVE_WORDS = {
    "good", "great", "excellent", "best", "wonderful", "amazing", "fantastic",
    "success", "successful", "win", "won", "victory", "improve", "improved",
    "benefit", "positive", "happy", "celebrated", "praised", "loved",
    "supported", "helped", "saved", "achieved", "agreement", "peace",
    "recovery", "survived", "hope", "progress", "gain", "increased",
    "strong", "better", "outstanding", "remarkable", "impressive",
}

NEGATIVE_WORDS = {
    "bad", "terrible", "worst", "awful", "horrible", "disaster", "failure",
    "fail", "failed", "lost", "loss", "dead", "death", "killed", "murder",
    "crime", "attack", "violence", "war", "crash", "collapse", "arrested",
    "charged", "accused", "crisis", "danger", "threat", "injured", "victim",
    "protest", "riot", "shooting", "bombing", "fire", "damage", "accident",
    "declined", "fell", "dropped", "cut", "layoff", "bankrupt", "scandal",
}


def _tokenize(text: str) -> List[str]:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return [t for t in text.split() if t not in STOPWORDS and len(t) > 1]


def lexicon_sentiment(text: str) -> int:
    tokens = set(_tokenize(text))
    pos = len(tokens & POSITIVE_WORDS)
    neg = len(tokens & NEGATIVE_WORDS)
    return 1 if pos >= neg else 0


class SimpleTFIDF:
    def __init__(self, max_features: int = 5000) -> None:
        self.max_features = max_features
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

    def fit(self, texts: List[str]) -> "SimpleTFIDF":
        N = len(texts)
        df: Counter = Counter()
        all_tokens: Counter = Counter()
        for text in texts:
            tokens = _tokenize(text)
            all_tokens.update(tokens)
            df.update(set(tokens))
        top_terms = [t for t, _ in all_tokens.most_common(self.max_features)]
        self.vocab = {t: i for i, t in enumerate(top_terms)}
        self.idf = {
            t: math.log((1 + N) / (1 + df.get(t, 0))) + 1
            for t in self.vocab
        }
        return self

    def transform(self, text: str) -> Dict[int, float]:
        tokens = _tokenize(text)
        total = len(tokens)
        if total == 0:
            return {}
        tf = Counter(tokens)
        return {
            self.vocab[term]: (count / total) * self.idf.get(term, 1.0)
            for term, count in tf.items()
            if term in self.vocab
        }


class LogisticRegression:
    """Binary logistic regression with mini-batch gradient descent."""

    def __init__(self, lr: float = 0.1, epochs: int = 50, reg: float = 0.01) -> None:
        self.lr = lr
        self.epochs = epochs
        self.reg = reg
        self.weights: Dict[int, float] = {}
        self.bias: float = 0.0

    @staticmethod
    def _sigmoid(z: float) -> float:
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        ez = math.exp(z)
        return ez / (1.0 + ez)

    def _predict_proba(self, x: Dict[int, float]) -> float:
        z = self.bias + sum(self.weights.get(i, 0.0) * v for i, v in x.items())
        return self._sigmoid(z)

    def fit(self, X: List[Dict[int, float]], y: List[int]) -> "LogisticRegression":
        n = len(X)
        for _ in range(self.epochs):
            grad_w: Dict[int, float] = {}
            grad_b = 0.0
            for xi, yi in zip(X, y):
                err = self._predict_proba(xi) - yi
                grad_b += err
                for idx, val in xi.items():
                    grad_w[idx] = grad_w.get(idx, 0.0) + err * val
            self.bias -= self.lr * grad_b / n
            for idx, gw in grad_w.items():
                w = self.weights.get(idx, 0.0)
                self.weights[idx] = w - self.lr * (gw / n + self.reg * w)
        return self

    def predict_one(self, x: Dict[int, float]) -> Tuple[int, float]:
        prob = self._predict_proba(x)
        return (1 if prob >= 0.5 else 0), prob


def precision_recall_f1(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
    tn = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 0)
    p  = tp / (tp + fp) if (tp + fp) else 0.0
    r  = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {
        "precision": p, "recall": r, "f1": f1,
        "accuracy": (tp + tn) / len(y_true) if y_true else 0.0,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


class SentimentConsistencyAnalyzer:
    """TF-IDF + Logistic Regression sentiment classifier for consistency analysis."""

    def __init__(self, max_features: int = 3000) -> None:
        self.vectorizer = SimpleTFIDF(max_features=max_features)
        self.classifier = LogisticRegression(lr=0.1, epochs=100, reg=0.01)
        self._fitted = False

    def fit_with_lexicon(self, texts: List[str]) -> "SentimentConsistencyAnalyzer":
        """Pseudo-label using positive/negative word lexicon, then train."""
        labels = [lexicon_sentiment(t) for t in texts]
        return self.fit(texts, labels)

    def fit(self, texts: List[str], labels: List[int]) -> "SentimentConsistencyAnalyzer":
        self.vectorizer.fit(texts)
        X = [self.vectorizer.transform(t) for t in texts]
        self.classifier.fit(X, labels)
        self._fitted = True
        return self

    def predict_sentiment(self, text: str) -> Tuple[int, float]:
        if not self._fitted:
            raise RuntimeError("Call .fit() or .fit_with_lexicon() first.")
        return self.classifier.predict_one(self.vectorizer.transform(text))

    def evaluate_consistency(
        self, articles: List[str], summaries: List[str]
    ) -> Dict:
        if not self._fitted:
            raise RuntimeError("Call .fit() or .fit_with_lexicon() first.")
        pair_results = []
        for art, summ in zip(articles, summaries):
            a_label, a_conf = self.predict_sentiment(art)
            s_label, s_conf = self.predict_sentiment(summ)
            pair_results.append({
                "article_sentiment":    "positive" if a_label else "negative",
                "article_confidence":   round(a_conf, 3),
                "summary_sentiment":    "positive" if s_label else "negative",
                "summary_confidence":   round(s_conf, 3),
                "consistent":           (a_label == s_label),
            })
        n = len(pair_results)
        pos = sum(1 for r in pair_results if r["article_sentiment"] == "positive")
        return {
            "consistency_rate": round(sum(r["consistent"] for r in pair_results) / n, 4),
            "total_pairs": n,
            "consistent_pairs": sum(r["consistent"] for r in pair_results),
            "article_sentiment_dist": {"positive": pos, "negative": n - pos},
            "per_pair": pair_results,
        }

    def save_results(self, results: Dict, output_path: str) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
