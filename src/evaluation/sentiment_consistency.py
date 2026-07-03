"""Sentiment Consistency Analysis.

Trains a Logistic Regression classifier to predict sentiment (positive /
negative) of text, then checks whether the Transformer-generated summary
preserves the sentiment of its source article.

Concepts used
-------------
- Sentiment Analysis              (nlp.pdf)
- Logistic Regression             (ml.pdf)
- TF-IDF feature engineering      (nlp.pdf / ml.pdf)
- Precision, Recall, F1, Confusion Matrix (ml.pdf)
- Train/test split                (ml.pdf)

The classifier is trained on (text -> sentiment) pairs using TF-IDF features.
It is then applied to both source articles and generated summaries. Sentiment
consistency = fraction of (article, summary) pairs where the predicted
sentiment matches.

Usage
-----
    from src.evaluation.sentiment_consistency import SentimentConsistencyAnalyzer

    analyzer = SentimentConsistencyAnalyzer()
    analyzer.fit(train_texts, train_labels)        # 0=negative, 1=positive
    results = analyzer.evaluate_consistency(articles, summaries)
    print(results["consistency_rate"])

    # If no labeled data, use lexicon-based pseudo-labeling:
    analyzer.fit_with_lexicon(train_texts)
    results = analyzer.evaluate_consistency(articles, summaries)
"""

from __future__ import annotations

import json
import math
import re
import string
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Simple TF-IDF vectorizer (from scratch, no sklearn)
# ---------------------------------------------------------------------------

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
    """Rule-based sentiment: +1 if more positive words, 0 otherwise."""
    tokens = set(_tokenize(text))
    pos = len(tokens & POSITIVE_WORDS)
    neg = len(tokens & NEGATIVE_WORDS)
    return 1 if pos >= neg else 0


# ---------------------------------------------------------------------------
# Minimal TF-IDF vectorizer
# ---------------------------------------------------------------------------

class SimpleTFIDF:
    """Sparse TF-IDF vectorizer (dict representation, no numpy required)."""

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

        # Select top max_features by corpus frequency
        top_terms = [t for t, _ in all_tokens.most_common(self.max_features)]
        self.vocab = {t: i for i, t in enumerate(top_terms)}
        self.idf = {
            t: math.log((1 + N) / (1 + df.get(t, 0))) + 1
            for t in self.vocab
        }
        return self

    def transform(self, text: str) -> Dict[int, float]:
        """Return {feature_index: tfidf_value} sparse vector."""
        tokens = _tokenize(text)
        total = len(tokens)
        if total == 0:
            return {}
        tf = Counter(tokens)
        vec: Dict[int, float] = {}
        for term, count in tf.items():
            if term in self.vocab:
                idx = self.vocab[term]
                vec[idx] = (count / total) * self.idf.get(term, 1.0)
        return vec

    def dim(self) -> int:
        return len(self.vocab)


# ---------------------------------------------------------------------------
# Logistic Regression (from scratch, sigmoid + gradient descent)
# ---------------------------------------------------------------------------

class LogisticRegression:
    """Binary logistic regression trained with mini-batch gradient descent.

    P(y=1|x) = sigmoid(w^T x + b)
    Loss: binary cross-entropy

    Parameters
    ----------
    lr : learning rate
    epochs : training epochs
    reg : L2 regularization coefficient
    """

    def __init__(self, lr: float = 0.1, epochs: int = 50, reg: float = 0.01) -> None:
        self.lr = lr
        self.epochs = epochs
        self.reg = reg
        self.weights: Dict[int, float] = {}
        self.bias: float = 0.0
        self.dim: int = 0

    @staticmethod
    def _sigmoid(z: float) -> float:
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        ez = math.exp(z)
        return ez / (1.0 + ez)

    def _predict_proba(self, x: Dict[int, float]) -> float:
        z = self.bias + sum(self.weights.get(i, 0.0) * v for i, v in x.items())
        return self._sigmoid(z)

    def fit(
        self,
        X: List[Dict[int, float]],
        y: List[int],
    ) -> "LogisticRegression":
        n = len(X)
        for epoch in range(self.epochs):
            grad_w: Dict[int, float] = {}
            grad_b = 0.0
            total_loss = 0.0

            for xi, yi in zip(X, y):
                prob = self._predict_proba(xi)
                err = prob - yi
                eps = 1e-9
                total_loss -= yi * math.log(prob + eps) + (1 - yi) * math.log(1 - prob + eps)

                grad_b += err
                for idx, val in xi.items():
                    grad_w[idx] = grad_w.get(idx, 0.0) + err * val

            # Update parameters
            self.bias -= self.lr * grad_b / n
            for idx, gw in grad_w.items():
                w = self.weights.get(idx, 0.0)
                # L2 regularization gradient: + reg * w
                self.weights[idx] = w - self.lr * (gw / n + self.reg * w)

        return self

    def predict(self, X: List[Dict[int, float]]) -> List[int]:
        return [1 if self._predict_proba(x) >= 0.5 else 0 for x in X]

    def predict_one(self, x: Dict[int, float]) -> Tuple[int, float]:
        prob = self._predict_proba(x)
        return (1 if prob >= 0.5 else 0), prob


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def precision_recall_f1(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """Binary precision, recall, F1 for positive class (label=1)."""
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
    tn = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy  = (tp + tn) / len(y_true) if y_true else 0.0
    return {
        "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


# ---------------------------------------------------------------------------
# Main Analyzer
# ---------------------------------------------------------------------------

class SentimentConsistencyAnalyzer:
    """Trains a Logistic Regression sentiment classifier and measures
    how consistently generated summaries preserve source article sentiment.
    """

    def __init__(self, max_features: int = 3000) -> None:
        self.vectorizer = SimpleTFIDF(max_features=max_features)
        self.classifier = LogisticRegression(lr=0.1, epochs=100, reg=0.01)
        self._fitted = False

    def fit_with_lexicon(self, texts: List[str]) -> "SentimentConsistencyAnalyzer":
        """Pseudo-label texts using lexicon, then train LR classifier.

        This approach allows training without manually labeled sentiment data.
        Labels are assigned by the positive/negative word lexicon.
        """
        labels = [lexicon_sentiment(t) for t in texts]
        return self.fit(texts, labels)

    def fit(self, texts: List[str], labels: List[int]) -> "SentimentConsistencyAnalyzer":
        """Train the TF-IDF + Logistic Regression classifier."""
        self.vectorizer.fit(texts)
        X = [self.vectorizer.transform(t) for t in texts]
        self.classifier.fit(X, labels)
        self._fitted = True
        return self

    def predict_sentiment(self, text: str) -> Tuple[int, float]:
        """Predict sentiment label and confidence for a single text."""
        if not self._fitted:
            raise RuntimeError("Call .fit() or .fit_with_lexicon() first.")
        x = self.vectorizer.transform(text)
        return self.classifier.predict_one(x)

    def evaluate_consistency(
        self,
        articles: List[str],
        summaries: List[str],
    ) -> Dict:
        """Measure sentiment consistency between articles and their summaries.

        For each (article, summary) pair:
          - Predict sentiment of article
          - Predict sentiment of summary
          - Consistent = both have the same predicted label

        Returns
        -------
        dict with: consistency_rate, article_sentiments, summary_sentiments,
                   per_pair_results, classifier_metrics_on_articles
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() or .fit_with_lexicon() first.")

        article_preds, summary_preds = [], []
        pair_results = []

        for art, summ in zip(articles, summaries):
            a_label, a_conf = self.predict_sentiment(art)
            s_label, s_conf = self.predict_sentiment(summ)
            consistent = (a_label == s_label)
            article_preds.append(a_label)
            summary_preds.append(s_label)
            pair_results.append({
                "article_sentiment": "positive" if a_label else "negative",
                "article_confidence": round(a_conf, 3),
                "summary_sentiment": "positive" if s_label else "negative",
                "summary_confidence": round(s_conf, 3),
                "consistent": consistent,
            })

        consistency_rate = sum(r["consistent"] for r in pair_results) / len(pair_results)
        pos_articles = sum(article_preds)
        neg_articles = len(article_preds) - pos_articles

        return {
            "consistency_rate": round(consistency_rate, 4),
            "total_pairs": len(pair_results),
            "consistent_pairs": sum(r["consistent"] for r in pair_results),
            "article_sentiment_dist": {
                "positive": pos_articles,
                "negative": neg_articles,
            },
            "per_pair": pair_results,
        }

    def save_results(self, results: Dict, output_path: str) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Sentiment results saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick demo with synthetic examples
    train_texts = [
        "The company achieved record profits this quarter.",
        "Violence erupted in the city leaving many dead.",
        "Scientists celebrated a major breakthrough in cancer research.",
        "The economy collapsed causing widespread poverty.",
        "The team won the championship after years of hard work.",
        "The disaster killed hundreds and destroyed homes.",
        "Investors praised the company's strong performance.",
        "Crime rates surged to the highest levels in decades.",
    ]
    analyzer = SentimentConsistencyAnalyzer()
    analyzer.fit_with_lexicon(train_texts)

    articles = [
        "The economy is recovering strongly with positive indicators across all sectors.",
        "Violence broke out in the capital killing dozens of people.",
    ]
    summaries = [
        "Economy shows positive recovery.",
        "Attack kills many in capital city.",
    ]

    results = analyzer.evaluate_consistency(articles, summaries)
    print(f"Consistency rate: {results['consistency_rate']:.1%}")
    for i, pair in enumerate(results["per_pair"]):
        print(f"  Pair {i+1}: article={pair['article_sentiment']}, "
              f"summary={pair['summary_sentiment']}, "
              f"consistent={pair['consistent']}")
