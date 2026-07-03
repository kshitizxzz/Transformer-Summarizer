"""News topic classifier: Random Forest on TF-IDF features.

Classifies articles into topic categories (politics, sports, business,
health, crime) and stratifies summarization evaluation by topic.

Usage:
    clf = TopicClassifier(n_trees=30)
    clf.fit(train_texts, train_labels)
    results = clf.cross_validate(texts, labels, k=5)
    print(results["macro_f1"])
"""

from __future__ import annotations

import json
import math
import random
import string
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "politics": [
        "government", "president", "congress", "senate", "election", "vote",
        "democrat", "republican", "minister", "parliament", "law", "bill",
        "policy", "political", "officials", "administration", "campaign",
    ],
    "sports": [
        "game", "team", "player", "coach", "season", "championship", "won",
        "score", "match", "tournament", "league", "football", "basketball",
        "soccer", "tennis", "baseball", "olympic", "athlete",
    ],
    "business": [
        "company", "market", "stock", "economy", "trade", "profit", "revenue",
        "investment", "bank", "financial", "billion", "million", "ceo",
        "corporation", "industry", "product", "sales", "shareholders",
    ],
    "health": [
        "hospital", "doctor", "patient", "disease", "health", "medical",
        "treatment", "drug", "cancer", "virus", "vaccine", "surgery",
        "study", "research", "clinical", "symptoms", "therapy",
    ],
    "crime": [
        "police", "arrested", "murder", "shooting", "court", "prison",
        "sentence", "crime", "criminal", "suspect", "investigation",
        "officer", "weapon", "stabbing", "robbery", "convicted", "trial",
    ],
}

TOPIC_TO_ID = {t: i for i, t in enumerate(["politics", "sports", "business", "health", "crime", "other"])}
ID_TO_TOPIC = {i: t for t, i in TOPIC_TO_ID.items()}

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "have",
    "has", "had", "do", "does", "did", "to", "of", "in", "for", "on",
    "with", "at", "by", "from", "as", "and", "but", "or", "not", "it",
    "its", "he", "she", "they", "we", "i", "that", "this", "said", "new",
}


def assign_topic_label(text: str) -> str:
    text_lower = text.lower()
    scores = {
        topic: sum(1 for kw in kws if kw in text_lower)
        for topic, kws in TOPIC_KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "other"


def _tokenize(text: str) -> List[str]:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return [t for t in text.split() if t not in STOPWORDS and len(t) > 2]


class TFIDFVectorizer:
    def __init__(self, max_features: int = 3000) -> None:
        self.max_features = max_features
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

    def fit(self, texts: List[str]) -> "TFIDFVectorizer":
        N = len(texts)
        df: Counter = Counter()
        freq: Counter = Counter()
        for text in texts:
            tokens = _tokenize(text)
            freq.update(tokens)
            df.update(set(tokens))
        top = [t for t, _ in freq.most_common(self.max_features)]
        self.vocab = {t: i for i, t in enumerate(top)}
        self.idf = {
            t: math.log((1 + N) / (1 + df.get(t, 0))) + 1
            for t in self.vocab
        }
        return self

    def transform(self, text: str) -> List[float]:
        tokens = _tokenize(text)
        total = len(tokens) or 1
        tf = Counter(tokens)
        vec = [0.0] * len(self.vocab)
        for term, cnt in tf.items():
            if term in self.vocab:
                vec[self.vocab[term]] = (cnt / total) * self.idf.get(term, 1.0)
        return vec

    def fit_transform(self, texts: List[str]) -> List[List[float]]:
        self.fit(texts)
        return [self.transform(t) for t in texts]


class DecisionTree:
    """CART decision tree using information gain (entropy) criterion."""

    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 5,
        max_features: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.rng = random.Random(random_state)
        self.tree: Optional[dict] = None

    @staticmethod
    def _entropy(labels: List[int]) -> float:
        n = len(labels)
        if n == 0:
            return 0.0
        counts = Counter(labels)
        return -sum((c / n) * math.log2(c / n + 1e-9) for c in counts.values())

    def _information_gain(
        self, X: List[List[float]], y: List[int], feature: int, threshold: float
    ) -> float:
        left_y  = [yi for xi, yi in zip(X, y) if xi[feature] <= threshold]
        right_y = [yi for xi, yi in zip(X, y) if xi[feature] > threshold]
        n = len(y)
        if not left_y or not right_y:
            return 0.0
        H_parent   = self._entropy(y)
        H_children = (len(left_y) / n) * self._entropy(left_y) + \
                     (len(right_y) / n) * self._entropy(right_y)
        return H_parent - H_children

    def _best_split(self, X: List[List[float]], y: List[int]) -> Tuple[int, float, float]:
        n_features = len(X[0])
        features = list(range(n_features))
        if self.max_features and self.max_features < n_features:
            features = self.rng.sample(features, self.max_features)
        best_ig, best_feat, best_thresh = -1.0, 0, 0.0
        for feat in features:
            vals = sorted(set(xi[feat] for xi in X))
            thresholds = [(vals[i] + vals[i + 1]) / 2 for i in range(len(vals) - 1)]
            for thresh in thresholds[:20]:
                ig = self._information_gain(X, y, feat, thresh)
                if ig > best_ig:
                    best_ig, best_feat, best_thresh = ig, feat, thresh
        return best_feat, best_thresh, best_ig

    def _build(self, X: List[List[float]], y: List[int], depth: int) -> dict:
        if depth >= self.max_depth or len(X) < self.min_samples_split or len(set(y)) == 1:
            label = Counter(y).most_common(1)[0][0]
            return {"leaf": True, "label": label}
        feat, thresh, ig = self._best_split(X, y)
        if ig <= 0:
            label = Counter(y).most_common(1)[0][0]
            return {"leaf": True, "label": label}
        left_mask  = [xi[feat] <= thresh for xi in X]
        X_left  = [xi for xi, m in zip(X, left_mask) if m]
        y_left  = [yi for yi, m in zip(y, left_mask) if m]
        X_right = [xi for xi, m in zip(X, left_mask) if not m]
        y_right = [yi for yi, m in zip(y, left_mask) if not m]
        if not X_left or not X_right:
            label = Counter(y).most_common(1)[0][0]
            return {"leaf": True, "label": label}
        return {
            "leaf": False, "feature": feat, "threshold": thresh,
            "left": self._build(X_left, y_left, depth + 1),
            "right": self._build(X_right, y_right, depth + 1),
        }

    def fit(self, X: List[List[float]], y: List[int]) -> "DecisionTree":
        self.tree = self._build(X, y, 0)
        return self

    def _predict_one(self, x: List[float], node: dict) -> int:
        if node["leaf"]:
            return node["label"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        return self._predict_one(x, node["right"])

    def predict(self, X: List[List[float]]) -> List[int]:
        return [self._predict_one(x, self.tree) for x in X]


class RandomForest:
    """Random Forest: bagging of decision trees with random feature subsets."""

    def __init__(
        self,
        n_trees: int = 50,
        max_depth: int = 8,
        min_samples_split: int = 5,
        max_features: Optional[int] = None,
        random_state: int = 42,
    ) -> None:
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees: List[DecisionTree] = []
        self.rng = random.Random(random_state)

    def _bootstrap_sample(
        self, X: List[List[float]], y: List[int]
    ) -> Tuple[List[List[float]], List[int]]:
        n = len(X)
        indices = [self.rng.randint(0, n - 1) for _ in range(n)]
        return [X[i] for i in indices], [y[i] for i in indices]

    def fit(self, X: List[List[float]], y: List[int]) -> "RandomForest":
        n_features = len(X[0])
        max_feat = self.max_features or max(1, int(math.sqrt(n_features)))
        self.trees = []
        for i in range(self.n_trees):
            X_boot, y_boot = self._bootstrap_sample(X, y)
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_feat,
                random_state=self.random_state + i,
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
        return self

    def predict(self, X: List[List[float]]) -> List[int]:
        all_preds = [tree.predict(X) for tree in self.trees]
        return [
            Counter(preds[i] for preds in all_preds).most_common(1)[0][0]
            for i in range(len(X))
        ]


class TopicClassifier:
    """End-to-end topic classifier: TF-IDF features -> Random Forest."""

    def __init__(self, n_trees: int = 30, max_depth: int = 8) -> None:
        self.vectorizer = TFIDFVectorizer(max_features=2000)
        self.rf = RandomForest(n_trees=n_trees, max_depth=max_depth)
        self._fitted = False

    def fit(self, texts: List[str], labels: List[int]) -> "TopicClassifier":
        X = self.vectorizer.fit_transform(texts)
        self.rf.fit(X, labels)
        self._fitted = True
        return self

    def predict(self, texts: List[str]) -> List[int]:
        X = [self.vectorizer.transform(t) for t in texts]
        return self.rf.predict(X)

    @staticmethod
    def _prf1_multiclass(
        y_true: List[int], y_pred: List[int], labels: List[int]
    ) -> Dict:
        per_class = {}
        for label in labels:
            tp = sum(1 for a, b in zip(y_true, y_pred) if a == label and b == label)
            fp = sum(1 for a, b in zip(y_true, y_pred) if a != label and b == label)
            fn = sum(1 for a, b in zip(y_true, y_pred) if a == label and b != label)
            p  = tp / (tp + fp) if (tp + fp) else 0.0
            r  = tp / (tp + fn) if (tp + fn) else 0.0
            f  = 2 * p * r / (p + r) if (p + r) else 0.0
            per_class[ID_TO_TOPIC.get(label, str(label))] = {
                "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
                "support": sum(1 for a in y_true if a == label),
            }
        macro_f1 = sum(v["f1"] for v in per_class.values()) / len(per_class)
        accuracy = sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)
        return {"per_class": per_class, "macro_f1": round(macro_f1, 4), "accuracy": round(accuracy, 4)}

    def cross_validate(self, texts: List[str], labels: List[int], k: int = 5) -> Dict:
        """k-fold cross-validation."""
        n = len(texts)
        fold_size = n // k
        indices = list(range(n))
        random.Random(42).shuffle(indices)
        all_unique_labels = sorted(set(labels))
        fold_results = []
        for fold in range(k):
            val_idx   = indices[fold * fold_size: (fold + 1) * fold_size]
            train_idx = indices[:fold * fold_size] + indices[(fold + 1) * fold_size:]
            fold_clf  = TopicClassifier(n_trees=self.rf.n_trees, max_depth=self.rf.max_depth)
            fold_clf.fit([texts[i] for i in train_idx], [labels[i] for i in train_idx])
            y_pred    = fold_clf.predict([texts[i] for i in val_idx])
            metrics   = self._prf1_multiclass([labels[i] for i in val_idx], y_pred, all_unique_labels)
            fold_results.append(metrics)
            print(f"  Fold {fold + 1}/{k}: macro F1 = {metrics['macro_f1']:.4f}")
        return {
            "k_folds": k,
            "macro_f1":  round(sum(r["macro_f1"]  for r in fold_results) / k, 4),
            "accuracy":  round(sum(r["accuracy"]  for r in fold_results) / k, 4),
            "fold_details": fold_results,
        }

    def save_results(self, results: Dict, output_path: str) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
