"""Topic Classifier: Random Forest on TF-IDF features.

Trains a Random Forest classifier to assign news articles to topic
categories. Evaluates per-class precision, recall, and F1 using
5-fold cross-validation. Also stratifies summarization evaluation
by topic to show which categories the Transformer handles best.

Concepts used
-------------
- Random Forest           (ml.pdf: Bagging + Decision Trees)
- Decision Trees          (ml.pdf: Entropy / Gini / Information Gain)
- Bagging & Bootstrap     (ml.pdf)
- 5-fold Cross-Validation (ml.pdf)
- TF-IDF Feature Engineering (nlp.pdf / ml.pdf)
- Precision, Recall, F1  (ml.pdf)
- Confusion Matrix        (ml.pdf)
- Entropy / Information Gain (ml.pdf: Decision Tree splitting)

Usage
-----
    from src.evaluation.topic_classifier import TopicClassifier

    clf = TopicClassifier(n_trees=50)
    clf.fit(train_texts, train_labels)
    predictions = clf.predict(test_texts)
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


# ---------------------------------------------------------------------------
# Keyword-based topic labeler (CNN/DailyMail has no explicit topic labels)
# ---------------------------------------------------------------------------

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


def assign_topic_label(text: str) -> str:
    """Assign a topic label based on keyword counts."""
    text_lower = text.lower()
    scores = {
        topic: sum(1 for kw in kws if kw in text_lower)
        for topic, kws in TOPIC_KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "other"
    return best


TOPIC_TO_ID = {t: i for i, t in enumerate(["politics", "sports", "business", "health", "crime", "other"])}
ID_TO_TOPIC = {i: t for t, i in TOPIC_TO_ID.items()}


# ---------------------------------------------------------------------------
# Stopwords and tokenizer
# ---------------------------------------------------------------------------

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "and", "but", "or", "not", "no", "it", "its", "he", "she",
    "they", "we", "i", "that", "this", "which", "who", "what", "also",
    "said", "say", "says", "new", "one", "two", "three", "year", "years",
}


def _tokenize(text: str) -> List[str]:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return [t for t in text.split() if t not in STOPWORDS and len(t) > 2]


# ---------------------------------------------------------------------------
# TF-IDF Vectorizer (from scratch)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Decision Tree (for use inside Random Forest)
# ---------------------------------------------------------------------------

class DecisionTree:
    """CART decision tree using information gain (entropy) criterion.

    Concepts: entropy, information gain, recursive splitting, max_depth,
    min_samples_split -- all covered in ml.pdf.
    """

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
        """IG = H(parent) - weighted_avg(H(children))."""
        left_y  = [yi for xi, yi in zip(X, y) if xi[feature] <= threshold]
        right_y = [yi for xi, yi in zip(X, y) if xi[feature] > threshold]
        n = len(y)
        if not left_y or not right_y:
            return 0.0
        H_parent = self._entropy(y)
        H_children = (len(left_y) / n) * self._entropy(left_y) + \
                     (len(right_y) / n) * self._entropy(right_y)
        return H_parent - H_children

    def _best_split(
        self, X: List[List[float]], y: List[int]
    ) -> Tuple[int, float, float]:
        """Find feature + threshold with highest information gain."""
        n_features = len(X[0])
        features = list(range(n_features))
        if self.max_features and self.max_features < n_features:
            features = self.rng.sample(features, self.max_features)

        best_ig, best_feat, best_thresh = -1.0, 0, 0.0
        for feat in features:
            vals = sorted(set(xi[feat] for xi in X))
            thresholds = [(vals[i] + vals[i + 1]) / 2 for i in range(len(vals) - 1)]
            for thresh in thresholds[:20]:  # limit for speed
                ig = self._information_gain(X, y, feat, thresh)
                if ig > best_ig:
                    best_ig, best_feat, best_thresh = ig, feat, thresh
        return best_feat, best_thresh, best_ig

    def _build(self, X: List[List[float]], y: List[int], depth: int) -> dict:
        # Leaf conditions
        if (depth >= self.max_depth or
                len(X) < self.min_samples_split or
                len(set(y)) == 1):
            label = Counter(y).most_common(1)[0][0]
            return {"leaf": True, "label": label, "proba": Counter(y)}

        feat, thresh, ig = self._best_split(X, y)
        if ig <= 0:
            label = Counter(y).most_common(1)[0][0]
            return {"leaf": True, "label": label, "proba": Counter(y)}

        left_mask  = [xi[feat] <= thresh for xi in X]
        X_left  = [xi for xi, m in zip(X, left_mask) if m]
        y_left  = [yi for yi, m in zip(y, left_mask) if m]
        X_right = [xi for xi, m in zip(X, left_mask) if not m]
        y_right = [yi for yi, m in zip(y, left_mask) if not m]

        if not X_left or not X_right:
            label = Counter(y).most_common(1)[0][0]
            return {"leaf": True, "label": label, "proba": Counter(y)}

        return {
            "leaf": False,
            "feature": feat,
            "threshold": thresh,
            "left":  self._build(X_left,  y_left,  depth + 1),
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


# ---------------------------------------------------------------------------
# Random Forest (Bagging of Decision Trees)
# ---------------------------------------------------------------------------

class RandomForest:
    """Random Forest classifier (Bagging of Decision Trees).

    Each tree:
      1. Bootstrap sample from training data (sampling with replacement)
      2. At each split, consider only sqrt(n_features) random features
      3. Final prediction = majority vote across all trees

    Concepts: Bagging, Bootstrap sampling, Ensemble, Information Gain, Entropy
    """

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
        """Bootstrap: sample n examples WITH replacement."""
        n = len(X)
        indices = [self.rng.randint(0, n - 1) for _ in range(n)]
        return [X[i] for i in indices], [y[i] for i in indices]

    def fit(self, X: List[List[float]], y: List[int]) -> "RandomForest":
        n_features = len(X[0])
        # sqrt(n_features) features per split -- standard RF heuristic
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
        """Majority vote across all trees."""
        all_preds = [tree.predict(X) for tree in self.trees]
        final = []
        for i in range(len(X)):
            votes = Counter(preds[i] for preds in all_preds)
            final.append(votes.most_common(1)[0][0])
        return final


# ---------------------------------------------------------------------------
# Main classifier with evaluation
# ---------------------------------------------------------------------------

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

    def cross_validate(
        self, texts: List[str], labels: List[int], k: int = 5
    ) -> Dict:
        """k-fold cross-validation.

        Splits data into k folds, trains on k-1, evaluates on 1, repeats.
        Returns averaged macro F1 across all folds.
        """
        n = len(texts)
        fold_size = n // k
        indices = list(range(n))
        random.Random(42).shuffle(indices)

        fold_results = []
        all_unique_labels = sorted(set(labels))

        for fold in range(k):
            val_idx   = indices[fold * fold_size: (fold + 1) * fold_size]
            train_idx = indices[:fold * fold_size] + indices[(fold + 1) * fold_size:]

            X_train = [texts[i] for i in train_idx]
            y_train = [labels[i] for i in train_idx]
            X_val   = [texts[i] for i in val_idx]
            y_val   = [labels[i] for i in val_idx]

            # Fresh vectorizer + RF per fold
            fold_clf = TopicClassifier(
                n_trees=self.rf.n_trees, max_depth=self.rf.max_depth
            )
            fold_clf.fit(X_train, y_train)
            y_pred = fold_clf.predict(X_val)

            metrics = self._prf1_multiclass(y_val, y_pred, all_unique_labels)
            fold_results.append(metrics)
            print(f"  Fold {fold + 1}/{k}: macro F1 = {metrics['macro_f1']:.4f}, "
                  f"accuracy = {metrics['accuracy']:.4f}")

        avg_macro_f1 = sum(r["macro_f1"] for r in fold_results) / k
        avg_accuracy = sum(r["accuracy"] for r in fold_results) / k

        return {
            "k_folds": k,
            "macro_f1": round(avg_macro_f1, 4),
            "accuracy": round(avg_accuracy, 4),
            "fold_details": fold_results,
        }

    def label_texts(self, texts: List[str]) -> List[str]:
        """Return topic names for a list of texts (using keyword labeler)."""
        return [assign_topic_label(t) for t in texts]

    def save_results(self, results: Dict, output_path: str) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Topic classifier results saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick demo
    texts = [
        "The president signed a new law in Congress today.",
        "The team won the championship in the final game.",
        "Stock market hit record highs as profits surged.",
        "Doctors found a new treatment for the disease.",
        "Police arrested the suspect in the shooting case.",
        "The senator voted against the bill in parliament.",
        "Athletes competed in the Olympic basketball tournament.",
        "The company reported billion dollar revenue growth.",
        "Hospital patients received the new vaccine therapy.",
        "Criminal was convicted and sentenced to prison.",
    ]
    labels_str = [assign_topic_label(t) for t in texts]
    labels_int = [TOPIC_TO_ID.get(l, TOPIC_TO_ID["other"]) for l in labels_str]

    print("Auto-assigned labels:")
    for t, l in zip(texts, labels_str):
        print(f"  [{l}] {t[:60]}")

    print("\nRunning 3-fold CV...")
    clf = TopicClassifier(n_trees=10, max_depth=5)
    results = clf.cross_validate(texts, labels_int, k=3)
    print(f"\nMacro F1 (3-fold CV): {results['macro_f1']:.4f}")
    print(f"Accuracy (3-fold CV): {results['accuracy']:.4f}")
