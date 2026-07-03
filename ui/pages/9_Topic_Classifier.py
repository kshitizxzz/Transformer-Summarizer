"""Topic Classifier Page.

Trains a Random Forest classifier (TF-IDF features -> news topic) and
evaluates it with 5-fold cross-validation. Also shows per-topic summarization
performance to reveal which news categories the Transformer handles best.

Concepts: Random Forest, Decision Trees, Bagging, Bootstrap, 5-fold CV,
          Entropy/Information Gain, TF-IDF, Precision/Recall/F1 (ml.pdf)
"""

import json
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Topic Classifier", page_icon="🌲", layout="wide")
st.title("🌲 Random Forest Topic Classifier")

st.markdown("""
**Goal:** Classify news articles by topic (Politics, Sports, Business, Health, Crime)
using a **Random Forest** trained on **TF-IDF features**, then stratify the
Transformer's summarization performance by topic.

**Concepts:** Random Forest (Bagging + Decision Trees), Bootstrap Sampling,
Information Gain / Entropy, 5-fold Cross-Validation, TF-IDF Feature Engineering
""")

# ------------------------------------------------------------------ #
# Architecture explanation
# ------------------------------------------------------------------ #
st.subheader("Random Forest Architecture")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**1. TF-IDF Vectorization**")
    st.info(
        "Each article is represented as a sparse vector of TF-IDF weights "
        "(top 2000 features). High TF-IDF = rare but important word for this doc."
    )
with col2:
    st.markdown("**2. Random Forest = Bagging + Trees**")
    st.info(
        "30 Decision Trees, each trained on a **bootstrap sample** "
        "(sampling with replacement). Each split considers √(features) random features. "
        "Final prediction = majority vote."
    )
with col3:
    st.markdown("**3. Decision Tree Splits**")
    st.info(
        "At each node, pick the feature + threshold that maximizes "
        "**Information Gain** (entropy reduction). "
        "IG = H(parent) - weighted_avg(H(children))"
    )

# ------------------------------------------------------------------ #
# Live demo: classify a custom article
# ------------------------------------------------------------------ #
st.subheader("Live Topic Classifier Demo")

try:
    from src.evaluation.topic_classifier import (
        TopicClassifier, assign_topic_label, TOPIC_TO_ID, ID_TO_TOPIC,
        TOPIC_KEYWORDS
    )

    demo_texts = {
        "Politics": "The president signed a new executive order today after Congress voted to pass the legislation.",
        "Sports":   "The basketball team won the championship game scoring 98 points in the final quarter.",
        "Business": "The company reported record quarterly profits with revenue growing 25 percent year over year.",
        "Health":   "Doctors found a new treatment for the disease that reduced symptoms by 60 percent in clinical trials.",
        "Crime":    "Police arrested three suspects in connection with the armed robbery at a downtown bank.",
    }

    selected = st.selectbox("Choose a demo article:", list(demo_texts.keys()))
    article_input = st.text_area("Article text:", value=demo_texts[selected], height=100)

    if st.button("Classify Topic"):
        predicted = assign_topic_label(article_input)
        text_lower = article_input.lower()

        # Count keyword hits per topic
        kw_hits = {
            topic: sum(1 for kw in kws if kw in text_lower)
            for topic, kws in TOPIC_KEYWORDS.items()
        }

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Topic", predicted.upper())
        with col2:
            st.markdown("**Keyword Hits per Topic:**")
            kw_data = sorted(kw_hits.items(), key=lambda x: -x[1])
            st.dataframe(
                [{"Topic": t, "Keyword Hits": c} for t, c in kw_data],
                use_container_width=True
            )

        st.caption(
            "After keyword-based labeling, a **Random Forest** trained on TF-IDF features "
            "learns to generalize beyond simple keyword matching."
        )

except Exception as e:
    st.error(f"Could not load topic classifier: {e}")

st.divider()

# ------------------------------------------------------------------ #
# Decision Tree / Information Gain explainer
# ------------------------------------------------------------------ #
st.subheader("Decision Tree: Entropy and Information Gain")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Entropy (Impurity measure):**")
    st.latex(r"H(S) = -\sum_{c} p_c \log_2 p_c")
    st.code("""
# Pure node (all same class): H = 0
# Max impurity (equal split): H = log2(num_classes)

# Example: 60 politics, 40 sports
H = -(0.6 * log2(0.6) + 0.4 * log2(0.4))
H = 0.971  # bits
    """, language="python")

with col2:
    st.markdown("**Information Gain:**")
    st.latex(r"IG = H(parent) - \frac{|left|}{n} H(left) - \frac{|right|}{n} H(right)")
    st.code("""
# Best split = feature + threshold that maximizes IG
# Random Forest: at each node, consider only
#   sqrt(n_features) randomly selected features
# -> reduces correlation between trees
    """, language="python")

# ------------------------------------------------------------------ #
# Results from eval_results.json
# ------------------------------------------------------------------ #
results_path = PROJECT_ROOT / "logs" / "eval_results.json"
if results_path.exists():
    with open(results_path) as f:
        results = json.load(f)

    topic_data = results.get("topic_classifier", {})
    cv_results = topic_data.get("cv_results", {})
    per_topic  = topic_data.get("per_topic_ngram", {})

    if cv_results:
        st.subheader("Cross-Validation Results (5-fold)")
        c1, c2 = st.columns(2)
        c1.metric("Macro F1", f"{cv_results.get('macro_f1', 0):.4f}")
        c2.metric("Accuracy",  f"{cv_results.get('accuracy', 0):.4f}")

        fold_details = cv_results.get("fold_details", [])
        if fold_details:
            st.markdown("**Per-fold results:**")
            fold_rows = [
                {"Fold": i + 1,
                 "Macro F1": r.get("macro_f1", 0),
                 "Accuracy": r.get("accuracy", 0)}
                for i, r in enumerate(fold_details)
            ]
            st.dataframe(fold_rows, use_container_width=True)

    if per_topic:
        st.subheader("Per-Topic Summarization Performance (Transformer)")
        st.caption("Unigram/Bigram overlap F1 by news category. Shows which topics the Transformer handles best.")
        rows = sorted(
            [{"Topic": t.title(), "Count": v["count"],
              "Unigram F1": v["unigram_f1"], "Bigram F1": v["bigram_f1"]}
             for t, v in per_topic.items()],
            key=lambda x: -x["Unigram F1"]
        )
        st.dataframe(rows, use_container_width=True)
