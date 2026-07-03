"""Sentiment Consistency Analysis Page.

Shows how well the Transformer preserves the sentiment of the source article
in its generated summary. Uses a Logistic Regression classifier trained on
TF-IDF features to predict positive/negative sentiment.

Concepts: Sentiment Analysis, Logistic Regression, TF-IDF, Precision/Recall/F1
"""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="wide")
st.title("💬 Sentiment Consistency Analysis")

st.markdown("""
**Goal:** Does the generated summary preserve the emotional tone of the source article?

**Method:**
1. Train a **Logistic Regression** classifier on TF-IDF features to predict sentiment
2. Apply it to both the source article and generated summary
3. Measure **consistency rate** = fraction of pairs where sentiment matches

**Concepts used:** Sentiment Analysis (nlp.pdf), Logistic Regression (ml.pdf),
TF-IDF Feature Engineering (nlp.pdf), Precision/Recall/F1 (ml.pdf)
""")

st.divider()

# ------------------------------------------------------------------ #
# Live demo: classify sentiment of custom text
# ------------------------------------------------------------------ #
st.subheader("Live Sentiment Classifier Demo")
st.caption("Using Logistic Regression trained on TF-IDF features with lexicon-based pseudo-labels.")

train_examples = [
    "The company achieved record profits and celebrated a successful year.",
    "Violence erupted in the city leaving many people dead.",
    "Scientists made a breakthrough discovery in cancer treatment.",
    "The disaster killed hundreds and destroyed homes across the region.",
    "The team won the championship after years of hard work.",
    "Crime rates surged to the highest levels in decades.",
    "Investors praised the company's outstanding performance and growth.",
    "The economic collapse caused widespread poverty and unemployment.",
    "The new policy will improve healthcare access for millions of people.",
    "The attack on civilians drew widespread international condemnation.",
    "Researchers found a promising new treatment for the disease.",
    "Floods destroyed crops leaving farmers facing financial ruin.",
]

try:
    from src.evaluation.sentiment_consistency import SentimentConsistencyAnalyzer

    @st.cache_resource
    def load_sentiment_model():
        analyzer = SentimentConsistencyAnalyzer()
        analyzer.fit_with_lexicon(train_examples)
        return analyzer

    analyzer = load_sentiment_model()

    col1, col2 = st.columns(2)
    with col1:
        article_text = st.text_area(
            "Source Article:",
            value="The economy is recovering strongly with positive growth across all sectors. "
                  "Employment rates have risen to record highs and consumer confidence is at its best.",
            height=120,
        )
    with col2:
        summary_text = st.text_area(
            "Generated Summary:",
            value="Economy shows strong recovery with record employment.",
            height=120,
        )

    if st.button("Analyze Sentiment Consistency"):
        a_label, a_conf = analyzer.predict_sentiment(article_text)
        s_label, s_conf = analyzer.predict_sentiment(summary_text)
        consistent = (a_label == s_label)

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Article Sentiment",
            "Positive 😊" if a_label else "Negative 😔",
            f"Confidence: {a_conf:.1%}",
        )
        c2.metric(
            "Summary Sentiment",
            "Positive 😊" if s_label else "Negative 😔",
            f"Confidence: {s_conf:.1%}",
        )
        c3.metric(
            "Consistent?",
            "✅ Yes" if consistent else "❌ No",
        )

        if consistent:
            st.success("The summary preserves the sentiment of the source article.")
        else:
            st.warning(
                "Sentiment mismatch detected. The summary may not faithfully "
                "represent the tone of the source article."
            )

except Exception as e:
    st.error(f"Could not load sentiment module: {e}")

st.divider()

# ------------------------------------------------------------------ #
# How it works
# ------------------------------------------------------------------ #
st.subheader("How the Classifier Works")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Step 1: TF-IDF Feature Engineering**")
    st.code("""
# For each token in text:
TF(token) = count(token) / total_tokens

IDF(token) = log((1 + N) / (1 + df(token))) + 1

TF-IDF(token) = TF(token) * IDF(token)

# Sentence -> sparse vector of TF-IDF weights
    """, language="python")

with col2:
    st.markdown("**Step 2: Logistic Regression**")
    st.code("""
# Binary cross-entropy loss:
P(y=1|x) = sigmoid(w^T x + b)

Loss = -[y*log(p) + (1-y)*log(1-p)]

# Gradient update:
w <- w - lr * (p - y) * x
b <- b - lr * (p - y)

# L2 regularization:
w <- w - lr * lambda * w
    """, language="python")

st.markdown("**Step 3: Pseudo-labeling (No manual labels needed)**")
st.info(
    "Since we don't have sentiment labels for CNN/DailyMail articles, we use a "
    "**lexicon-based pseudo-labeler**: articles with more positive words (success, "
    "win, celebrated, helped...) are labeled positive; articles with more negative "
    "words (killed, crime, disaster, attack...) are labeled negative. "
    "The Logistic Regression then learns a weighted combination of TF-IDF features "
    "that generalizes beyond the lexicon."
)

# ------------------------------------------------------------------ #
# Corpus results (from eval_results.json)
# ------------------------------------------------------------------ #
import json
results_path = PROJECT_ROOT / "logs" / "eval_results.json"
if results_path.exists():
    with open(results_path) as f:
        results = json.load(f)
    sentiment = results.get("sentiment_consistency", {})
    if sentiment:
        st.subheader("Corpus-Level Consistency Results")
        c1, c2 = st.columns(2)
        c1.metric("Transformer Consistency", f"{sentiment.get('transformer', 0):.1%}")
        c2.metric("TF-IDF Baseline Consistency", f"{sentiment.get('tfidf_baseline', 0):.1%}")
        st.caption(
            f"Evaluated on {results.get('eval_set_size', '?')} article-summary pairs."
        )
