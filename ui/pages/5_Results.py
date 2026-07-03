"""Results: N-gram overlap scores, baseline comparison, and qualitative examples.

Reads from logs/eval_results.json produced by src.evaluation.evaluate.
Shows Transformer vs TF-IDF extractive baseline on:
  - Unigram / Bigram / Trigram overlap F1
  - LCS (Longest Common Subsequence) F1
  - Qualitative side-by-side examples
"""

import json
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Results", page_icon="📊", layout="wide")
st.title("📊 Evaluation Results")

results_path = PROJECT_ROOT / "logs" / "eval_results.json"

if not results_path.exists():
    st.info(
        "No evaluation results found. Run:\n\n"
        "```bash\n"
        "python -m src.evaluation.evaluate \\\n"
        "    --data_path data/test.csv \\\n"
        "    --checkpoint checkpoints/best.pt \\\n"
        "    --vocab_path data/vocab.json\n"
        "```"
    )
    st.stop()

with open(results_path, encoding="utf-8") as f:
    results = json.load(f)

# ------------------------------------------------------------------ #
# N-gram overlap headline metrics
# ------------------------------------------------------------------ #
st.subheader("N-gram Overlap: Transformer vs TF-IDF Baseline")
st.caption(
    "N-gram overlap measures what fraction of reference summary words appear in the generated output. "
    "Precision = overlap/generated, Recall = overlap/reference, F1 = harmonic mean."
)

ngram = results.get("ngram_overlap", {})
trans = ngram.get("transformer", {})
tfidf = ngram.get("tfidf_baseline", {})

if trans:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Transformer (Abstractive)**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Unigram F1", f"{trans.get('unigram', {}).get('f1', 0):.3f}")
        m2.metric("Bigram F1",  f"{trans.get('bigram',  {}).get('f1', 0):.3f}")
        m3.metric("Trigram F1", f"{trans.get('trigram', {}).get('f1', 0):.3f}")
        m4.metric("LCS F1",     f"{trans.get('lcs',     {}).get('f1', 0):.3f}")
    with col2:
        st.markdown("**TF-IDF Extractive Baseline**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Unigram F1", f"{tfidf.get('unigram', {}).get('f1', 0):.3f}")
        m2.metric("Bigram F1",  f"{tfidf.get('bigram',  {}).get('f1', 0):.3f}")
        m3.metric("Trigram F1", f"{tfidf.get('trigram', {}).get('f1', 0):.3f}")
        m4.metric("LCS F1",     f"{tfidf.get('lcs',     {}).get('f1', 0):.3f}")

    # Precision / Recall breakdown table
    st.subheader("Precision / Recall / F1 Breakdown")
    rows = []
    for metric in ("unigram", "bigram", "trigram", "lcs"):
        label = metric.capitalize() if metric != "lcs" else "LCS"
        t = trans.get(metric, {})
        tf = tfidf.get(metric, {})
        rows.append({
            "Metric": label,
            "Transformer P": round(t.get("precision", 0), 3),
            "Transformer R": round(t.get("recall",    0), 3),
            "Transformer F1": round(t.get("f1",       0), 3),
            "TF-IDF P":   round(tf.get("precision", 0), 3),
            "TF-IDF R":   round(tf.get("recall",    0), 3),
            "TF-IDF F1":  round(tf.get("f1",        0), 3),
        })
    st.dataframe(rows, use_container_width=True)

# ------------------------------------------------------------------ #
# Sentiment consistency
# ------------------------------------------------------------------ #
sentiment = results.get("sentiment_consistency", {})
if sentiment:
    st.subheader("Sentiment Consistency (Logistic Regression)")
    st.caption("Fraction of summaries that preserve the sentiment (positive/negative) of the source article.")
    c1, c2 = st.columns(2)
    c1.metric("Transformer", f"{sentiment.get('transformer', 0):.1%}")
    c2.metric("TF-IDF Baseline", f"{sentiment.get('tfidf_baseline', 0):.1%}")

# ------------------------------------------------------------------ #
# NER preservation
# ------------------------------------------------------------------ #
ner = results.get("ner_preservation", {})
if ner:
    st.subheader("Named Entity Preservation (NER)")
    st.caption("Fraction of named entities (persons, places, orgs) from the source that appear in the summary.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Transformer**")
        t_ner = ner.get("transformer", {})
        st.metric("Entity Recall",    f"{t_ner.get('avg_entity_recall', 0):.3f}")
        st.metric("Entity Precision", f"{t_ner.get('avg_entity_precision', 0):.3f}")
        st.metric("Entity F1",        f"{t_ner.get('avg_entity_f1', 0):.3f}")
    with c2:
        st.markdown("**TF-IDF Baseline**")
        tf_ner = ner.get("tfidf_baseline", {})
        st.metric("Entity Recall",    f"{tf_ner.get('avg_entity_recall', 0):.3f}")
        st.metric("Entity Precision", f"{tf_ner.get('avg_entity_precision', 0):.3f}")
        st.metric("Entity F1",        f"{tf_ner.get('avg_entity_f1', 0):.3f}")

# ------------------------------------------------------------------ #
# Topic-wise performance
# ------------------------------------------------------------------ #
topic_data = results.get("topic_classifier", {})
per_topic  = topic_data.get("per_topic_ngram", {})
cv_results = topic_data.get("cv_results", {})

if cv_results:
    st.subheader("Topic Classifier (Random Forest, 5-fold CV)")
    st.caption("Trained on TF-IDF features -> news category. Shows which topics the Transformer handles best.")
    c1, c2 = st.columns(2)
    c1.metric("Macro F1 (5-fold CV)", f"{cv_results.get('macro_f1', 0):.4f}")
    c2.metric("Accuracy (5-fold CV)", f"{cv_results.get('accuracy', 0):.4f}")

if per_topic:
    st.subheader("Per-Topic N-gram Overlap (Transformer)")
    topic_rows = [
        {"Topic": t, "Count": v["count"],
         "Unigram F1": v["unigram_f1"], "Bigram F1": v["bigram_f1"]}
        for t, v in sorted(per_topic.items(), key=lambda x: -x[1]["unigram_f1"])
    ]
    st.dataframe(topic_rows, use_container_width=True)

# ------------------------------------------------------------------ #
# Qualitative examples
# ------------------------------------------------------------------ #
st.subheader("Side-by-Side Example Outputs")
for i, ex in enumerate(results.get("examples", []), start=1):
    with st.expander(f"Example {i} — {ex.get('topic', '').title()}"):
        st.markdown("**Article (truncated)**")
        st.write(ex.get("article", "")[:500] + "...")
        st.markdown("**Reference Summary**")
        st.write(ex.get("reference", ""))
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Transformer (Abstractive)**")
            st.write(ex.get("transformer", ""))
        with c2:
            st.markdown("**TF-IDF Baseline (Extractive)**")
            st.write(ex.get("tfidf", ""))
