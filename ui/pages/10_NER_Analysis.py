"""NER Entity Preservation Analysis Page.

Shows how well the Transformer preserves named entities (people, places,
organizations) from source articles in the generated summaries.
"""

import json
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="NER Analysis", page_icon="🏷️", layout="wide")
st.title("🏷️ Named Entity Preservation (NER)")

st.markdown("""
**Goal:** Measure factual faithfulness — does the summary retain the key
**named entities** (WHO, WHERE, WHEN) from the source article?

**Method:** Apply Named Entity Recognition (NER) to both the source and summary.
Compute entity recall = entities appearing in summary / entities in source.

**Entity types tracked:** PERSON, ORGANIZATION, LOCATION, COUNTRY, EVENT
""")

# ------------------------------------------------------------------ #
# Live NER demo
# ------------------------------------------------------------------ #
st.subheader("Live NER Demo")

try:
    from src.evaluation.ner_preservation import NERPreservationAnalyzer, extract_entities

    @st.cache_resource
    def load_ner():
        return NERPreservationAnalyzer(use_spacy=True)

    ner_analyzer = load_ner()

    col1, col2 = st.columns(2)
    with col1:
        art = st.text_area(
            "Source Article:",
            value=(
                "President Barack Obama signed the Affordable Care Act in Washington D.C. "
                "Senator John McCain opposed the bill while Nancy Pelosi championed it in Congress. "
                "The law was passed after months of debate in the United States Senate."
            ),
            height=120,
        )
    with col2:
        summ = st.text_area(
            "Generated Summary:",
            value="Obama signed a new health care law after debate in Congress.",
            height=120,
        )

    if st.button("Run NER Analysis"):
        art_entities  = extract_entities(art,  ner_analyzer.nlp)
        summ_entities = extract_entities(summ, ner_analyzer.nlp)
        metrics = ner_analyzer.analyze_pair(art, summ)

        c1, c2, c3 = st.columns(3)
        c1.metric("Entity Recall",    f"{metrics['recall']:.3f}",
                  help="Fraction of source entities found in summary")
        c2.metric("Entity Precision", f"{metrics['precision']:.3f}",
                  help="Fraction of summary entities that came from source")
        c3.metric("Entity F1",        f"{metrics['f1']:.3f}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Source entities:**")
            for e in sorted(art_entities):
                st.write(f"• {e}")
        with col2:
            st.markdown("**Summary entities:**")
            for e in sorted(summ_entities):
                found = any(e.lower() in s.lower() or s.lower() in e.lower()
                            for s in art_entities)
                prefix = "✅" if found else "❌"
                st.write(f"{prefix} {e}")

        if not ner_analyzer.nlp:
            st.caption("Using rule-based NER (capitalized noun phrases). For better results: `pip install spacy && python -m spacy download en_core_web_sm`")

except Exception as e:
    st.error(f"Could not load NER module: {e}")

st.divider()

# ------------------------------------------------------------------ #
# Corpus results
# ------------------------------------------------------------------ #
results_path = PROJECT_ROOT / "logs" / "eval_results.json"
if results_path.exists():
    with open(results_path) as f:
        results = json.load(f)

    ner = results.get("ner_preservation", {})
    if ner:
        st.subheader("Corpus-Level NER Results")
        st.caption(f"Evaluated on {results.get('eval_set_size', '?')} article-summary pairs.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Transformer (Abstractive)**")
            t_ner = ner.get("transformer", {})
            st.metric("Avg Entity Recall",    f"{t_ner.get('avg_entity_recall', 0):.3f}")
            st.metric("Avg Entity Precision", f"{t_ner.get('avg_entity_precision', 0):.3f}")
            st.metric("Avg Entity F1",        f"{t_ner.get('avg_entity_f1', 0):.3f}")
        with col2:
            st.markdown("**TF-IDF Baseline (Extractive)**")
            tf_ner = ner.get("tfidf_baseline", {})
            st.metric("Avg Entity Recall",    f"{tf_ner.get('avg_entity_recall', 0):.3f}")
            st.metric("Avg Entity Precision", f"{tf_ner.get('avg_entity_precision', 0):.3f}")
            st.metric("Avg Entity F1",        f"{tf_ner.get('avg_entity_f1', 0):.3f}")

        st.info(
            "**Expected pattern:** Extractive summaries should have higher entity recall "
            "because they copy sentences verbatim. Abstractive (Transformer) summaries may "
            "paraphrase or omit some entities, but generate more fluent, concise text."
        )
