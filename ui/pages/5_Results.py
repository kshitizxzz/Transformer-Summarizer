"""Results: ROUGE / perplexity scores and qualitative example summaries.

Reads from `logs/eval_results.json`, expected to look like:

    {
      "rouge": {"rouge-1": {...}, "rouge-2": {...}, "rouge-l": {...}},
      "perplexity": 23.4,
      "examples": [
        {"article": "...", "reference": "...", "generated": "..."},
        ...
      ]
    }

This file is produced by running evaluation over a held-out set, e.g. a
small script that calls `src.evaluation.rouge.compute_rouge_corpus` and
`src.evaluation.perplexity.compute_perplexity` and dumps the result.
"""

import json
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Results", page_icon="🏆", layout="wide")
st.title("🏆 Results")

results_path = PROJECT_ROOT / "logs" / "eval_results.json"

if not results_path.exists():
    st.info(
        "No evaluation results found yet. Run evaluation (ROUGE + perplexity) over a held-out "
        "split and save the output to `logs/eval_results.json` to populate this page."
    )
else:
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    st.subheader("Headline metrics")
    cols = st.columns(4)
    rouge = results.get("rouge", {})
    cols[0].metric("ROUGE-1 F1", f"{rouge.get('rouge-1', {}).get('f1', 0):.3f}")
    cols[1].metric("ROUGE-2 F1", f"{rouge.get('rouge-2', {}).get('f1', 0):.3f}")
    cols[2].metric("ROUGE-L F1", f"{rouge.get('rouge-l', {}).get('f1', 0):.3f}")
    cols[3].metric("Perplexity", f"{results.get('perplexity', float('nan')):.2f}")

    st.subheader("Example outputs")
    for i, ex in enumerate(results.get("examples", []), start=1):
        with st.expander(f"Example {i}"):
            st.markdown("**Article**")
            st.write(ex.get("article", ""))
            st.markdown("**Reference summary**")
            st.write(ex.get("reference", ""))
            st.markdown("**Generated summary**")
            st.write(ex.get("generated", ""))
