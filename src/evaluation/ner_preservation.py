"""Named Entity Recognition (NER) Preservation Analysis.

Uses spaCy NER to extract named entities (persons, organizations, locations,
dates, etc.) from source articles and generated summaries. Measures what
fraction of source entities appear in the generated summary.

High entity preservation means the model captures the WHO, WHERE, and WHEN
of the news story -- critical for factual faithfulness.

Concepts used
-------------
- Named Entity Recognition (NER)   (nlp.pdf)
- Precision, Recall, F1            (ml.pdf)
- Text preprocessing               (nlp.pdf)

Usage
-----
    from src.evaluation.ner_preservation import NERPreservationAnalyzer

    analyzer = NERPreservationAnalyzer()
    results = analyzer.analyze(articles, transformer_summaries, tfidf_summaries)
    print(results["transformer"]["avg_entity_recall"])

    # Single pair:
    result = analyzer.analyze_pair(article, summary)
    print(result["entity_recall"])
"""

from __future__ import annotations

import json
import re
import string
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Fallback NER: rule-based if spaCy unavailable
# ---------------------------------------------------------------------------

def _rule_based_ner(text: str) -> Set[str]:
    """Simple rule-based NER fallback.

    Extracts capitalized multi-word sequences as potential named entities.
    Covers: person names, place names, organization names.
    """
    # Find capitalized word sequences (not at sentence start)
    sentences = re.split(r'[.!?]\s+', text)
    entities: Set[str] = set()

    for sent in sentences:
        words = sent.split()
        i = 0
        while i < len(words):
            word = re.sub(r'[^\w\']', '', words[i])
            if (len(word) > 1 and word[0].isupper() and i > 0
                    and word.lower() not in {
                        "the", "a", "an", "in", "on", "at", "is", "are",
                        "was", "were", "has", "have", "had", "this", "that",
                        "he", "she", "they", "it", "we", "i", "his", "her",
                        "their", "its", "mr", "mrs", "ms", "dr", "said",
                    }):
                # Try to extend the entity with consecutive capitalized words
                entity_words = [word]
                j = i + 1
                while j < len(words):
                    next_word = re.sub(r'[^\w\']', '', words[j])
                    if next_word and next_word[0].isupper() and len(next_word) > 1:
                        entity_words.append(next_word)
                        j += 1
                    else:
                        break
                entity = " ".join(entity_words)
                if len(entity) > 2:
                    entities.add(entity)
                i = j
            else:
                i += 1

    return entities


def _try_spacy(text: str, nlp=None) -> Optional[Set[str]]:
    """Try to extract entities using spaCy if available."""
    if nlp is None:
        return None
    doc = nlp(text)
    return {
        ent.text.strip()
        for ent in doc.ents
        if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "NORP", "FAC", "EVENT"}
        and len(ent.text.strip()) > 1
    }


def extract_entities(text: str, nlp=None) -> Set[str]:
    """Extract named entities from text using spaCy or rule-based fallback."""
    if nlp is not None:
        result = _try_spacy(text, nlp)
        if result is not None:
            return result
    return _rule_based_ner(text)


# ---------------------------------------------------------------------------
# Entity overlap metrics
# ---------------------------------------------------------------------------

def entity_overlap(
    source_entities: Set[str], summary_entities: Set[str]
) -> Dict[str, float]:
    """Compute entity precision, recall, F1.

    entity_recall = |entities in both| / |entities in source|
    -- Recall matters most: how many source entities appear in summary?
    """
    if not source_entities:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "source_count": 0, "summary_count": len(summary_entities), "overlap_count": 0}

    # Normalize for matching: lowercase + strip
    src_norm  = {e.lower().strip() for e in source_entities}
    summ_norm = {e.lower().strip() for e in summary_entities}

    # Count entities that appear (even partially) in summary
    overlap = sum(
        1 for se in src_norm
        if any(se in ss or ss in se for ss in summ_norm)
    )

    precision = overlap / len(summ_norm) if summ_norm else 0.0
    recall    = overlap / len(src_norm)
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "precision":      round(precision, 4),
        "recall":         round(recall, 4),
        "f1":             round(f1, 4),
        "source_count":   len(source_entities),
        "summary_count":  len(summary_entities),
        "overlap_count":  overlap,
        "source_entities": sorted(source_entities),
        "summary_entities": sorted(summary_entities),
    }


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------

class NERPreservationAnalyzer:
    """Analyzes how well generated summaries preserve named entities.

    Compares Transformer-generated summaries vs TF-IDF extractive baseline
    on entity preservation -- measures factual faithfulness.
    """

    def __init__(self, use_spacy: bool = True) -> None:
        self.nlp = None
        if use_spacy:
            try:
                import spacy
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    print("Using spaCy en_core_web_sm for NER")
                except OSError:
                    print("spaCy model not found. Run: python -m spacy download en_core_web_sm")
                    print("Falling back to rule-based NER.")
            except ImportError:
                print("spaCy not installed. Falling back to rule-based NER.")

    def analyze_pair(
        self, article: str, summary: str
    ) -> Dict:
        """Analyze entity preservation for a single (article, summary) pair."""
        src_entities  = extract_entities(article, self.nlp)
        summ_entities = extract_entities(summary, self.nlp)
        metrics = entity_overlap(src_entities, summ_entities)
        return metrics

    def analyze(
        self,
        articles: List[str],
        transformer_summaries: List[str],
        tfidf_summaries: Optional[List[str]] = None,
    ) -> Dict:
        """Analyze entity preservation across a corpus.

        Parameters
        ----------
        articles : source articles
        transformer_summaries : generated by Transformer
        tfidf_summaries : generated by TF-IDF baseline (optional)

        Returns
        -------
        dict with per-pair results and averages for each method
        """
        transformer_results = []
        tfidf_results = []

        for i, (article, t_summ) in enumerate(zip(articles, transformer_summaries)):
            t_metrics = self.analyze_pair(article, t_summ)
            transformer_results.append(t_metrics)

            if tfidf_summaries and i < len(tfidf_summaries):
                tf_metrics = self.analyze_pair(article, tfidf_summaries[i])
                tfidf_results.append(tf_metrics)

        def _average(results: List[Dict]) -> Dict:
            if not results:
                return {}
            n = len(results)
            return {
                "avg_entity_recall":    round(sum(r["recall"]    for r in results) / n, 4),
                "avg_entity_precision": round(sum(r["precision"] for r in results) / n, 4),
                "avg_entity_f1":        round(sum(r["f1"]        for r in results) / n, 4),
                "avg_source_entities":  round(sum(r["source_count"]  for r in results) / n, 2),
                "avg_summary_entities": round(sum(r["summary_count"] for r in results) / n, 2),
                "total_pairs": n,
            }

        output: Dict = {
            "transformer": {**_average(transformer_results), "per_pair": transformer_results},
        }
        if tfidf_results:
            output["tfidf_baseline"] = {**_average(tfidf_results), "per_pair": tfidf_results}

        return output

    def print_comparison(self, results: Dict) -> None:
        """Pretty-print comparison between methods."""
        print("\n--- NER Entity Preservation ---")
        for method in ("transformer", "tfidf_baseline"):
            if method not in results:
                continue
            r = results[method]
            label = "Transformer" if method == "transformer" else "TF-IDF Baseline"
            print(f"  {label}:")
            print(f"    Entity Recall:    {r.get('avg_entity_recall', 0):.3f}")
            print(f"    Entity Precision: {r.get('avg_entity_precision', 0):.3f}")
            print(f"    Entity F1:        {r.get('avg_entity_f1', 0):.3f}")
            print(f"    Avg entities/src: {r.get('avg_source_entities', 0):.1f}")
        print()

    def save_results(self, results: Dict, output_path: str) -> None:
        # Remove per_pair details to keep file small
        slim = {}
        for k, v in results.items():
            slim[k] = {kk: vv for kk, vv in v.items() if kk != "per_pair"}
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(slim, f, indent=2)
        print(f"NER results saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    articles = [
        "President Barack Obama signed the Affordable Care Act in Washington D.C. "
        "The Senate and Congress both supported the bill after months of debate.",
        "Apple CEO Tim Cook announced a new iPhone at the San Francisco conference. "
        "Google and Microsoft also unveiled competing products.",
    ]
    transformer_summaries = [
        "Obama signed a new health care bill in the capital.",
        "Apple announced a new phone at a tech conference.",
    ]
    tfidf_summaries = [
        "President Barack Obama signed the Affordable Care Act in Washington D.C.",
        "Apple CEO Tim Cook announced a new iPhone at the San Francisco conference.",
    ]

    analyzer = NERPreservationAnalyzer(use_spacy=True)
    results = analyzer.analyze(articles, transformer_summaries, tfidf_summaries)
    analyzer.print_comparison(results)
