"""Named entity preservation analysis.

Uses spaCy (or a rule-based fallback) to extract named entities from source
articles and generated summaries, then measures what fraction of source
entities appear in the summary — a proxy for factual faithfulness.

Usage:
    analyzer = NERPreservationAnalyzer()
    results  = analyzer.analyze(articles, transformer_summaries, tfidf_summaries)
    print(results["transformer"]["avg_entity_recall"])
"""

from __future__ import annotations

import json
import re
import string
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def _rule_based_ner(text: str) -> Set[str]:
    """Extract capitalized noun phrases as candidate named entities."""
    sentences = re.split(r'[.!?]\s+', text)
    entities: Set[str] = set()
    _skip = {
        "the", "a", "an", "in", "on", "at", "is", "are", "was", "were",
        "has", "have", "had", "this", "that", "he", "she", "they", "it",
        "we", "i", "his", "her", "their", "its", "mr", "mrs", "ms", "dr",
        "said", "also", "with", "from", "for", "after", "before",
    }
    for sent in sentences:
        words = sent.split()
        i = 0
        while i < len(words):
            word = re.sub(r"[^\w']", "", words[i])
            if len(word) > 1 and word[0].isupper() and i > 0 and word.lower() not in _skip:
                entity_words = [word]
                j = i + 1
                while j < len(words):
                    nw = re.sub(r"[^\w']", "", words[j])
                    if nw and nw[0].isupper() and len(nw) > 1:
                        entity_words.append(nw)
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


def extract_entities(text: str, nlp=None) -> Set[str]:
    if nlp is not None:
        try:
            doc = nlp(text)
            result = {
                ent.text.strip() for ent in doc.ents
                if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "NORP", "FAC", "EVENT"}
                and len(ent.text.strip()) > 1
            }
            return result
        except Exception:
            pass
    return _rule_based_ner(text)


def entity_overlap(source_entities: Set[str], summary_entities: Set[str]) -> Dict[str, float]:
    if not source_entities:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "source_count": 0, "summary_count": len(summary_entities), "overlap_count": 0}
    src_norm  = {e.lower().strip() for e in source_entities}
    summ_norm = {e.lower().strip() for e in summary_entities}
    overlap   = sum(1 for se in src_norm if any(se in ss or ss in se for ss in summ_norm))
    p  = overlap / len(summ_norm) if summ_norm else 0.0
    r  = overlap / len(src_norm)
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {
        "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
        "source_count":   len(source_entities),
        "summary_count":  len(summary_entities),
        "overlap_count":  overlap,
        "source_entities":  sorted(source_entities),
        "summary_entities": sorted(summary_entities),
    }


class NERPreservationAnalyzer:
    def __init__(self, use_spacy: bool = True) -> None:
        self.nlp = None
        if use_spacy:
            try:
                import spacy
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    pass  # fall back to rule-based
            except ImportError:
                pass

    def analyze_pair(self, article: str, summary: str) -> Dict:
        return entity_overlap(
            extract_entities(article, self.nlp),
            extract_entities(summary, self.nlp),
        )

    def analyze(
        self,
        articles: List[str],
        transformer_summaries: List[str],
        tfidf_summaries: Optional[List[str]] = None,
    ) -> Dict:
        transformer_results = [self.analyze_pair(a, s) for a, s in zip(articles, transformer_summaries)]
        tfidf_results = (
            [self.analyze_pair(a, s) for a, s in zip(articles, tfidf_summaries)]
            if tfidf_summaries else []
        )

        def _avg(results: List[Dict]) -> Dict:
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

        output = {"transformer": {**_avg(transformer_results), "per_pair": transformer_results}}
        if tfidf_results:
            output["tfidf_baseline"] = {**_avg(tfidf_results), "per_pair": tfidf_results}
        return output

    def print_comparison(self, results: Dict) -> None:
        print("\n--- NER Entity Preservation ---")
        for method in ("transformer", "tfidf_baseline"):
            if method not in results:
                continue
            r = results[method]
            label = "Transformer" if method == "transformer" else "TF-IDF Baseline"
            print(f"  {label}:  Recall={r.get('avg_entity_recall',0):.3f}  "
                  f"P={r.get('avg_entity_precision',0):.3f}  F1={r.get('avg_entity_f1',0):.3f}")
        print()

    def save_results(self, results: Dict, output_path: str) -> None:
        slim = {k: {kk: vv for kk, vv in v.items() if kk != "per_pair"} for k, v in results.items()}
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(slim, f, indent=2)
