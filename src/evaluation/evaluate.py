"""Comprehensive Evaluation: N-gram Overlap + Baseline Comparison.

Evaluates the trained Transformer against:
  1. TF-IDF extractive baseline
  2. (Optional) Bi-LSTM + Bahdanau attention baseline

Metrics computed (all from scratch):
  - Unigram / Bigram / Trigram overlap F1
  - LCS (Longest Common Subsequence) F1
  - NER entity preservation rate
  - Sentiment consistency rate

Output is written to `logs/eval_results.json` for the Streamlit dashboard.

Example
-------
    python -m src.evaluation.evaluate \\
        --data_path data/test.csv \\
        --checkpoint checkpoints/best.pt \\
        --vocab_path data/vocab.json \\
        --max_eval_examples 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from src.evaluation.inference import Summarizer
from src.evaluation.ngram_eval import compute_ngram_corpus, print_ngram_report
from src.evaluation.sentiment_consistency import SentimentConsistencyAnalyzer
from src.evaluation.ner_preservation import NERPreservationAnalyzer
from src.evaluation.topic_classifier import TopicClassifier, assign_topic_label, TOPIC_TO_ID
from src.models.tfidf_baseline import TFIDFSummarizer
from src.preprocessing.dataset import SummarizationDataset, get_dataloader
from src.preprocessing.tokenizer import SimpleTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Transformer + baselines with n-gram overlap metrics"
    )
    parser.add_argument("--data_path",   type=str, required=True)
    parser.add_argument("--checkpoint",  type=str, default="checkpoints/best.pt")
    parser.add_argument("--vocab_path",  type=str, default="data/vocab.json")
    parser.add_argument("--article_col", type=str, default="article")
    parser.add_argument("--summary_col", type=str, default="highlights")
    parser.add_argument("--max_src_len", type=int, default=400)
    parser.add_argument("--max_tgt_len", type=int, default=100)
    parser.add_argument("--max_summary_len", type=int, default=100)
    parser.add_argument("--max_eval_examples", type=int, default=200)
    parser.add_argument("--num_qualitative_examples", type=int, default=5)
    parser.add_argument("--output_path", type=str, default="logs/eval_results.json")
    parser.add_argument("--device",      type=str, default=None)
    parser.add_argument("--tfidf_sentences", type=int, default=3,
                        help="Number of sentences for TF-IDF baseline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = SimpleTokenizer()

    # ------------------------------------------------------------------ #
    # 1. Load Transformer
    # ------------------------------------------------------------------ #
    print("Loading Transformer checkpoint...")
    summarizer = Summarizer(args.checkpoint, args.vocab_path, device=args.device)

    # ------------------------------------------------------------------ #
    # 2. Load dataset
    # ------------------------------------------------------------------ #
    dataset = SummarizationDataset(
        args.data_path,
        summarizer.vocab,
        summarizer.tokenizer,
        article_col=args.article_col,
        summary_col=args.summary_col,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
    )
    records = dataset.records
    if args.max_eval_examples:
        records = records[: args.max_eval_examples]

    articles   = [r[0] for r in records]
    references = [r[1] for r in records]

    # ------------------------------------------------------------------ #
    # 3. Generate Transformer summaries
    # ------------------------------------------------------------------ #
    print(f"Generating Transformer summaries for {len(records)} examples...")
    transformer_summaries = []
    for i, (article, _) in enumerate(records):
        summary = summarizer.summarize(
            article,
            max_src_len=args.max_src_len,
            max_summary_len=args.max_summary_len,
            method="greedy",
        )
        transformer_summaries.append(summary)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(records)}")

    # ------------------------------------------------------------------ #
    # 4. Generate TF-IDF baseline summaries
    # ------------------------------------------------------------------ #
    print("Generating TF-IDF extractive baseline summaries...")
    tfidf_model = TFIDFSummarizer(num_sentences=args.tfidf_sentences)
    tfidf_model.fit(articles)  # build corpus-level IDF
    tfidf_summaries = [tfidf_model.summarize(a) for a in articles]

    # ------------------------------------------------------------------ #
    # 5. N-gram overlap evaluation (both models vs references)
    # ------------------------------------------------------------------ #
    print("\nComputing N-gram overlap scores...")
    ref_tokens   = [tokenizer.tokenize(r) for r in references]
    trans_tokens = [tokenizer.tokenize(s) for s in transformer_summaries]
    tfidf_tokens = [tokenizer.tokenize(s) for s in tfidf_summaries]

    trans_ngram = compute_ngram_corpus(ref_tokens, trans_tokens)
    tfidf_ngram = compute_ngram_corpus(ref_tokens, tfidf_tokens)

    print("\n--- Transformer ---")
    print_ngram_report(trans_ngram)
    print("--- TF-IDF Baseline ---")
    print_ngram_report(tfidf_ngram)

    # ------------------------------------------------------------------ #
    # 6. Sentiment consistency
    # ------------------------------------------------------------------ #
    print("Running sentiment consistency analysis...")
    sentiment_analyzer = SentimentConsistencyAnalyzer()
    sentiment_analyzer.fit_with_lexicon(articles[:100])

    trans_sentiment = sentiment_analyzer.evaluate_consistency(articles, transformer_summaries)
    tfidf_sentiment = sentiment_analyzer.evaluate_consistency(articles, tfidf_summaries)
    print(f"  Transformer sentiment consistency: {trans_sentiment['consistency_rate']:.1%}")
    print(f"  TF-IDF sentiment consistency:      {tfidf_sentiment['consistency_rate']:.1%}")

    # ------------------------------------------------------------------ #
    # 7. NER preservation
    # ------------------------------------------------------------------ #
    print("Running NER entity preservation analysis...")
    ner_analyzer = NERPreservationAnalyzer(use_spacy=True)
    ner_results = ner_analyzer.analyze(articles, transformer_summaries, tfidf_summaries)
    ner_analyzer.print_comparison(ner_results)

    # ------------------------------------------------------------------ #
    # 8. Topic classifier
    # ------------------------------------------------------------------ #
    print("Running topic classifier...")
    topic_labels_str = [assign_topic_label(a) for a in articles]
    topic_labels_int = [TOPIC_TO_ID.get(l, 5) for l in topic_labels_str]

    topic_cv_results = None
    if len(articles) >= 50:
        print("  Running 5-fold cross-validation on topic classifier...")
        topic_clf = TopicClassifier(n_trees=20, max_depth=6)
        topic_cv_results = topic_clf.cross_validate(articles, topic_labels_int, k=5)
        print(f"  Topic classifier macro F1: {topic_cv_results['macro_f1']:.4f}")

    # Per-topic n-gram scores
    topic_ngram: Dict = {}
    for topic in set(topic_labels_str):
        idxs = [i for i, t in enumerate(topic_labels_str) if t == topic]
        if len(idxs) < 3:
            continue
        t_refs  = [ref_tokens[i]   for i in idxs]
        t_hyps  = [trans_tokens[i] for i in idxs]
        scores  = compute_ngram_corpus(t_refs, t_hyps)
        topic_ngram[topic] = {
            "unigram_f1": round(scores["unigram"]["f1"], 4),
            "bigram_f1":  round(scores["bigram"]["f1"],  4),
            "count": len(idxs),
        }

    # ------------------------------------------------------------------ #
    # 9. Qualitative examples
    # ------------------------------------------------------------------ #
    qualitative = []
    for i in range(min(args.num_qualitative_examples, len(articles))):
        qualitative.append({
            "article":     articles[i],
            "reference":   references[i],
            "transformer": transformer_summaries[i],
            "tfidf":       tfidf_summaries[i],
            "topic":       topic_labels_str[i],
        })

    # ------------------------------------------------------------------ #
    # 10. Write output
    # ------------------------------------------------------------------ #
    results = {
        "ngram_overlap": {
            "transformer": {
                "unigram": trans_ngram["unigram"],
                "bigram":  trans_ngram["bigram"],
                "trigram": trans_ngram["trigram"],
                "lcs":     trans_ngram["lcs"],
            },
            "tfidf_baseline": {
                "unigram": tfidf_ngram["unigram"],
                "bigram":  tfidf_ngram["bigram"],
                "trigram": tfidf_ngram["trigram"],
                "lcs":     tfidf_ngram["lcs"],
            },
        },
        "sentiment_consistency": {
            "transformer":    trans_sentiment["consistency_rate"],
            "tfidf_baseline": tfidf_sentiment["consistency_rate"],
        },
        "ner_preservation": {
            "transformer": {
                "avg_entity_recall":    ner_results.get("transformer", {}).get("avg_entity_recall", 0),
                "avg_entity_precision": ner_results.get("transformer", {}).get("avg_entity_precision", 0),
                "avg_entity_f1":        ner_results.get("transformer", {}).get("avg_entity_f1", 0),
            },
            "tfidf_baseline": {
                "avg_entity_recall":    ner_results.get("tfidf_baseline", {}).get("avg_entity_recall", 0),
                "avg_entity_precision": ner_results.get("tfidf_baseline", {}).get("avg_entity_precision", 0),
                "avg_entity_f1":        ner_results.get("tfidf_baseline", {}).get("avg_entity_f1", 0),
            },
        },
        "topic_classifier": {
            "cv_results":      topic_cv_results,
            "per_topic_ngram": topic_ngram,
        },
        "eval_set_size": len(records),
        "examples":      qualitative,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
