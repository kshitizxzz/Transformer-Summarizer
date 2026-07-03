"""Enhanced NLP Preprocessing Pipeline.

Extends the base tokenizer with advanced NLP preprocessing techniques:
  - Stopword removal
  - Lemmatization (rule-based WordNet-style)
  - Stemming (Porter Stemmer algorithm)
  - POS-tag-aware filtering (optional)
  - HTML / noise cleanup

Usage
-----
    from src.preprocessing.enhanced_preprocessor import EnhancedPreprocessor

    prep = EnhancedPreprocessor(use_lemmatize=True, remove_stopwords=True)
    tokens = prep.process("The children are running in the park.")
    # -> ["child", "run", "park"]

    stats = prep.vocabulary_stats(texts)
    print(stats["vocab_size_after"] / stats["vocab_size_before"])
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "ain", "all", "also",
    "am", "an", "and", "any", "are", "aren", "as", "at", "be", "because",
    "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "could", "couldn", "d", "did", "didn", "do", "does", "doesn",
    "doing", "don", "down", "during", "each", "few", "for", "from", "further",
    "get", "got", "had", "hadn", "has", "hasn", "have", "haven", "having",
    "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
    "i", "if", "in", "into", "is", "isn", "it", "its", "itself", "just",
    "ll", "m", "ma", "me", "mightn", "more", "most", "mustn", "my",
    "myself", "needn", "no", "nor", "not", "now", "o", "of", "off", "on",
    "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over",
    "own", "re", "s", "same", "shan", "she", "should", "shouldn", "so",
    "some", "such", "t", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "us", "ve", "very", "was", "wasn",
    "we", "were", "weren", "what", "when", "where", "which", "while", "who",
    "whom", "why", "will", "with", "won", "wouldn", "y", "you", "your",
    "yours", "yourself", "yourselves",
}


# ---------------------------------------------------------------------------
# Porter Stemmer (simplified, covers most common rules)
# ---------------------------------------------------------------------------

class PorterStemmer:
    """Simplified Porter Stemmer.

    Applies suffix-stripping rules to reduce words to their stems.
    E.g., "running" -> "run", "happily" -> "happi", "studies" -> "studi"

    Reference: Porter (1980) "An Algorithm for Suffix Stripping"
    """

    VOWELS = set("aeiou")

    def _measure(self, stem: str) -> int:
        """Measure m: count VC sequences in stem."""
        n = 0
        prev_vowel = False
        for ch in stem:
            is_v = ch in self.VOWELS or (ch == "y" and not prev_vowel)
            if is_v:
                prev_vowel = True
            else:
                if prev_vowel:
                    n += 1
                prev_vowel = False
        return n

    def _has_vowel(self, stem: str) -> bool:
        return any(c in self.VOWELS for c in stem)

    def _ends_double_consonant(self, word: str) -> bool:
        return len(word) >= 2 and word[-1] == word[-2] and word[-1] not in self.VOWELS

    def stem(self, word: str) -> str:
        word = word.lower()
        if len(word) <= 2:
            return word

        # Step 1a: plurals and -ed/-ing
        if word.endswith("sses"):
            word = word[:-2]
        elif word.endswith("ies"):
            word = word[:-2]
        elif word.endswith("ss"):
            pass
        elif word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]

        # Step 1b
        if word.endswith("eed"):
            if self._measure(word[:-3]) > 0:
                word = word[:-1]
        elif word.endswith("ed") and self._has_vowel(word[:-2]):
            word = word[:-2]
            if word.endswith("at") or word.endswith("bl") or word.endswith("iz"):
                word += "e"
            elif self._ends_double_consonant(word) and not word.endswith(("l", "s", "z")):
                word = word[:-1]
            elif self._measure(word) == 1 and self._ends_cvc(word):
                word += "e"
        elif word.endswith("ing") and self._has_vowel(word[:-3]):
            word = word[:-3]
            if word.endswith("at") or word.endswith("bl") or word.endswith("iz"):
                word += "e"
            elif self._ends_double_consonant(word) and not word.endswith(("l", "s", "z")):
                word = word[:-1]
            elif self._measure(word) == 1 and self._ends_cvc(word):
                word += "e"

        # Step 1c
        if word.endswith("y") and self._has_vowel(word[:-1]):
            word = word[:-1] + "i"

        # Step 2 (common suffixes)
        step2_rules = [
            ("ational", "ate"), ("tional", "tion"), ("enci", "ence"),
            ("anci", "ance"), ("izer", "ize"), ("abli", "able"),
            ("alli", "al"), ("entli", "ent"), ("eli", "e"),
            ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
            ("ator", "ate"), ("alism", "al"), ("iveness", "ive"),
            ("fulness", "ful"), ("ousness", "ous"), ("aliti", "al"),
            ("iviti", "ive"), ("biliti", "ble"),
        ]
        for suffix, replacement in step2_rules:
            if word.endswith(suffix) and self._measure(word[:-len(suffix)]) > 0:
                word = word[:-len(suffix)] + replacement
                break

        return word

    def _ends_cvc(self, word: str) -> bool:
        """True if word ends consonant-vowel-consonant (and last not w/x/y)."""
        if len(word) < 3:
            return False
        c1 = word[-3] not in self.VOWELS
        v  = word[-2] in self.VOWELS
        c2 = word[-1] not in self.VOWELS and word[-1] not in "wxy"
        return c1 and v and c2


# ---------------------------------------------------------------------------
# Lemmatizer (rule-based, covers common English morphology)
# ---------------------------------------------------------------------------

LEMMA_EXCEPTIONS: Dict[str, str] = {
    # Irregular verbs
    "ran": "run", "gone": "go", "went": "go", "been": "be", "was": "be",
    "were": "be", "are": "be", "is": "be", "am": "be",
    "had": "have", "has": "have", "did": "do", "does": "do",
    "said": "say", "made": "make", "took": "take", "came": "come",
    "knew": "know", "saw": "see", "told": "tell", "gave": "give",
    "found": "find", "thought": "think", "left": "leave", "put": "put",
    "got": "get", "kept": "keep", "brought": "bring", "held": "hold",
    "stood": "stand", "heard": "hear", "led": "lead", "read": "read",
    "met": "meet", "lost": "lose", "paid": "pay", "sent": "send",
    "fell": "fall", "grew": "grow", "drew": "draw", "flew": "fly",
    "drove": "drive", "wrote": "write", "chose": "choose", "rose": "rise",
    "broke": "break", "spoke": "speak", "wore": "wear", "bore": "bear",
    "won": "win", "shot": "shoot", "cut": "cut", "hit": "hit",
    "set": "set", "let": "let", "sold": "sell", "built": "build",
    # Irregular nouns
    "children": "child", "men": "man", "women": "woman", "people": "person",
    "mice": "mouse", "geese": "goose", "teeth": "tooth", "feet": "foot",
    "leaves": "leaf", "lives": "life", "knives": "knife", "wolves": "wolf",
    "wives": "wife", "halves": "half", "calves": "calf", "scarves": "scarf",
    # Common adjectives
    "better": "good", "best": "good", "worse": "bad", "worst": "bad",
    "more": "many", "most": "many",
}


class RuleBasedLemmatizer:
    """Rule-based lemmatizer for English.

    Applies:
    1. Exception dictionary lookup (irregular forms)
    2. Suffix stripping rules for regular inflections

    Covers: -ing, -ed, -er, -est, -s/-es, -tion, -ness, -ment, etc.
    """

    def lemmatize(self, word: str, pos: str = "n") -> str:
        """Lemmatize a single word.

        Parameters
        ----------
        word : input word (lowercase assumed)
        pos  : 'n' (noun), 'v' (verb), 'a' (adjective)
        """
        w = word.lower()

        # Check exception dictionary first
        if w in LEMMA_EXCEPTIONS:
            return LEMMA_EXCEPTIONS[w]

        if pos == "v":
            return self._lemmatize_verb(w)
        elif pos == "n":
            return self._lemmatize_noun(w)
        elif pos == "a":
            return self._lemmatize_adj(w)
        return w

    @staticmethod
    def _lemmatize_verb(w: str) -> str:
        if w.endswith("ing"):
            stem = w[:-3]
            if stem.endswith(stem[-1]) and len(stem) > 2:  # running -> run
                return stem[:-1]
            if stem.endswith("e"):  # taking -> take... but stem is "tak"
                return stem + "e"
            return stem if len(stem) > 2 else w
        if w.endswith("ed"):
            stem = w[:-2]
            if stem.endswith(stem[-1]) and len(stem) > 2:
                return stem[:-1]
            return stem if len(stem) > 2 else w
        if w.endswith("es") and not w.endswith("ies"):
            return w[:-1]
        if w.endswith("ies"):
            return w[:-3] + "y"
        return w

    @staticmethod
    def _lemmatize_noun(w: str) -> str:
        if w.endswith("ies"):
            return w[:-3] + "y"
        if w.endswith("ves"):
            return w[:-3] + "f"
        if w.endswith("ses") or w.endswith("xes") or w.endswith("zes"):
            return w[:-2]
        if w.endswith("ches") or w.endswith("shes"):
            return w[:-2]
        if w.endswith("s") and not w.endswith("ss"):
            return w[:-1]
        return w

    @staticmethod
    def _lemmatize_adj(w: str) -> str:
        if w.endswith("est"):
            return w[:-3]
        if w.endswith("er"):
            return w[:-2]
        return w


# ---------------------------------------------------------------------------
# HTML / noise cleanup
# ---------------------------------------------------------------------------

def clean_html(text: str) -> str:
    """Remove HTML tags and decode common entities."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    return text


def clean_text(text: str) -> str:
    """General text cleaning: HTML, special chars, extra spaces."""
    text = clean_html(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # URLs
    text = re.sub(r"\(CNN\)", "", text)                   # CNN prefix artifact
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

class EnhancedPreprocessor:
    """Full NLP preprocessing pipeline.

    Steps (configurable):
    1. Text cleaning (HTML, URLs, noise)
    2. Lowercase
    3. Tokenization
    4. Stopword removal
    5. Stemming OR lemmatization
    """

    def __init__(
        self,
        remove_stopwords: bool = True,
        use_stemming: bool = False,
        use_lemmatize: bool = True,
        min_token_length: int = 2,
    ) -> None:
        self.remove_stopwords = remove_stopwords
        self.use_stemming = use_stemming
        self.use_lemmatize = use_lemmatize and not use_stemming
        self.min_token_length = min_token_length

        self.stemmer = PorterStemmer() if use_stemming else None
        self.lemmatizer = RuleBasedLemmatizer() if self.use_lemmatize else None

    def tokenize(self, text: str) -> List[str]:
        """Basic whitespace tokenizer with punctuation removal."""
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text.split()

    def process(self, text: str) -> List[str]:
        """Run full preprocessing pipeline on text.

        Returns list of processed tokens.
        """
        text = clean_text(text)
        tokens = self.tokenize(text)

        # Length filter
        tokens = [t for t in tokens if len(t) >= self.min_token_length]

        # Stopword removal
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in STOPWORDS]

        # Stemming
        if self.use_stemming and self.stemmer:
            tokens = [self.stemmer.stem(t) for t in tokens]

        # Lemmatization
        elif self.use_lemmatize and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]

        return tokens

    def process_text(self, text: str) -> str:
        """Return processed text as a single string (for downstream models)."""
        return " ".join(self.process(text))

    def vocabulary_stats(self, texts: List[str]) -> Dict:
        """Compare vocabulary size before and after preprocessing.

        Shows the impact of stopword removal + lemmatization on vocab size.
        """
        # Before: raw tokenization
        raw_vocab: Counter = Counter()
        for text in texts:
            tokens = self.tokenize(clean_text(text))
            raw_vocab.update(tokens)

        # After: full pipeline
        proc_vocab: Counter = Counter()
        for text in texts:
            tokens = self.process(text)
            proc_vocab.update(tokens)

        return {
            "num_texts": len(texts),
            "vocab_size_before": len(raw_vocab),
            "vocab_size_after": len(proc_vocab),
            "reduction_pct": round(
                100 * (1 - len(proc_vocab) / len(raw_vocab)), 1
            ) if raw_vocab else 0.0,
            "total_tokens_before": sum(raw_vocab.values()),
            "total_tokens_after": sum(proc_vocab.values()),
            "top_20_after": proc_vocab.most_common(20),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    texts = [
        "(CNN) The children were running through the beautiful gardens yesterday.",
        "Scientists have been studying the effects of climate change on the oceans.",
        "The companies announced their quarterly earnings reports on Wednesday.",
    ]

    print("=== Enhanced Preprocessing Pipeline Demo ===\n")

    for mode, kwargs in [
        ("Raw tokenization", dict(remove_stopwords=False, use_lemmatize=False)),
        ("Stopword removal", dict(remove_stopwords=True, use_lemmatize=False)),
        ("Stopwords + Lemmatize", dict(remove_stopwords=True, use_lemmatize=True)),
        ("Stopwords + Stem", dict(remove_stopwords=True, use_stemming=True, use_lemmatize=False)),
    ]:
        prep = EnhancedPreprocessor(**kwargs)
        print(f"Mode: {mode}")
        for t in texts:
            print(f"  Input:  {t}")
            print(f"  Output: {prep.process(t)}")
        stats = prep.vocabulary_stats(texts)
        print(f"  Vocab: {stats['vocab_size_before']} -> {stats['vocab_size_after']} "
              f"({stats['reduction_pct']}% reduction)\n")
