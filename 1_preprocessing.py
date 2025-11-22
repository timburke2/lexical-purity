"""
Preprocess the NLTK Brown corpus into cleaned JSONL documents and indices.

Inputs:
    NLTK Brown corpus downloads plus script-level configuration flags.
Outputs:
    data/processed/brown_<variant>.jsonl along with term_index_<variant>.json
    and term_freqs_<variant>.json for both the stopword-filtered and optional
    stopword-retaining variants.
Usage:
    python scripts/1_preprocessing.py
"""

import os
import json
import random
from collections import Counter
from nltk.corpus import brown, stopwords
from nltk import download
import string

# -----------------------------
# Configuration
# -----------------------------
OUTPUT_DIR = "data/processed"
LOWERCASE = True
KEEP_NUMERIC = False
REMOVE_STOPWORDS = True        # main output
INCLUDE_STOPWORDS_VARIANT = True
MIN_TOKENS = 200
SEED = 42

random.seed(SEED)

# Ensure required NLTK data is present (harmless if already installed)
download("brown")
download("punkt")
download("stopwords")

STOPWORDS = set(stopwords.words("english"))
PUNCTUATION = set(string.punctuation)


def ensure_dirs():
    """Create the processed data directory if it does not already exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_token(token, remove_stopwords=True):
    """
    Normalize and filter an individual token.

    Args:
        token: Raw token string from the Brown corpus.
        remove_stopwords: Whether to drop tokens that appear in STOPWORDS.
    Returns:
        A cleaned token string or None when the token should be discarded.
    """
    if not KEEP_NUMERIC and token.isnumeric():
        return None
    if token in PUNCTUATION:
        return None
    token = token.lower() if LOWERCASE else token
    if remove_stopwords and token in STOPWORDS:
        return None
    if token.strip() == "":
        return None
    return token


def process_document(fileid, category, remove_stopwords=True):
    """
    Build the cleaned representation for a single Brown document.

    Args:
        fileid: Identifier returned by nltk.corpus.brown.fileids.
        category: Genre/category for the document.
        remove_stopwords: Whether to drop stopwords during cleaning.
    Returns:
        Dictionary with document id, genre, token count, and cleaned tokens.
    """
    raw_tokens = brown.words(fileids=[fileid])
    cleaned = []
    for t in raw_tokens:
        ct = clean_token(t, remove_stopwords)
        if ct is not None:
            cleaned.append(ct)
    return {
        "id": fileid,
        "genre": category,
        "n_tokens": len(cleaned),
        "tokens": cleaned,
    }


def preprocess(remove_stopwords=True):
    """
    Generate cleaned documents plus supporting indices for one variant.

    Args:
        remove_stopwords: When True, drop stopwords; otherwise keep them.
    Returns:
        Metadata about the generated files and corpus statistics.
    """
    suffix = "clean" if remove_stopwords else "nostop"
    out_path = os.path.join(OUTPUT_DIR, f"brown_{suffix}.jsonl")
    term_index = {}
    term_freqs = Counter()
    vocab = set()
    total_tokens = 0
    n_docs = 0

    fileids = list(brown.fileids())
    random.shuffle(fileids)  # deterministic due to seeded RNG
    genres = {fid: brown.categories(fid)[0] for fid in fileids}

    with open(out_path, "w") as out_f:
        for fid in fileids:
            doc = process_document(fid, genres[fid], remove_stopwords)
            if doc["n_tokens"] < MIN_TOKENS:
                continue
            json.dump(doc, out_f)
            out_f.write("\n")

            n_docs += 1
            total_tokens += doc["n_tokens"]
            vocab.update(doc["tokens"])

            for tok in set(doc["tokens"]):
                term_index.setdefault(tok, []).append(doc["id"])
            term_freqs.update(doc["tokens"])

    # write supporting files
    idx_path = os.path.join(OUTPUT_DIR, f"term_index_{suffix}.json")
    freq_path = os.path.join(OUTPUT_DIR, f"term_freqs_{suffix}.json")
    with open(idx_path, "w") as f:
        json.dump(term_index, f, indent=2)
    with open(freq_path, "w") as f:
        json.dump(dict(term_freqs), f, indent=2)

    return {
        "out_path": out_path,
        "idx_path": idx_path,
        "freq_path": freq_path,
        "n_docs": n_docs,
        "total_tokens": total_tokens,
        "vocab_size": len(vocab),
    }


def main():
    """Run preprocessing for the configured variants and log summaries."""
    ensure_dirs()
    # main (stopwords removed)
    res_clean = preprocess(remove_stopwords=REMOVE_STOPWORDS)

    print(f"Wrote {res_clean['out_path']} ({res_clean['n_docs']} docs, vocab {res_clean['vocab_size']})")

    if INCLUDE_STOPWORDS_VARIANT:
        res_ns = preprocess(remove_stopwords=not REMOVE_STOPWORDS)
        print(f"Wrote {res_ns['out_path']} ({res_ns['n_docs']} docs, vocab {res_ns['vocab_size']})")


if __name__ == "__main__":
    main()
