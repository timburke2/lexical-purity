"""
Measure pairwise Jaccard similarity for the high-purity Brown corpus terms.

Inputs:
    data/results/purity_baseline_<variant>.csv and the associated
    data/processed/term_index_<variant>.json files derived from prior scripts.
Outputs:
    data/results/term_term_jaccard_<variant>.csv (or a custom --out_csv path)
    summarizing every eligible term pair and its overlap statistics.
Usage:
    python scripts/4_term_term_jaccard.py --variant clean --g_thresh 0.15
"""


import os
import json
import argparse
from itertools import combinations

import numpy as np
import pandas as pd

DATA_DIR      = "data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR   = os.path.join(DATA_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_term_index(variant: str):
    """
    Load the mapping of term -> document ids for one preprocessing variant.

    Args:
        variant: "clean" or "nostop", aligning with file suffixes.
    Returns:
        Dict mapping term strings to the list of document ids they appear in.
    """
    idx_path = os.path.join(PROCESSED_DIR, f"term_index_{variant}.json")
    with open(idx_path, "r") as f:
        term_index = json.load(f)
    return term_index


def load_purity(variant: str):
    """
    Read the baseline purity CSV for the requested variant.

    Args:
        variant: "clean" or "nostop".
    Returns:
        Pandas DataFrame with columns including 'term' and 'G_true'.
    """
    purity_path = os.path.join(RESULTS_DIR, f"purity_baseline_{variant}.csv")
    return pd.read_csv(purity_path)


def build_docsets_for_terms(term_index, terms):
    """
    Convert selected terms into document-id sets for set math.

    Args:
        term_index: Mapping produced by load_term_index.
        terms: Iterable of terms whose doc sets we want.
    Returns:
        Dict term -> set(doc_id) with missing terms omitted.
    """
    docsets = {}
    for t in terms:
        ids = term_index.get(t)
        if ids is None:
            # term missing from index (should be rare); skip
            continue
        docsets[t] = set(ids)
    return docsets


def compute_jaccard(docsets):
    """
    Enumerate pairwise Jaccard scores between every provided term.

    Args:
        docsets: Dict term -> set of doc_ids.
    Returns:
        List of dict rows with term_i, term_j, jaccard, n_intersect, and n_union.
    """
    terms = sorted(docsets.keys())
    rows = []

    for t_i, t_j in combinations(terms, 2):
        s_i = docsets[t_i]
        s_j = docsets[t_j]
        inter = len(s_i & s_j)
        union = len(s_i | s_j)
        if union == 0:
            j = 0.0
        else:
            j = inter / union
        rows.append(
            {
                "term_i": t_i,
                "term_j": t_j,
                "jaccard": float(j),
                "n_intersect": inter,
                "n_union": union,
            }
        )
    return rows


def main():
    """Parse arguments, compute the similarities, and persist the CSV."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        type=str,
        default="clean",
        choices=["clean", "nostop"],
        help="Which preprocessing variant to use (default: clean)",
    )
    parser.add_argument(
        "--g_thresh",
        type=float,
        default=0.15,
        help="Minimum G_true to include a term (default: 0.15)",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Optional output CSV path (default: data/results/term_term_jaccard_<variant>.csv)",
    )
    args = parser.parse_args()

    print(f"Loading purity baseline for variant='{args.variant}' ...")
    df_purity = load_purity(args.variant)

    # Filter by G_true threshold
    df_high = df_purity[df_purity["G_true"] >= args.g_thresh].copy()
    df_high = df_high.sort_values("G_true", ascending=False)
    terms_high = df_high["term"].tolist()
    print(f"Found {len(terms_high)} terms with G_true >= {args.g_thresh}")

    if not terms_high:
        print("No terms above threshold; nothing to do.")
        return

    print("Loading term index ...")
    term_index = load_term_index(args.variant)

    print("Building document sets for high-G terms ...")
    docsets = build_docsets_for_terms(term_index, terms_high)
    print(f"Have docsets for {len(docsets)} terms (some may be missing from index).")

    print("Computing pairwise Jaccard similarities ...")
    rows = compute_jaccard(docsets)
    if not rows:
        print("No pairwise combinations to compute (need at least 2 terms).")
        return

    df_j = pd.DataFrame(rows)

    # Basic summary
    print("\n=== Jaccard summary for high-G terms ===")
    print(f"Number of term pairs: {len(df_j)}")
    print(f"Jaccard mean:   {df_j['jaccard'].mean():.4f}")
    print(f"Jaccard median: {df_j['jaccard'].median():.4f}")
    print(f"Jaccard min:    {df_j['jaccard'].min():.4f}")
    print(f"Jaccard max:    {df_j['jaccard'].max():.4f}")

    # Save CSV
    out_csv = (
        args.out_csv
        if args.out_csv is not None
        else os.path.join(RESULTS_DIR, f"term_term_jaccard_{args.variant}.csv")
    )
    df_j.to_csv(out_csv, index=False)
    print(f"\nSaved Jaccard table to {out_csv}")


if __name__ == "__main__":
    main()
