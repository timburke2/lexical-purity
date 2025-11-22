"""
Study the spectrum of the term-term Jaccard similarity matrix.

Inputs:
    data/results/term_term_jaccard_<variant>.csv produced by
    scripts/4_term_term_jaccard.py, or a manually supplied --in_csv path.
Outputs:
    data/results/term_term_jaccard_<variant>_spectrum.csv and
    data/results/term_term_jaccard_<variant>_eigvec1_loadings.csv describing
    the eigenvalues plus the leading eigenvector loadings.
Usage:
    python scripts/5_jaccard_spectrum.py --variant clean --tag g0.20
"""

import os
import argparse
import numpy as np
import pandas as pd

DATA_DIR      = "data"
RESULTS_DIR   = os.path.join(DATA_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_jaccard_table(path: str) -> pd.DataFrame:
    """
    Load the long-form Jaccard CSV and validate required columns exist.

    Args:
        path: File path to the CSV exported by script 4.
    Returns:
        Pandas DataFrame containing at least term_i, term_j, and jaccard columns.
    """
    df = pd.read_csv(path)
    required_cols = {"term_i", "term_j", "jaccard"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    return df


def build_similarity_matrix(df: pd.DataFrame):
    """
    Convert the long-form table into a dense symmetric similarity matrix.

    Args:
        df: DataFrame returned by load_jaccard_table.
    Returns:
        Tuple of (matrix, ordered_terms) where matrix[i, j] holds Jaccard scores.
    """
    terms = sorted(set(df["term_i"]).union(set(df["term_j"])))
    n = len(terms)
    term_to_idx = {t: i for i, t in enumerate(terms)}

    S = np.eye(n, dtype=float)
    for _, row in df.iterrows():
        i = term_to_idx[row["term_i"]]
        j = term_to_idx[row["term_j"]]
        jacc = float(row["jaccard"])
        S[i, j] = jacc
        S[j, i] = jacc

    return S, terms


def compute_spectrum(S: np.ndarray):
    """
    Compute and sort the eigen decomposition of the symmetric similarity matrix.

    Args:
        S: Symmetric numpy.ndarray built by build_similarity_matrix.
    Returns:
        eigvals_sorted, eigvecs_sorted ordered from largest to smallest eigenvalue.
    """
    eigvals_full, eigvecs_full = np.linalg.eigh(S)

    # Sort descending
    idx = np.argsort(eigvals_full)[::-1]
    eigvals_sorted = eigvals_full[idx]
    eigvecs_sorted = eigvecs_full[:, idx]
    return eigvals_sorted, eigvecs_sorted


def main():
    """Load the similarity data, compute the spectrum, and persist summaries."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_csv",
        type=str,
        default=None,
        help=(
            "Input Jaccard CSV (default: data/results/term_term_jaccard_<variant>.csv)"
        ),
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="clean",
        help="Variant label used in filenames if --in_csv is not given (default: clean)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional tag to append to output filenames (e.g., 'g0.20')",
    )
    args = parser.parse_args()

    if args.in_csv is None:
        in_csv = os.path.join(RESULTS_DIR, f"term_term_jaccard_{args.variant}.csv")
    else:
        in_csv = args.in_csv

    print(f"Loading Jaccard table from {in_csv} ...")
    df_j = load_jaccard_table(in_csv)
    print(f"Loaded {len(df_j):,} term-term pairs.")

    print("Building similarity matrix ...")
    S, terms = build_similarity_matrix(df_j)
    n = len(terms)
    print(f"Similarity matrix shape: {n} × {n}")

    print("Computing eigenvalues/eigenvectors ...")
    eigvals, eigvecs = compute_spectrum(S)
    trace = eigvals.sum()
    fractions = eigvals / trace
    cum_fractions = np.cumsum(fractions)

    # ---- Console summary ----
    top_k = min(10, n)
    print("\n=== Top eigenvalues of Jaccard similarity ===")
    for k in range(top_k):
        print(
            f"  λ{k+1:2d} = {eigvals[k]:8.4f}  "
            f"({fractions[k]*100:5.2f}% of total, cumulative {cum_fractions[k]*100:5.2f}%)"
        )

    # ---- Prepare outputs ----
    tag_str = f"_{args.tag}" if args.tag else ""
    base = os.path.splitext(os.path.basename(in_csv))[0]

    out_eigvals_csv = os.path.join(
        RESULTS_DIR, f"{base}_spectrum{tag_str}.csv"
    )
    out_eigvec1_csv = os.path.join(
        RESULTS_DIR, f"{base}_eigvec1_loadings{tag_str}.csv"
    )

    # Save eigenvalues + fractions
    df_eig = pd.DataFrame(
        {
            "rank": np.arange(1, n + 1),
            "eigenvalue": eigvals,
            "fraction": fractions,
            "cum_fraction": cum_fractions,
        }
    )
    df_eig.to_csv(out_eigvals_csv, index=False)
    print(f"\nSaved eigenvalue spectrum to {out_eigvals_csv}")

    # Save top eigenvector term loadings
    eigvec1 = eigvecs[:, 0]
    df_vec1 = pd.DataFrame(
        {
            "term": terms,
            "loading": eigvec1,
        }
    ).sort_values("loading", ascending=False)
    df_vec1.to_csv(out_eigvec1_csv, index=False)
    print(f"Saved top eigenvector loadings to {out_eigvec1_csv}")


if __name__ == "__main__":
    main()
