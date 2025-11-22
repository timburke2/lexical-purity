"""
Construct a recursive lexical-purity decision tree over the Brown corpus.

Inputs:
    data/processed/brown_<variant>.jsonl and term_index_<variant>.json emitted
    by scripts/1_preprocessing.py for the requested variant.
Outputs:
    data/results/purity_tree_<variant>.json capturing entropy, split terms, and
    the top-G contributors at each node.
Usage:
    python scripts/6_recursive_purity_tree.py --variant clean --max_depth 5 --top_k 10
"""

import os
import json
import argparse
from collections import Counter

import numpy as np

# ---------------------
# Configuration
# ---------------------
DATA_DIR = "data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------
# Entropy & purity (copied from 2_compute_baseline_purity.py)
# ---------------------
def corpus_entropy(docs):
    """Compute Shannon entropy (bits) over token distribution for a list of docs."""
    total = sum(len(d["tokens"]) for d in docs)
    if total == 0:
        return 0.0
    freq = Counter(tok for d in docs for tok in d["tokens"])
    probs = np.array(list(freq.values()), dtype=float) / total
    return -np.sum(probs * np.log2(probs))


def purity_gain(term, term_index, docs, H_C):
    """
    Return (G_true, H_Ct, H_Cnot, n_docs_t, n_docs_not, H_weighted) for a given subset S (= docs).

    term_index is always the global mapping term -> doc_ids over the full corpus;
    we restrict to the subset S by filtering docs against doc_ids in term_index[term].
    """
    docset_t = set(term_index.get(term, []))
    # Restrict to current subset S
    docs_t = [d for d in docs if d["id"] in docset_t]
    docs_nt = [d for d in docs if d["id"] not in docset_t]
    if not docs_t or not docs_nt:
        return 0.0, 0.0, 0.0, 0, 0, H_C

    P_t = len(docs_t) / len(docs)
    P_nt = 1 - P_t
    H_t = corpus_entropy(docs_t)
    H_nt = corpus_entropy(docs_nt)
    H_weighted = P_t * H_t + P_nt * H_nt
    G = H_C - H_weighted
    return G, H_t, H_nt, len(docs_t), len(docs_nt), H_weighted


# ---------------------
# Recursive tree builder
# ---------------------
def build_purity_tree(docs, term_index, depth, max_depth, top_k, path):
    """
    Recursively build a purity tree for the given subset of docs.

    Args:
      docs      : list[doc_dict] for the current node
      term_index: global term -> [doc_ids] mapping
      depth     : current depth (0 at root)
      max_depth : maximum depth to recurse
      top_k     : how many top-G terms to record per node
      path      : list of {"term": str, "has": bool} from root to this node

    Returns:
      node_dict representing this node and its children.
    """
    print(f"[depth {depth}] Starting node with {len(docs)} docs, path length={len(path)}")

    n_docs = len(docs)
    H_S = corpus_entropy(docs)

    node = {
        "depth": depth,
        "path": path,
        "n_docs": n_docs,
        "entropy": float(H_S),
        "split_term": None,
        "split_G": 0.0,
        "top_terms": [],
        "left": None,
        "right": None,
    }

    # Stopping conditions
    if depth >= max_depth or n_docs <= 1:
        return node

    # Collect candidate terms: those that vary within this subset
    doc_ids_in_S = {d["id"] for d in docs}
    candidates = []
    for term, docids in term_index.items():
        # Intersect docids with current subset
        docset_t_in_S = doc_ids_in_S.intersection(docids)
        n_t = len(docset_t_in_S)
        if n_t == 0 or n_t == n_docs:
            # term is either absent or present in all docs in S -> no split
            continue
        candidates.append(term)

    if not candidates:
        # No term induces a non-trivial split in this subset
        return node

    # Compute purity gain for all candidate terms
    term_scores = []
    for term in candidates:
        G, H_t, H_nt, n_docs_t, n_docs_nt, H_weighted = purity_gain(
            term, term_index, docs, H_S
        )
        if G <= 0:
            continue
        term_scores.append(
            {
                "term": term,
                "G_true": float(G),
                "H_weighted": float(H_weighted),
                "H_Ct": float(H_t),
                "H_Cnot": float(H_nt),
                "n_docs_t": int(n_docs_t),
                "n_docs_not": int(n_docs_nt),
            }
        )

    if not term_scores:
        # No term actually improves purity in this subset
        return node

    # Sort by G descending
    term_scores.sort(key=lambda r: r["G_true"], reverse=True)

    # Top splitter and top_k terms
    best = term_scores[0]
    node["split_term"] = best["term"]
    node["split_G"] = best["G_true"]
    node["top_terms"] = term_scores[:top_k]
    print(f"[depth {depth}] Split on '{best['term']}' (G={best['G_true']:.4f}), children: "
      f"{best['n_docs_t']} / {best['n_docs_not']}")


    # If we can't split meaningfully (e.g. best G is extremely tiny), we still record the node
    # and stop recursion naturally when children degenerate.
    split_term = best["term"]
    docset_split = set(term_index[split_term])

    # Define children:
    # left  = docs that HAVE split_term
    # right = docs that DO NOT have split_term
    docs_left = [d for d in docs if d["id"] in docset_split]
    docs_right = [d for d in docs if d["id"] not in docset_split]

    # If either side is empty, we stop (shouldn't happen if candidates were filtered correctly)
    if not docs_left or not docs_right:
        return node

    # Recurse on children
    left_path = path + [{"term": split_term, "has": True}]
    right_path = path + [{"term": split_term, "has": False}]

    node["left"] = build_purity_tree(
        docs_left,
        term_index,
        depth=depth + 1,
        max_depth=max_depth,
        top_k=top_k,
        path=left_path,
    )
    node["right"] = build_purity_tree(
        docs_right,
        term_index,
        depth=depth + 1,
        max_depth=max_depth,
        top_k=top_k,
        path=right_path,
    )

    return node


# ---------------------
# Main
# ---------------------
def main():
    """Load corpus artifacts, build the purity tree, and write the JSON output."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        type=str,
        default="clean",
        choices=["clean", "nostop"],
        help="Which preprocessed Brown variant to use (default: clean)",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=5,
        help="Maximum recursion depth (default: 5)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top-G terms to record per node (default: 10)",
    )
    args = parser.parse_args()

    variant = args.variant
    max_depth = args.max_depth
    top_k = args.top_k

    proc_prefix = f"brown_{variant}"
    docs_path = os.path.join(PROCESSED_DIR, f"{proc_prefix}.jsonl")
    idx_path = os.path.join(PROCESSED_DIR, f"term_index_{variant}.json")

    print(f"Loading docs from {docs_path} ...")
    with open(docs_path) as f:
        docs = [json.loads(l) for l in f]
    print(f"Loaded {len(docs)} documents.")

    print(f"Loading term index from {idx_path} ...")
    with open(idx_path) as f:
        term_index = json.load(f)
    print(f"Loaded term index with {len(term_index)} terms.")

    print(f"\nBuilding purity tree (variant={variant}, max_depth={max_depth}, top_k={top_k}) ...")
    root = build_purity_tree(
        docs=docs,
        term_index=term_index,
        depth=0,
        max_depth=max_depth,
        top_k=top_k,
        path=[],
    )

    out_path = os.path.join(RESULTS_DIR, f"purity_tree_{variant}.json")
    with open(out_path, "w") as f:
        json.dump(root, f, indent=2)

    print(f"\nDone. Tree written to {out_path}")


if __name__ == "__main__":
    main()
