# Lexical Purity Playground

This project explores **lexical purity** in the NLTK Brown corpus. A sequence of
Python scripts cleans the corpus, computes per-term entropy reductions
(`G_true`), compares those terms with topic models, and analyzes their
co-occurrence structure.

## Repository layout

```
lexical_purity/
├── data/
│   ├── processed/   # JSONL docs, term indices, term-frequency maps
│   ├── results/     # Purity metrics, Jaccard tables, spectra
│   └── visuals/     # Graphviz DOT files and rendered trees
├── scripts/        # Numbered pipeline stages (1–7) plus utilities
└── README.md
```

The scripts build on one another; run them in order unless you already have the
intermediate artifacts.

## Environment setup

1. Python 3.9+ with the following packages:
   ```bash
   pip install nltk numpy pandas matplotlib scikit-learn tqdm
   ```
2. The preprocessing step will download the Brown corpus, punkt tokenizer, and
   stopword lists the first time it runs.

## Pipeline overview

| Step | Script | Purpose | Key outputs |
| ---- | ------ | ------- | ----------- |
| 1 | `scripts/1_preprocessing.py` | Clean Brown documents (stopwords optional), build term indices and frequency maps. | `data/processed/brown_<variant>.jsonl`, `term_index_<variant>.json`, `term_freqs_<variant>.json` |
| 2 | `scripts/2_compute_baseline_purity.py` | Compute purity gain `G_true` per term plus histogram. | `data/results/purity_baseline_<variant>.csv`, metadata JSON, distribution PNG |
| 3 | `scripts/3_compare_LDA.py` | Vectorize corpus using purity vocabulary, fit LDA, compare top topic words with top `G_true` terms. | Console report, optional CSVs (`lda_k2_topics.csv`, `lda_k2_lp_assignment.csv`) |
| 4 | `scripts/4_term_term_jaccard.py` | Build document sets for high-purity terms and measure pairwise Jaccard overlaps. | `data/results/term_term_jaccard_<variant>.csv` |
| 5 | `scripts/5_jaccard_spectrum.py` | Treat the Jaccard table as a similarity matrix and study its eigen spectrum and leading eigenvector loadings. | `<base>_spectrum.csv`, `<base>_eigvec1_loadings.csv` |
| 6 | `scripts/6_recursive_purity_tree.py` | Recursively split documents on the best `G_true` term to visualize hierarchical purity structure. | `data/results/purity_tree_<variant>.json` |
| 7 | `scripts/7_visualize_purity_tree.py` | Convert the saved purity tree into a Graphviz DOT file for rendering. | `data/visuals/purity_tree_<variant>.dot` (then render to PNG/SVG via `dot`) |

> **Variants:** `clean` removes stopwords; `nostop` retains them. Step 1 can
> generate both variants; subsequent steps take a `--variant` flag.

## Usage snippets

Run each script from the project root (`lexical_purity/`):

```bash
# Step 1: preprocessing (creates clean + nostop versions by default)
python scripts/1_preprocessing.py

# Step 2: baseline purity for the clean corpus
python scripts/2_compute_baseline_purity.py --variant clean

# Step 3: compare LDA topics to top purity terms
python scripts/3_compare_LDA.py --n_topics 2 --n_top_words 40 --save_csv

# Step 4: Jaccard overlaps for terms with G_true >= 0.15
python scripts/4_term_term_jaccard.py --variant clean --g_thresh 0.15

# Step 5: eigen spectrum of the Jaccard matrix
python scripts/5_jaccard_spectrum.py --variant clean --tag g0.15

# Step 6: purity decision tree capped at depth 5
python scripts/6_recursive_purity_tree.py --variant clean --max_depth 5 --top_k 10

# Step 7: Graphviz visualization of the purity tree (requires Graphviz)
python scripts/7_visualize_purity_tree.py --variant clean --max_depth 5
dot -Tpng data/visuals/purity_tree_clean.dot -o data/visuals/purity_tree_clean.png
```

## Notes and tips

- The `data/` directories are created on demand. Ensure the repository is
  writable before running the scripts.
- Each step logs its progress to the console; check the printed paths for the
  generated files.
- Larger values of `MIN_DOCS_PER_TERM`, `g_thresh`, or `max_depth` change the
  granularity of the analysis—tune them based on the questions you want to ask.
