# scripts/show_top_purity_terms.py
import pandas as pd
from pathlib import Path
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 25
path = Path("data/results/purity_baseline_clean.csv")
df = pd.read_csv(path)

cols = ["term", "G_true", "n_docs_total", "freq_total"]
df = df.sort_values("G_true", ascending=False)
print(df[cols].head(N).to_string(index=False))


