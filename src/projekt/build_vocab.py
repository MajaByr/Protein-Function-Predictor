from pathlib import Path

import pandas as pd

MIN_COUNT = 1  # a term must appear in at least this many training proteins

# 1. GO terms from the training set only (so eval/test terms do not leak into the model)
train = pd.read_csv("data/splits/train.tsv", sep="\t")
counts = train["go_ids"].str.split(";").explode().value_counts()
terms = counts[counts >= MIN_COUNT].index

# 2. Ontology of each term from the GAF files (column 4 = GO ID, column 8 = aspect: P, F or C)
aspects = {}
for gaf_file in Path("data/GAF").glob("*.gaf.gz"):
    gaf = pd.read_csv(gaf_file, sep="\t", comment="!", header=None, usecols=[4, 8])
    aspects.update(zip(gaf[4], gaf[8]))

# 3. Split into BP / MF / CC and save (sorted = fixed order, which defines the output vector)
Path("data/vocab").mkdir(parents=True, exist_ok=True)
total = 0
for name, letter in {"BP": "P", "MF": "F", "CC": "C"}.items():
    selected = sorted(t for t in terms if aspects.get(t) == letter)
    Path(f"data/vocab/{name}.txt").write_text("\n".join(selected) + "\n")
    print(name, len(selected))
    total += len(selected)

print("terms without an ontology in GAF (skipped):", len(terms) - total)