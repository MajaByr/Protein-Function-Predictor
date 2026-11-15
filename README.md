# :dna: Protein Function Predictor (PFP) :dna:

*End-to-end framework that predicts protein function (in the form of Gene Ontology term probabilities) from an amino acid protein sequence.*

---

## How to run

### Requirements

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- [Docker](https://www.docker.com/) (for MMseqs2 clustering)

All input data (GAF and UniProt TSV files) are already in the repository – no manual download needed.
The only exception is the GO ontology file for the evaluation (`data/go-basic.obo`, step 6).

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/MajaByr/Protein-Predictor.git
cd Protein-Predictor

# 2. Install the Python environment (creates .venv from uv.lock)
uv sync

# 3. Merge GAF and UniProt data, keep only GO terms with high-quality evidence
uv run python -m projekt.merge

# 4. Cluster sequences with MMseqs2 (min-seq-id 0.5, coverage 0.8, cov-mode 0)
#    run step by step inside a Docker container – see docs/02_TRAIN_TEST.md

# 5. Assign whole clusters to train/eval/test at 9:1:1 ratio
uv run python -m projekt.train_test_split

# 6. Build the GO term vocabulary (from the train split only) – defines the model output vector
uv run python src/projekt/build_vocab.py

# 7. Download the GO ontology (OBO) used by the evaluation
#    (the repo's results were made with data-version releases/2026-07-26, same as the GAF files)
curl -L -o data/go-basic.obo https://purl.obolibrary.org/obo/go/go-basic.obo

# 8. Predict GO terms for the whole test split
#    (currently random scores from a dummy model – tests the pipeline, not a real model)
#    creates data/splits/test.fasta and predictions.csv (~750 MB, ~1.5 min)
uv run python src/projekt/pipeline.py

# 9. Evaluate the predictions with CAFA-evaluator (~7 min) – see "Evaluation" below
```

### Output

| File | Created by | Content |
|------|-----------|---------|
| `data/processed/master.tsv` | step 3 | one protein per row: `entry`, `organism`, `go_ids` (labels), `sequence` |
| `data/processed/proteins.fasta` | step 3 | sequences of the proteins in `master.tsv` |
| `data/processed/go_comparison.tsv` | step 3 | per organism: GO terms kept vs dropped per protein |
| `data/clusters/clusters_cluster.tsv` | step 4 | MMseqs2 clusters: representative, member |
| `data/splits/{train,eval,test}.tsv` | step 5 | `master.tsv` rows split by cluster, plus `cluster` column |
| `data/vocab/{BP,MF,CC}.txt` | step 6 | GO terms seen in train, one per line, per ontology |
| `data/go-basic.obo` | step 7 | GO ontology (term hierarchy) |
| `data/splits/test.fasta` | step 8 | sequences of the test split (git-ignored) |
| `predictions.csv` | step 8 | `ProteinID,GOterm,Score` – every test protein × every GO term (git-ignored) |
| `evaluation/` | step 9 | CAFA-evaluator inputs and results (git-ignored), see below |

Detailed description (in Polish): [docs/01_DATA.md](docs/01_DATA.md), [docs/02_TRAIN_TEST.md](docs/02_TRAIN_TEST.md).

## Evaluation

Predictions are evaluated with [CAFA-evaluator](https://github.com/BioComputingUP/CAFA-evaluator) separately for BP, MF and CC. Two classes do the work:

| Class | File | Role |
|-------|------|------|
| `CafaInputFiles` | `src/projekt/cafa_input_files.py` | writes the files CAFA-evaluator needs: ground truth (from `test.tsv`), predictions (converted from `predictions.csv`), information accretion (GO term weights for S-min, from `train.tsv`) |
| `CafaEvaluator` | `src/projekt/cafa_evaluator.py` | runs the evaluation; gives F-max, S-min, precision/recall/F1 per threshold, curves plot, saves results |

Run the example notebook [evaluation/evaluation_example.ipynb](evaluation/evaluation_example.ipynb) (kernel: the project's `.venv`), or the same steps in Python:

```python
from projekt.cafa_evaluator import CafaEvaluator
from projekt.cafa_input_files import CafaInputFiles

input_files = CafaInputFiles(obo_file="data/go-basic.obo", work_dir="evaluation")
input_files.write_ground_truth("data/splits/test.tsv")
input_files.write_information_accretion("data/splits/train.tsv")
input_files.write_predictions("predictions.csv")

evaluator = CafaEvaluator(input_files)
evaluator.evaluate()          # slow: parses ~30M prediction rows
print(evaluator.summary())    # F-max and S-min per ontology
evaluator.save_results()      # -> evaluation/results/
```

Results in `evaluation/results/`:

| File | Content |
|------|---------|
| `summary.tsv` | F-max and S-min per ontology, with the thresholds where they occur |
| `evaluation_all.tsv` | precision, recall, F1, S for every threshold (the data behind the curves) |
| `evaluation_best_*.tsv` | best threshold per ontology for each metric (CAFA-evaluator output) |
| `curves.png` | precision-recall curve and F1 vs threshold, one line per ontology |

To evaluate a real model, write its predictions in the same `ProteinID,GOterm,Score` CSV format and pass the file to `write_predictions`.

## Data sources and versions

### GO annotations (GAF, UniProt-centric)

Source: https://geneontology.org/docs/download-go-annotations/downloads/

Stored gzip-compressed (`gzip -9`) to stay under GitHub's 100 MB file limit; pandas reads them directly.

| Organism  | File                     | Taxon ID | Generated        | GO release |
|-----------|--------------------------|----------|------------------|------------|
| Human     | `HUMAN-uniprot.gaf.gz`   | 9606     | 2026-07-28 19:42 | 2026-07-26 |
| Zebrafish | `DANRE-uniprot.gaf.gz`   | 7955     | 2026-07-28 19:38 | 2026-07-26 |
| Rat       | `RAT-uniprot.gaf.gz`     | 10116    | 2026-07-28 19:51 | 2026-07-26 |

### UniProt ID-to-sequence mappings

Source: https://www.uniprot.org/ — UniProtKB release **2026_03** (released 2026-09-02), accessed 2026-10-16.

Query: `(organism_id:<taxon>) AND (reviewed:true)` (Swiss-Prot only), columns: Entry, Gene Ontology IDs, Sequence, format TSV.

| Organism  | Taxon ID | Swiss-Prot entries |
|-----------|----------|--------------------|
| Human     | 9606     | 20431              |
| Zebrafish | 7955     | 3385               |
| Rat       | 10116    | 8233               |

### GO ontology (OBO)

Source: https://geneontology.org/docs/download-ontology/ (`go-basic.obo`, downloaded with the link in step 7). Used to propagate annotations to parent terms and to split terms into BP / MF / CC during evaluation. The file in the repo's results has `data-version: releases/2026-07-26`; step 7 downloads the *current* release, so scores may differ slightly.
