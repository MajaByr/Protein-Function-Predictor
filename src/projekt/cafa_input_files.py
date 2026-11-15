from pathlib import Path

import numpy as np
import pandas as pd
from cafaeval.parser import gt_parser, obo_parser


class CafaInputFiles:
    """Creates the input files CAFA-evaluator needs and remembers where they are.

    Files (all inside work_dir):
        ground_truth.tsv          '<protein>\\t<GO term>' per line, true annotations (e.g. of the test split)
        information_accretion.tsv '<GO term>\\t<weight>' per line, needed only for S-min
        predictions/<name>.tsv    '<protein>\\t<GO term>\\t<score>' per line, no header
    """

    def __init__(self, obo_file: str, work_dir: str):
        self.obo_file = Path(obo_file)
        self.work_dir = Path(work_dir)
        self.ground_truth_file = self.work_dir / "ground_truth.tsv"
        self.information_accretion_file = self.work_dir / "information_accretion.tsv"
        self.predictions_dir = self.work_dir / "predictions"

    def write_ground_truth(self, split_tsv: str) -> Path:
        return self._write_annotations(split_tsv, self.ground_truth_file)

    def write_predictions(self, predictions_csv: str, chunk_rows: int = 1_000_000) -> Path:
        """Converts predictions.csv (ProteinID,GOterm,Score with header) to a tab-separated file without header."""
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.predictions_dir / f"{Path(predictions_csv).stem}.tsv"
        output_file.unlink(missing_ok=True) # we append below, so start from an empty file

        # read in chunks to avoid memory overload
        for chunk in pd.read_csv(predictions_csv, chunksize=chunk_rows, dtype={"Score": float}):
            chunk.to_csv(output_file, sep="\t", header=False, index=False, mode="a")
        return output_file

    def write_information_accretion(self, train_tsv: str) -> Path:
        """Information accretion (IA) of a GO term = how surprising the term is, given that all its parents are known.

        IA(t) = -log2 P(t | all parents of t), estimated on the training set.
        Rare, specific terms get a high IA. It weights the errors in S-min, so S-min cannot be computed without it.
        """
        train_ground_truth_file = self._write_annotations(train_tsv, self.work_dir / "ground_truth_train.tsv")

        # binary matrices (proteins x terms) of the training annotations, one per ontology
        ontologies = obo_parser(str(self.obo_file), ("is_a", "part_of"))
        train_annotations = gt_parser(str(train_ground_truth_file), ontologies)  # already propagated to the root

        lines = []
        for namespace, annotations in train_annotations.items():
            ontology = ontologies[namespace]
            for term_id, term in ontology.terms_dict.items():
                term_count = annotations.matrix[:, term["index"]].sum()
                parent_indices = np.nonzero(ontology.dag[term["index"]])[0]
                if term_count == 0 or len(parent_indices) == 0:
                    continue  # term unseen in training, or a root term: no IA, CAFA-evaluator treats it as 0

                # Proteins annotated with every parent of the term (the term itself is always one of them)
                parents_count = annotations.matrix[:, parent_indices].all(axis=1).sum()
                lines.append(f"{term_id}\t{np.log2(parents_count / term_count)}")

        self.information_accretion_file.write_text("\n".join(lines) + "\n")
        return self.information_accretion_file

    def _write_annotations(self, split_tsv: str, output_file: Path) -> Path:
        """Turns a split table (column go_ids = 'GO:1;GO:2') into one '<protein>\\t<GO term>' line per annotation."""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        split = pd.read_csv(split_tsv, sep="\t", dtype=str)
        annotations = split.assign(go_id=split["go_ids"].str.split(";")).explode("go_id")
        annotations[["entry", "go_id"]].to_csv(output_file, sep="\t", header=False, index=False)
        return output_file
