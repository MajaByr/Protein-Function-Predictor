from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from cafaeval.evaluation import cafa_eval, write_results

from projekt.cafa_input_files import CafaInputFiles

# Namespaces as named in the OBO file corresponding to short names
ONTOLOGY_NAMES = {
    "biological_process": "BP",
    "molecular_function": "MF",
    "cellular_component": "CC",
}


class CafaEvaluator:
    """Runs CAFA-evaluator on prepared input files and reports F-max, S-min and the precision-recall curves.

    BP, MF and CC are evaluated separately (each has its own rows in `results`).

    Typical use:
        input_files = CafaInputFiles("data/go-basic.obo", "evaluation")
        input_files.write_ground_truth("data/splits/test.tsv")
        input_files.write_information_accretion("data/splits/train.tsv")
        input_files.write_predictions("predictions.csv")

        evaluator = CafaEvaluator(input_files)
        evaluator.evaluate()
        evaluator.summary()
        evaluator.save_results()
    """

    def __init__(self, input_files: CafaInputFiles):
        self.input_files = input_files
        self.threshold_step = 0.01
        self.results = None # one row per (prediction file, ontology, threshold)
        self.best_results = None # best threshold per (prediction file, ontology) for each metric

    def evaluate(self, threshold_step: float = 0.01, n_cpu: int = 4) -> pd.DataFrame:
        """Evaluates every file in the predictions directory. Slow for big prediction files."""
        files = self.input_files

        # without information accretion CAFA-evaluator still works, but only gives unweighted S
        information_accretion = files.information_accretion_file if files.information_accretion_file.exists() else None

        self.threshold_step = threshold_step
        self.results, self.best_results = cafa_eval(
            str(files.obo_file), str(files.predictions_dir), str(files.ground_truth_file),
            ia=information_accretion, th_step=threshold_step, n_cpu=n_cpu,
        )
        return self.results

    def summary(self) -> pd.DataFrame:
        """F-max (best F1 over thresholds) and S-min (lowest S over thresholds) per ontology."""
        # s weighted by information accretion is the standard S-min; without IA only the unweighted S exists
        s_column = "s_w" if "s_w" in self.results.columns else "s"

        # for each (file, ontology) take the row (= threshold) with the highest F1 / the lowest S
        per_ontology = self.results.groupby(["filename", "ns"])
        f_max_rows = self.results.loc[per_ontology["f"].idxmax()]
        s_min_rows = self.results.loc[per_ontology[s_column].idxmin()]

        summary = pd.DataFrame({
            "F-max": f_max_rows["f"],
            "F-max threshold": f_max_rows.index.get_level_values("tau"),
            "S-min": s_min_rows[s_column],
            "S-min threshold": s_min_rows.index.get_level_values("tau"),
        }).droplevel("tau")

        # BP / MF / CC instead of the long namespace names
        summary.index = summary.index.set_levels(
            [ONTOLOGY_NAMES.get(ns, ns) for ns in summary.index.levels[1]], level="ns")
        return summary

    def curves(self) -> pd.DataFrame:
        """Precision, recall and F1 for every threshold (F-max is the maximum of the f column)."""
        return self.results[["pr", "rc", "f"]]

    def plot_curves(self):
        """Precision-recall curve plus precision, recall and F1 against the threshold (first prediction file).

        One line per ontology. F-max is the highest point of the F1 line.
        """
        filename = self.results.index.get_level_values("filename")[0]
        curves = self.curves().loc[filename]

        figure, axes = plt.subplots(2, 2, figsize=(11, 9))
        pr_axes, precision_axes, recall_axes, f_axes = axes.flatten()
        for namespace, ontology_curve in curves.groupby(level="ns"):
            name = ONTOLOGY_NAMES.get(namespace, namespace)
            thresholds = ontology_curve.index.get_level_values("tau")
            pr_axes.plot(ontology_curve["rc"], ontology_curve["pr"], label=name)
            precision_axes.plot(thresholds, ontology_curve["pr"], label=name)
            recall_axes.plot(thresholds, ontology_curve["rc"], label=name)
            f_axes.plot(thresholds, ontology_curve["f"], label=name)

        pr_axes.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
        precision_axes.set(xlabel="Threshold", ylabel="Precision", title="Precision vs threshold")
        recall_axes.set(xlabel="Threshold", ylabel="Recall", title="Recall vs threshold")
        f_axes.set(xlabel="Threshold", ylabel="F1", title="F1 vs threshold")
        for single_axes in axes.flatten():
            single_axes.set(xlim=(0, 1), ylim=(0, 1))
            single_axes.legend()
            single_axes.grid(alpha=0.3)
        figure.tight_layout()

        # Close it in pyplot, so Jupyter shows the returned figure once instead of also auto-displaying the open one
        plt.close(figure)
        return figure

    def save_results(self, out_dir: str | None = None) -> Path:
        """Saves per-threshold table, best-threshold tables, summary (F-max, S-min) and the curves plot."""
        out_dir = Path(out_dir) if out_dir else self.input_files.work_dir / "results"

        # Tables written by CAFA-evaluator itself: evaluation_all.tsv (every threshold) and evaluation_best_*.tsv
        write_results(self.results, self.best_results, str(out_dir), self.threshold_step)
        self.summary().to_csv(out_dir / "summary.tsv", sep="\t")
        self.plot_curves().savefig(out_dir / "curves.png", dpi=150)
        return out_dir
