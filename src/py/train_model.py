# src/py/train_model.py
from pathlib import Path
from dataclasses import dataclass

import pandas as pd

from src.py.ml.model import fit_model
from src.py.ml.preprocessing import (fit_preprocessing, kmer_feature_cols,
                                     locus_feature_cols, transform)
from src.py.utils.custom_logger import get_logger
from src.py.utils.exceptions import StepError
from src.py.utils.pipeline_step import PipelineStep
from src.py.utils.tools import require_files

logger = get_logger()

@dataclass(frozen=True)
class TrainOutputs:
    model_path: Path
    preprocessing_path: Path
    training_table_path: Path   # labeled feature table, input for the CV harness

class TrainModel(PipelineStep[TrainOutputs]):
    def __init__(self, feature_table_path, labels_path, output_dir, k: int, epochs: int,
                 label_col: str = "H3K4me3", batch_size: int = 64, seed: int = 47):
        self.feature_table_path = Path(feature_table_path)
        self.labels_path = Path(labels_path)
        self.output_dir = Path(output_dir)
        self.k = int(k)
        self.epochs = int(epochs)
        self.label_col = label_col
        self.batch_size = batch_size
        self.seed = seed

    def run(self) -> TrainOutputs:
        require_files(self.feature_table_path, self.labels_path)
        labels = pd.read_table(self.labels_path)
        df = pd.read_table(self.feature_table_path).merge(
            labels, on="name", how="inner", validate="one_to_one")
        if df.empty:
            raise StepError("No labeled transcripts survived filtering")

        # Filtering can remove one class more than the other; make that visible.
        self._log_balance("Labeled input", labels[self.label_col])
        self._log_balance("After filtering", df[self.label_col])

        self.output_dir.mkdir(parents=True, exist_ok=True)
        table_path = self.output_dir / "training_table.tsv"
        df.to_csv(table_path, sep="\t", index=False)

        prep = fit_preprocessing(df, kmer_feature_cols(self.k), locus_feature_cols(df),
                                 self.label_col)
        y = df[self.label_col].to_numpy(dtype="float32")
        logger.info("Training on %d transcripts for %d epochs", len(df), self.epochs)
        model, _ = fit_model(transform(df, prep), y, epochs=self.epochs,
                             batch_size=self.batch_size, seed=self.seed,
                             log_path=self.output_dir / "training_log.csv")

        model_path = self.output_dir / "final_model.keras"
        prep_path = self.output_dir / "preprocessing.json"
        model.save(model_path)
        prep.save(prep_path)
        logger.info("Saved %s and %s", model_path, prep_path)
        return TrainOutputs(model_path, prep_path, table_path)

    def _log_balance(self, label: str, y: pd.Series) -> None:
        n_pos = int(y.sum())
        logger.info("  %s: %d transcripts, %d positive (%.1f%%)",
                    label, len(y), n_pos, 100 * n_pos / len(y))