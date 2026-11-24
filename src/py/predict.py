# src/py/predict.py
from pathlib import Path
from dataclasses import dataclass

import pandas as pd

from src.py.utils.custom_logger import get_logger
from src.py.utils.exceptions import StepError
from src.py.utils.pipeline_step import PipelineStep
from src.py.ml.preprocessing import Preprocessing, transform
from src.py.utils.tools import require_files

logger = get_logger()

# Carried into the output so predictions are readable without a join.
_ID_COLS = ["name", "condition", "transcript_id", "chrom", "chromStart", "chromEnd",
            "strand", "locus_type"]

@dataclass(frozen=True)
class PredictionOutputs:
    predictions_path: Path
    num_transcripts: int
    num_positive: int

class Predict(PipelineStep[PredictionOutputs]):
    def __init__(
        self,
        feature_table_path: str | Path,
        model_path: str | Path,
        preprocessing_path: str | Path,
        output_path: str | Path,
        threshold: float = 0.5,
    ):
        self.feature_table_path = Path(feature_table_path)
        self.model_path = Path(model_path)
        self.preprocessing_path = Path(preprocessing_path)
        self.output_path = Path(output_path)
        self.threshold = float(threshold)

    def run(self) -> PredictionOutputs:
        logger.info("Running prediction")
        
        require_files(self.feature_table_path, self.model_path, self.preprocessing_path)
        # Imported here so the CLI starts quickly and other steps don't load TensorFlow.
        from tensorflow import keras

        df = pd.read_table(self.feature_table_path)
        prep = Preprocessing.load(self.preprocessing_path)
        X = transform(df, prep)

        model = keras.models.load_model(self.model_path)
        if model.input_shape[-1] != X.shape[1]:
            raise StepError(f"Model expects {model.input_shape[-1]} features, "
                            f"preprocessing produced {X.shape[1]}")

        score = model.predict(X, batch_size=1024, verbose=0).ravel()
        label = prep.label_col
        out = df[[c for c in _ID_COLS if c in df.columns]].copy()
        out[f"{label}_prob"] = score
        out[f"{label}_pred"] = (score >= self.threshold).astype(int)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(self.output_path, sep="\t", index=False)
        n_pos = int(out[f"{label}_pred"].sum())
        logger.info("Predicted %s for %d transcripts: %d positive at threshold %g -> %s",
                    label, len(out), n_pos, self.threshold, self.output_path)
        return PredictionOutputs(self.output_path, len(out), n_pos)
