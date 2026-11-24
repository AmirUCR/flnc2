# src/py/transcript_features.py
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.py.utils.custom_logger import get_logger
from src.py.utils.pipeline_step import PipelineStep
from src.py.utils.tools import require_files

logger = get_logger()

_BED12_COLS = ["chrom", "chromStart", "chromEnd", "name", "score", "strand",
               "thickStart", "thickEnd", "itemRgb", "blockCount", "blockSizes", "blockStarts"]

@dataclass(frozen=True)
class TranscriptFeaturesOutputs:
    table_path: Path
    num_transcripts: int

class TranscriptFeatures(PipelineStep[TranscriptFeaturesOutputs]):
    """Per-transcript features computed from each row on its own, so they are
    safe to compute before the train/eval/test split. No scaling here."""

    def __init__(self, bed_path: str | Path, output_path: str | Path):
        self.bed_path = Path(bed_path)
        self.output_path = Path(output_path)

    def run(self) -> TranscriptFeaturesOutputs:
        require_files(self.bed_path)
        bed = pd.read_table(self.bed_path, header=None, names=_BED12_COLS)

        exon_sizes = bed["blockSizes"].str.rstrip(",").str.split(",").map(
            lambda sizes: np.array(sizes, dtype=int))

        features = pd.DataFrame({
            "name": bed["name"],
            "transcript_length": bed["chromEnd"] - bed["chromStart"],
            "mean_exon_length": exon_sizes.map(np.mean).round(2),
            "sum_exon_lengths": exon_sizes.map(np.sum),
        })
        features["log_exon_lengths"] = np.log1p(features["sum_exon_lengths"])

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(self.output_path, sep="\t", index=False)
        logger.info("Transcript features for %d transcripts -> %s", len(features), self.output_path)
        return TranscriptFeaturesOutputs(table_path=self.output_path, num_transcripts=len(features))
