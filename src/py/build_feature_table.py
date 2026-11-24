# src/py/build_feature_table.py
from pathlib import Path
from dataclasses import dataclass

import pandas as pd

from src.py.utils.custom_logger import get_logger
from src.py.utils.exceptions import StepError
from src.py.utils.pipeline_step import PipelineStep
from src.py.utils.tools import require_files

logger = get_logger()

# Coordinate columns the k-mer tables repeat from the BED. The base table
# already has them, so they're dropped before joining.
_KMER_META_COLS = ["chrom", "chromStart", "chromEnd", "strand", "blockSizes", "blockStarts"]

@dataclass(frozen=True)
class FeatureTableOutputs:
    table_path: Path
    num_rows: int

class BuildFeatureTable(PipelineStep[FeatureTableOutputs]):
    def __init__(
        self,
        locus_table_path: str | Path,
        transcript_features_path: str | Path,
        transcript_kmers_path: str | Path,
        promoter_kmers_path: str | Path,
        output_path: str | Path,
    ):
        self.locus_table_path = Path(locus_table_path)
        self.transcript_features_path = Path(transcript_features_path)
        self.transcript_kmers_path = Path(transcript_kmers_path)
        self.promoter_kmers_path = Path(promoter_kmers_path)
        self.output_path = Path(output_path)

    def run(self) -> FeatureTableOutputs:
        require_files(self.locus_table_path, self.transcript_features_path,
                      self.transcript_kmers_path, self.promoter_kmers_path)

        df = pd.read_table(self.locus_table_path)
        features = pd.read_table(self.transcript_features_path)
        transcript_kmers = pd.read_table(self.transcript_kmers_path).drop(columns=_KMER_META_COLS)
        promoter_kmers = pd.read_table(self.promoter_kmers_path).drop(columns=_KMER_META_COLS)

        # Same k-mer names in both tables, so mark the promoter ones.
        promoter_kmers = promoter_kmers.rename(
            columns={c: f"{c}_flank" for c in promoter_kmers.columns if c != "name"})

        '''
        The locus table includes the sense_overlap rows that ClassifyLocus dropped
        from its BED; they have no k-mers, so the join removes them. 
        Any transcript without a promoter (one starting at a chromosome edge) 
        is removed at the last join. The per-join log line shows how many rows
        each table removed.
        '''
        logger.info("Building feature table from %d transcripts", len(df))
        for label, table in (("transcript features", features),
                             ("transcript k-mers", transcript_kmers),
                             ("promoter k-mers", promoter_kmers)):
            before = len(df)
            df = df.merge(table, how="inner", on="name", validate="one_to_one")
            logger.info("  + %s: %d rows (dropped %d)", label, len(df), before - len(df))

        if df.empty:
            raise StepError("Feature table is empty after joining; do the tables share names?")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_path, sep="\t", index=False)
        logger.info("Feature table: %d rows x %d columns -> %s",
                    len(df), df.shape[1], self.output_path)
        return FeatureTableOutputs(table_path=self.output_path, num_rows=len(df))
