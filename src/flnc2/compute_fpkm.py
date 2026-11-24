# src/py/compute_fpkm.py
from pathlib import Path
from dataclasses import dataclass

import pandas as pd

from flnc2.utils.custom_logger import get_logger
from flnc2.utils.exceptions import StepError
from flnc2.utils.pipeline_step import PipelineStep
from flnc2.align_reads import AlignmentOutputs
from flnc2.utils.tools import require_files

logger = get_logger()

@dataclass(frozen=True)
class FPKMOutputs:
    fpkm_path: Path
    num_genes: int

class ComputeFPKM(PipelineStep[FPKMOutputs]):
    """Gene FPKM from a featureCounts table, for expression filtering only
    (DESeq2/edgeR need the raw counts).

    Denominator is unique + multi aligned, reproducing the original rpkm.pl.
    featureCounts runs without -M, so multi-mappers are in the denominator but
    not the numerator, and every FPKM is deflated by the multi-mapping fraction.
    This is intentional, for comparability with the published values; changing
    it means re-deriving the expression cutoff.

    Units match per sample: PE columns are fragment counts (--countReadPairs)
    and the PE denominator is concordant pairs; SE columns and denominator are
    both reads.
    """

    def __init__(self, counts_path: str | Path, alignments: AlignmentOutputs,
                 output_path: str | Path):
        self.counts_path = Path(counts_path)
        self.alignments = alignments
        self.output_path = Path(output_path)

    def run(self) -> FPKMOutputs:
        logger.info("Computing FPKM")
        require_files(self.counts_path)
        fc = pd.read_table(self.counts_path, comment="#")

        samples = [c for c in fc.columns if c in self.alignments.samples]
        if not samples:
            raise StepError(f"No sample columns in {self.counts_path}")

        length_kb = fc["Length"].to_numpy() / 1000
        fpkm = pd.DataFrame(index=fc["Geneid"])
        for sid in samples:
            aln = self.alignments.samples[sid]
            if aln.unique_aligned is None or aln.multi_aligned is None:
                raise StepError(f"Could not read aligned counts for {sid} from {aln.log}")
            total = aln.unique_aligned + aln.multi_aligned
            if total == 0:
                raise StepError(f"{sid} has zero aligned {'pairs' if aln.paired else 'reads'}")
            fpkm[sid] = fc[sid].to_numpy() / (length_kb * total / 1e6)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fpkm.to_csv(self.output_path, sep="\t")
        logger.info("  FPKM: %d samples, %d genes -> %s", len(samples), len(fpkm), self.output_path)
        return FPKMOutputs(fpkm_path=self.output_path, num_genes=len(fpkm))
