# src/py/extract_promoter_sequences.py
import os
from pathlib import Path
from dataclasses import dataclass

import pybedtools

from flnc2.utils.custom_logger import get_logger
from flnc2.utils.exceptions import StepError
from flnc2.utils.pipeline_step import PipelineStep
from flnc2.utils.bed_to_fasta import BLOCK_SEPARATOR, bed_name_from_fasta_id
from flnc2.utils.tools import require_files, require_fai

logger = get_logger()

def parse_promoter_name(
    header: str, sep: str = BLOCK_SEPARATOR
) -> tuple[str, str, str, int, int]:
    """'ctrl|MSTRG.1.1|6510,8471|0,10743|orig:1000-19000'
       -> ('ctrl|MSTRG.1.1', '6510,8471', '0,10743', 1000, 19000)

    Splits from the right, so the transcript name may contain the separator.
    """
    name, sizes, starts, orig = bed_name_from_fasta_id(header).rsplit(sep, 3)
    if not orig.startswith("orig:"):
        raise ValueError(f"Not a promoter header: {header!r}")
    start, end = orig.removeprefix("orig:").split("-")
    return name, sizes, starts, int(start), int(end)

@dataclass(frozen=True)
class PromoterFastaOutputs:
    fasta_path: Path
    num_sequences: int
    num_truncated: int     # shorter than flank_length (near a chromosome end)
    num_skipped: int       # no strand, or no room upstream at all

class ExtractPromoterSequences(PipelineStep[PromoterFastaOutputs]):
    def __init__(
        self,
        bed_path: str | Path,
        genome_fasta_path: str | Path,
        output_dir: str | Path,
        flank_length: int = 1000,
        output_name: str = "flanking.fa",
    ):
        self.bed_path = Path(bed_path)
        self.genome_fasta_path = Path(genome_fasta_path)
        self.output_path = Path(output_dir).resolve() / output_name
        self.flank_length = int(flank_length)
        if self.flank_length < 1:
            raise StepError(f"flank_length must be >= 1, got {self.flank_length}")

    def run(self) -> PromoterFastaOutputs:
        logger.info("Flanking region FASTA extraction")
        require_files(self.bed_path, self.genome_fasta_path)
        # The .fai doubles as the chromosome sizes file for flank, so sizes
        # always match the genome the sequences come from.
        fai = require_fai(self.genome_fasta_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        bed = pybedtools.BedTool(str(self.bed_path)).sort()
        n_bed = bed.count()
        if n_bed == 0:
            raise StepError(f"No transcripts in {self.bed_path}")
        if bed.field_count() != 12:
            raise StepError(f"{self.bed_path} is not BED12 ({bed.field_count()} columns)")
        logger.info("Transcripts: %d", n_bed)

        # Upstream is undefined without a strand. StringTie gives "." to
        # single-exon transcripts in unstranded libraries.
        stranded = bed.filter(lambda iv: iv.strand in ("+", "-")).saveas()
        n_unstranded = n_bed - stranded.count()
        if n_unstranded:
            logger.warning("Skipping %d transcript(s) with no strand", n_unstranded)

        # BED6 with the block info and original coordinates in the name.
        # Flanking BED12 would keep the transcript's blocks and thick region,
        # which no longer describe the new interval.
        tagged = stranded.each(self._to_bed6).saveas()

        # -l N -r 0 -s: N bp upstream of the TSS only (chromStart on +,
        # chromEnd on -). Flanks are clipped at chromosome ends.
        flanks = tagged.flank(g=str(fai), l=self.flank_length, r=0, s=True).saveas()
        usable = flanks.filter(lambda iv: iv.length > 0).saveas()
        n_empty = flanks.count() - usable.count()
        if n_empty:
            logger.warning("Skipping %d transcript(s) starting at a chromosome end", n_empty)

        n_truncated = sum(1 for iv in usable if iv.length < self.flank_length)
        if n_truncated:
            logger.info("%d promoter(s) shorter than %d bp (near a chromosome end)",
                        n_truncated, self.flank_length)

        tmp = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        # No split: a flank is one contiguous interval, not a spliced transcript.
        usable.sequence(fi=str(self.genome_fasta_path), s=True, nameOnly=True)
        usable.save_seqs(str(tmp))
        os.replace(tmp, self.output_path)

        n_fa = self._count_records(self.output_path)
        expected = usable.count()
        if n_fa != expected:
            raise StepError(f"{expected} promoter intervals produced {n_fa} sequences. "
                            f"Do the BED chromosome names match the genome?")
        logger.info("Promoter FASTA: %d sequences -> %s", n_fa, self.output_path)

        return PromoterFastaOutputs(
            fasta_path=self.output_path,
            num_sequences=n_fa,
            num_truncated=n_truncated,
            num_skipped=n_unstranded + n_empty,
        )

    @staticmethod
    def _to_bed6(iv):
        sizes, starts = iv.fields[10].rstrip(","), iv.fields[11].rstrip(",")
        name = BLOCK_SEPARATOR.join((iv.name, sizes, starts, f"orig:{iv.start}-{iv.end}"))
        return pybedtools.create_interval_from_list(
            [iv.chrom, str(iv.start), str(iv.end), name, iv.score, iv.strand])

    @staticmethod
    def _count_records(path: Path) -> int:
        with path.open() as f:
            return sum(1 for line in f if line.startswith(">"))
