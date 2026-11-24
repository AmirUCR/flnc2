# src/py/bed_to_fasta.py
import os
import re
from pathlib import Path
from dataclasses import dataclass

import pybedtools

from src.py.utils.custom_logger import get_logger
from src.py.utils.exceptions import StepError
from src.py.utils.pipeline_step import PipelineStep
from src.py.utils.tools import require_files, require_fai

logger = get_logger()

BLOCK_SEPARATOR = "|"

_STRAND_SUFFIX = re.compile(r"\([+\-.]\)$")

def bed_name_from_fasta_id(seq_id: str) -> str:
    """Recover the BED name from a header written by bedtools getfasta.

    getfasta -s appends the strand: 'NAME(-)' with -nameOnly, or
    'NAME::chr1:100-200(-)' with -name. Both come back as 'NAME'.
    Accepts the header with or without '>' and ignores any description.
    """
    name = seq_id.lstrip(">").split()[0].split("::", 1)[0]
    return _STRAND_SUFFIX.sub("", name)

def parse_fasta_name(header: str, sep: str = BLOCK_SEPARATOR) -> tuple[str, str, str]:
    """'ctrl|MSTRG.1.1|6510,8471|0,10743' -> ('ctrl|MSTRG.1.1', '6510,8471', '0,10743').

    Splits from the right, so the transcript name may itself contain the
    separator (as condition-tagged names do). Accepts the header with or
    without the leading '>'.
    """
    name, sizes, starts = bed_name_from_fasta_id(header).rsplit(sep, 2)
    return name, sizes, starts

@dataclass(frozen=True)
class TranscriptFastaOutputs:
    fasta_path: Path
    num_sequences: int

class ExtractTranscriptSequences(PipelineStep[TranscriptFastaOutputs]):
    def __init__(
        self,
        bed_path: str | Path,
        genome_fasta_path: str | Path,
        output_dir: str | Path,
        output_name: str = "transcripts.fa",
        embed_blocks: bool = True,
    ):
        self.bed_path = Path(bed_path)
        self.genome_fasta_path = Path(genome_fasta_path)
        self.output_path = Path(output_dir).resolve() / output_name
        self.embed_blocks = embed_blocks

    def run(self) -> TranscriptFastaOutputs:
        logger.info("Transcript FASTA extraction")

        require_files(self.bed_path, self.genome_fasta_path)
        require_fai(self.genome_fasta_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        bed = pybedtools.BedTool(str(self.bed_path)).sort()
        n_bed = bed.count()
        if n_bed == 0:
            raise StepError(f"No transcripts in {self.bed_path}")
        # Extra columns make -split silently use the full genomic span
        # instead of the exon blocks.
        if bed.field_count() != 12:
            raise StepError(f"{self.bed_path} is not BED12 ({bed.field_count()} columns)")
        logger.info("Transcripts in %s: %d", self.bed_path.name, n_bed)

        if self.embed_blocks:
            bed = bed.each(self._embed_blocks).saveas()

        tmp = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        # split=True: concatenated exon blocks (mature transcript).
        # s=True: reverse-complement minus-strand transcripts.
        # nameOnly=True: header is the name alone, without "::chr:start-end(strand)".
        bed.sequence(fi=str(self.genome_fasta_path), s=True, split=True, nameOnly=True)
        bed.save_seqs(str(tmp))
        os.replace(tmp, self.output_path)

        n_fa = self._count_records(self.output_path)
        logger.info("Transcript FASTA: %d sequences -> %s", n_fa, self.output_path)
        if n_fa != n_bed:
            raise StepError(f"{n_bed} BED rows produced {n_fa} sequences. "
                            f"Do the BED chromosome names match the genome?")

        return TranscriptFastaOutputs(fasta_path=self.output_path, num_sequences=n_fa)

    @staticmethod
    def _embed_blocks(iv):
        # name|blockSizes|blockStarts, e.g. ctrl|MSTRG.9880.6|6510,8471|0,10743
        sizes, starts = iv.fields[10].rstrip(","), iv.fields[11].rstrip(",")
        iv.name = BLOCK_SEPARATOR.join((iv.name, sizes, starts))
        return iv

    @staticmethod
    def _count_records(path: Path) -> int:
        with path.open() as f:
            return sum(1 for line in f if line.startswith(">"))
