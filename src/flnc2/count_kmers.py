# src/py/count_kmers.py
import os
from pathlib import Path
from typing import Callable
from itertools import product
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
from Bio import SeqIO

from flnc2.utils.custom_logger import get_logger
from flnc2.utils.exceptions import StepError
from flnc2.utils.pipeline_step import PipelineStep
from flnc2.utils.tools import require_files

logger = get_logger()

ALPHABET = "ACTG"
_BED12_COLS = ["chrom", "chromStart", "chromEnd", "name", "score", "strand",
               "thickStart", "thickEnd", "itemRgb", "blockCount", "blockSizes", "blockStarts"]
_BED6_COLS = ["chrom", "chromStart", "chromEnd", "name", "strand", "blockSizes", "blockStarts"]

# Byte -> base index; -1 for anything that isn't A/C/T/G (N, IUPAC codes, ...).
# Lowercase maps too, since soft-masked genomes write repeats in lowercase.
_LUT = np.full(256, -1, dtype=np.int64)
for _i, _b in enumerate(ALPHABET):
    _LUT[ord(_b)] = _i
    _LUT[ord(_b.lower())] = _i

def kmer_names(k: int) -> list[str]:
    return ["".join(p) for p in product(ALPHABET, repeat=k)]

def count_kmers(seq: str, k: int) -> np.ndarray:
    """Counts of every k-mer, in kmer_names(k) order. Windows containing any
    non-ACGT base are skipped, as in the old scripts."""
    codes = _LUT[np.frombuffer(seq.encode("ascii", "replace"), dtype=np.uint8)]
    if len(codes) < k:
        return np.zeros(4 ** k, dtype=np.int64)
    windows = sliding_window_view(codes, k)
    windows = windows[(windows >= 0).all(axis=1)]
    weights = 4 ** np.arange(k - 1, -1, -1, dtype=np.int64)
    return np.bincount(windows @ weights, minlength=4 ** k)

@dataclass(frozen=True)
class KmerOutputs:
    table_path: Path
    k: int
    num_sequences: int

class CountKmers(PipelineStep[KmerOutputs]):
    def __init__(
        self,
        fasta_path: str | Path,
        bed_path: str | Path,
        output_path: str | Path,
        k: int,
        name_from_header: Callable[[str], str] = lambda header: header,
    ):
        self.fasta_path = Path(fasta_path)
        self.bed_path = Path(bed_path)
        self.output_path = Path(output_path).resolve()
        self.k = int(k)
        self.name_from_header = name_from_header
        if not 1 <= self.k <= 8:
            # 4**k columns: k=8 is already 65,536.
            raise StepError(f"k must be between 1 and 8, got {self.k}")

    def run(self) -> KmerOutputs:
        require_files(self.fasta_path, self.bed_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        meta = pd.read_table(self.bed_path, header=None, names=_BED12_COLS, dtype=str)
        meta = meta.set_index("name")

        names, rows = [], []
        for record in SeqIO.parse(str(self.fasta_path), "fasta"):
            names.append(self.name_from_header(record.id))
            rows.append(count_kmers(str(record.seq), self.k))
        if not names:
            raise StepError(f"No sequences in {self.fasta_path}")

        dupes = pd.Index(names)[pd.Index(names).duplicated()].unique()
        if len(dupes):
            raise StepError(f"{len(dupes)} duplicate name(s) in {self.fasta_path} (e.g. {dupes[0]})")
        missing = pd.Index(names).difference(meta.index)
        if len(missing):
            raise StepError(f"{len(missing)} FASTA name(s) not found in {self.bed_path} "
                            f"(e.g. {missing[0]})")

        df = pd.concat([
            meta.loc[names].reset_index()[_BED6_COLS],
            pd.DataFrame(np.vstack(rows), columns=kmer_names(self.k)),
        ], axis=1)

        tmp = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        df.to_csv(tmp, sep="\t", index=False)
        os.replace(tmp, self.output_path)

        logger.info("%d-mer counts for %d sequences -> %s", self.k, len(df), self.output_path)
        return KmerOutputs(table_path=self.output_path, k=self.k, num_sequences=len(df))
