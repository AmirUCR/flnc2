# src/py/classify_locus.py
import os
from pathlib import Path
from dataclasses import dataclass

import pybedtools
import pandas as pd

from flnc2.utils.custom_logger import get_logger
from flnc2.utils.exceptions import StepError
from flnc2.utils.pipeline_step import PipelineStep
from flnc2.utils.tools import require_files
from flnc2.utils.gtf_tools import cached_gtf_db, protein_coding_genes
from flnc2.combine_conditions_beds import split_name

logger = get_logger()

_BED12_COLS = ["chrom", "chromStart", "chromEnd", "name", "score", "strand",
               "thickStart", "thickEnd", "itemRgb", "blockCount", "blockSizes", "blockStarts"]

# First match wins, same order as the old classify().
LOCUS_PRECEDENCE = ("divergent", "antisense", "sense_overlap", "intergenic")
LOCUS_TYPES = LOCUS_PRECEDENCE + ("other",)

@dataclass(frozen=True)
class LocusOutputs:
    table_path: Path          # every transcript, with locus_type and one-hot columns
    bed_path: Path            # BED12 of transcripts kept (drop_types removed)
    counts: dict[str, int]    # locus_type -> number of transcripts

class ClassifyLocus(PipelineStep[LocusOutputs]):
    def __init__(
        self,
        bed_path: str | Path,
        gtf_path: str | Path,
        cache_dir: str | Path,
        output_dir: str | Path,
        divergent_window: int = 2000,
        drop_types: tuple[str, ...] = ("sense_overlap",),
    ):
        unknown = set(drop_types) - set(LOCUS_TYPES)
        if unknown:
            raise StepError(f"Unknown locus type(s) to drop: {', '.join(sorted(unknown))}")
        self.bed_path = Path(bed_path)
        self.gtf_path = Path(gtf_path)
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir).resolve()
        self.divergent_window = int(divergent_window)
        self.drop_types = tuple(drop_types)

    def run(self) -> LocusOutputs:
        logger.info("Locus classification")

        require_files(self.bed_path, self.gtf_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_table(self.bed_path, header=None, names=_BED12_COLS, dtype=str)
        if df.empty:
            raise StepError(f"No transcripts in {self.bed_path}")
        dupes = df["name"][df["name"].duplicated()].unique()
        if len(dupes):
            # Classification is keyed by name, so names must be unique.
            raise StepError(f"{len(dupes)} duplicate transcript name(s) in "
                            f"{self.bed_path} (e.g. {dupes[0]})")

        lnc = pybedtools.BedTool(str(self.bed_path)).sort()
        genes, coding_tss = self._coding_features()
        logger.info("Protein-coding genes: %d", genes.count())

        # -split: compare the lncRNA's exon blocks, not its full span.
        hits = {
            "sense_overlap": self._names(lnc.intersect(genes, s=True, u=True, split=True)),
            "antisense":     self._names(lnc.intersect(genes, S=True, u=True, split=True)),
            "intergenic":    self._names(lnc.intersect(genes, v=True, split=True)),
            # Head-to-head: an opposite-strand coding TSS within the window
            # upstream of the lncRNA's own TSS. sw=True makes l/r strand-aware.
            "divergent":     self._names(self._tss(lnc).window(
                                 coding_tss, l=self.divergent_window, r=0,
                                 sw=True, Sm=True, u=True)),
        }

        def classify(name: str) -> str:
            for locus_type in LOCUS_PRECEDENCE:
                if name in hits[locus_type]:
                    return locus_type
            return "other"

        df["locus_type"] = df["name"].map(classify)
        split = df["name"].map(split_name)
        df["condition"] = split.str[0]
        df["transcript_id"] = split.str[1]
        # Fixed one-hot columns, so the table has the same shape regardless of
        # which categories happen to occur in this dataset.
        for locus_type in LOCUS_TYPES:
            df[f"locus_{locus_type}"] = (df["locus_type"] == locus_type).astype(int)

        counts = {t: int((df["locus_type"] == t).sum()) for t in LOCUS_TYPES}
        for t in LOCUS_TYPES:
            logger.info("  %-14s %d", t, counts[t])

        table_path = self.output_dir / "locus_classification.tsv"
        bed_out = self.output_dir / "lncrna_classified.bed"
        kept = df[~df["locus_type"].isin(self.drop_types)]
        logger.info("Dropped %d / %d transcripts (%s)",
                    len(df) - len(kept), len(df), ", ".join(self.drop_types) or "none")

        self._atomic_tsv(df, table_path, header=True)
        self._atomic_tsv(kept[_BED12_COLS], bed_out, header=False)
        logger.info("Classification table: %s", table_path)
        logger.info("Kept transcripts BED: %s", bed_out)

        return LocusOutputs(table_path=table_path, bed_path=bed_out, counts=counts)

    # ── annotation ──────────────────────────────────────────────────────────
    def _coding_features(self) -> tuple[pybedtools.BedTool, pybedtools.BedTool]:
        db = cached_gtf_db(self.gtf_path, self.cache_dir)
        genes, tss = [], []
        for g in protein_coding_genes(db):
            gid = g.attributes["gene_id"][0]
            # GTF is 1-based inclusive; BED is 0-based half-open.
            genes.append(pybedtools.create_interval_from_list(
                [g.seqid, str(g.start - 1), str(g.end), gid, "0", g.strand]))
            pos = g.start - 1 if g.strand == "+" else g.end - 1
            tss.append(pybedtools.create_interval_from_list(
                [g.seqid, str(pos), str(pos + 1), gid, "0", g.strand]))
        if not genes:
            raise StepError(f"No protein_coding genes found in {self.gtf_path}")
        return (pybedtools.BedTool(genes).saveas().sort(),
                pybedtools.BedTool(tss).saveas().sort())

    @staticmethod
    def _tss(bed: pybedtools.BedTool) -> pybedtools.BedTool:
        """Single-base interval at each transcript's 5' end."""
        def to_tss(iv):
            pos = iv.start if iv.strand == "+" else iv.end - 1
            return pybedtools.create_interval_from_list(
                [iv.chrom, str(pos), str(pos + 1), iv.name, "0", iv.strand])
        return bed.each(to_tss).saveas().sort()

    @staticmethod
    def _names(bed: pybedtools.BedTool) -> set[str]:
        return {iv.name for iv in bed}

    @staticmethod
    def _atomic_tsv(df: pd.DataFrame, path: Path, header: bool) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        df.to_csv(tmp, sep="\t", index=False, header=header)
        os.replace(tmp, path)
