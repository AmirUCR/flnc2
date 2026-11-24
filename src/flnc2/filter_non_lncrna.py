import os
import re
from pathlib import Path
from dataclasses import dataclass

import pybedtools

from flnc2.utils.custom_logger import get_logger
from flnc2.utils.exceptions import StepError
from flnc2.utils.pipeline_step import PipelineStep
from flnc2.utils.py_bedtools import remove_impure_loci
from flnc2.utils.cache import ensure_cached
from flnc2.utils.gtf_tools import cached_gtf_db, transcripts_to_bed12

logger = get_logger()

DEFAULT_BIOTYPES_TO_EXCLUDE = (
    "protein_coding", "miRNA", "snRNA", "snoRNA", "Mt_tRNA", "Mt_rRNA",
    "IG_V_gene", "TR_V_gene", "IG_V_pseudogene", "TR_C_gene", "IG_C_gene",
    "TR_V_pseudogene", "pseudogene", "rRNA_pseudogene", "unitary_pseudogene",
    "processed_pseudogene", "unprocessed_pseudogene",
    "transcribed_unitary_pseudogene", "transcribed_unprocessed_pseudogene",
    "transcribed_processed_pseudogene",
)

@dataclass(frozen=True)
class FilterNonLncRNAOutputs:
    bed_path: Path
    summary_path: Path
    num_final: int

class FilterNonLncRNA(PipelineStep[FilterNonLncRNAOutputs]):
    def __init__(
        self,
        bed_path: str | Path,
        gtf_path: str | Path,
        output_dir: str | Path,
        cache_dir: str | Path,
        canonical_re: str,
        min_length: int,
        min_exon_length: int = 2,
        biotypes_to_exclude: tuple[str] | None = None,
    ):
        self.bed_path        = Path(bed_path)
        self.gtf_path        = Path(gtf_path)
        self.output_dir      = Path(output_dir)
        self.cache_dir       = Path(cache_dir)
        self.min_length      = min_length
        self.min_exon_length = min_exon_length
        self.canonical_re = re.compile(canonical_re)
        self.biotypes_to_exclude = (
            DEFAULT_BIOTYPES_TO_EXCLUDE if biotypes_to_exclude is None
            else biotypes_to_exclude
        )

    def run(self) -> FilterNonLncRNAOutputs:
        logger.info("Filtering out non-LncRNA transcripts by checking your BED input against the provided GTF...")

        try:
            for path in (self.bed_path, self.gtf_path):
                if not os.path.isfile(path):
                    raise StepError(f"FilterNonLncRNA error: required file not found: {path}")

            os.makedirs(self.output_dir, exist_ok=True)

            # Reference-derived exclusion set (condition-independent, build once).
            exclusion = self._build_exclusion_bed()
            num_exclude = exclusion.count()
            logger.info("  Non-lncRNA features to exclude: %d", num_exclude)

            input_bed = pybedtools.BedTool(self.bed_path)
            num_raw = input_bed.count()

            canonical = self._filter_canonical(input_bed)
            num_input = canonical.count()
            logger.info(
                "Input transcripts: %d (dropped %d on non-canonical scaffolds)",
                num_input, num_raw - num_input,
            )

            # No same-strand overlap with non-lncRNA.
            notoverlap = canonical.intersect(exclusion, s=True, v=True, wa=True).saveas()
            num_notoverlap = notoverlap.count()
            logger.info(
                "  no same-strand overlap with non-lncRNA: %d (dropped %d)",
                num_notoverlap, num_input - num_notoverlap,
            )

            # Overlapping non-lncRNA (used to flag impure loci).
            overlap = canonical.intersect(exclusion, s=True, u=True).saveas()
            num_overlap = overlap.count()
            logger.info("  overlapping non-lncRNA (used to flag impure loci): %d", num_overlap)

            # Drop loci shared with excluded transcripts.
            pureloci = notoverlap.intersect(overlap, s=True, v=True, wa=True).saveas()
            num_pure = pureloci.count()
            logger.info(
                "  after removing loci shared with excluded transcripts: %d (dropped %d)",
                num_pure, num_notoverlap - num_pure,
            )

            # Keep transcripts whose mature length (sum of blockSizes) >= min_length.
            pureloci = self._filter_mature_length(pureloci)
            num_long = pureloci.count()
            logger.info(
                "  after mature length >= %d nt: %d (dropped %d)",
                self.min_length, num_long, num_pure - num_long,
            )

            pureloci  = self._filter_small_exons(pureloci)
            num_final = pureloci.count()
            logger.info(
                "  after keeping transcripts with exon length >= %d nt: %d (dropped %d)",
                self.min_exon_length, num_final, num_long - num_final,
            )

            pureloci_path = os.path.join(self.output_dir, "pureloci.bed")
            pureloci.saveas(pureloci_path)

            pct = (100 * num_final / num_input) if num_input else 0.0
            logger.info("  Retained %d / %d transcripts (%.1f%%)", num_final, num_input, pct)
            logger.info("  BED: %s", pureloci_path)

            summary_path = self._write_summary(
                num_raw=num_raw,
                num_input=num_input,
                num_notoverlap=num_notoverlap,
                num_overlap=num_overlap,
                num_pure=num_pure,
                num_final=num_final,
            )

            return FilterNonLncRNAOutputs(
                bed_path=pureloci_path, 
                summary_path=summary_path, 
                num_final=num_final)
        except Exception as e:
            raise StepError(f"FilterNonLncRNA error: {e}.")

    def _build_exclusion_bed(self) -> pybedtools.BedTool:
        dest = Path(self.cache_dir) / f"{Path(self.gtf_path).name}.exclusion.bed"

        def build(tmp: Path) -> None:
            db = cached_gtf_db(self.gtf_path, self.cache_dir)
            bed = transcripts_to_bed12(db, keep_biotypes=self.biotypes_to_exclude)
            self._filter_canonical(bed).saveas(str(tmp))

        ensure_cached(
            dest,
            inputs={"gtf": Path(self.gtf_path)},
            build=build,
            # Rebuild if the settings that shape the BED change, not just the GTF.
            extra={"biotypes": sorted(self.biotypes_to_exclude),
                   "canonical_re": self.canonical_re.pattern},
        )
        return pybedtools.BedTool(str(dest))

    def _filter_canonical(self, bed: pybedtools.BedTool) -> pybedtools.BedTool:
        return bed.filter(lambda iv: self.canonical_re.search(iv.chrom)).saveas()

    def _filter_mature_length(self, bed: pybedtools.BedTool) -> pybedtools.BedTool:
        def keep(iv):
            sizes = iv.fields[10].rstrip(",").split(",")
            return sum(int(s) for s in sizes if s) >= self.min_length
        return bed.filter(keep).saveas()

    def _filter_small_exons(self, bed: pybedtools.BedTool) -> pybedtools.BedTool:
        def keep(iv):
            sizes = iv.fields[10].rstrip(",").split(",")
            return all(int(s) >= self.min_exon_length for s in sizes)
        return bed.filter(keep).saveas()

    def _write_summary(self, **counts) -> str:
        summary_file = self.output_dir / "nonlncrna_filter_summary.tsv"
        label = os.path.basename(self.bed_path)
        with open(summary_file, "w") as f:
            f.write("# num_raw: transcripts in the input BED.\n")
            f.write("# num_input: after dropping non-canonical scaffolds.\n")
            f.write("# num_notoverlap: no same-strand overlap with non-lncRNA.\n")
            f.write("# num_overlap: overlapping non-lncRNA (used to flag impure loci).\n")
            f.write("# num_pure: after removing loci shared with excluded transcripts.\n")
            f.write(f"# num_final: after mature length >= {self.min_length} and min exon length >= {self.min_exon_length}.\n")
            f.write("input\tnum_raw\tnum_input\tnum_notoverlap\tnum_overlap\tnum_pure\tnum_final\n")
            f.write(
                f"{label}\t{counts['num_raw']}\t{counts['num_input']}\t"
                f"{counts['num_notoverlap']}\t{counts['num_overlap']}\t"
                f"{counts['num_pure']}\t{counts['num_final']}\n"
            )
        logger.info("Find a report in %s", summary_file)
        return summary_file
