# src/py/combine_condition_beds.py
import os
from pathlib import Path
from itertools import chain
from dataclasses import dataclass

import pybedtools
import pandas as pd

from flnc2.utils.custom_logger import get_logger
from flnc2.utils.exceptions import StepError
from flnc2.utils.pipeline_step import PipelineStep
from flnc2.utils.tools import require_files

logger = get_logger()

NAME_SEPARATOR = "|"
_BED12_COLS = ["chrom", "start", "end", "name", "score", "strand",
               "thick_start", "thick_end", "rgb", "block_count", "block_sizes", "block_starts"]
# Two transcripts are the same structure if these all match.
_STRUCTURE_COLS = ["chrom", "start", "end", "strand", "block_count", "block_sizes", "block_starts"]

def tag_name(condition: str, name: str, sep: str = NAME_SEPARATOR) -> str:
    return f"{condition}{sep}{name}"

def split_name(tagged: str, sep: str = NAME_SEPARATOR) -> tuple[str, str]:
    """'ctrl|MSTRG_ctrl.12.1' -> ('ctrl', 'MSTRG_ctrl.12.1')"""
    condition, _, name = tagged.partition(sep)
    return condition, name

@dataclass(frozen=True)
class CombinedBedOutputs:
    bed_path: Path
    duplicates_path: Path
    num_transcripts: int
    num_structures: int          # distinct exon structures across all conditions
    num_shared_structures: int   # structures found in 2+ conditions

class CombineConditionBeds(PipelineStep[CombinedBedOutputs]):
    def __init__(
        self,
        condition_beds: dict[str, str | Path],
        output_dir: str | Path,
        separator: str = NAME_SEPARATOR,
    ):
        if not condition_beds:
            raise StepError("No condition BEDs given")
        self.condition_beds = {c: Path(p) for c, p in condition_beds.items()}
        self.output_dir = Path(output_dir).resolve()
        self.separator = separator

        bad = [c for c in self.condition_beds if separator in c]
        if bad:
            raise StepError(
                f"Condition name(s) contain the separator {separator!r}: {', '.join(bad)}")

    def run(self) -> CombinedBedOutputs:
        logger.info("Combining condition BEDs into one BED")

        require_files(*self.condition_beds.values())
        self.output_dir.mkdir(parents=True, exist_ok=True)
        bed_path = self.output_dir / "combined_conditions.bed"
        dup_path = self.output_dir / "shared_structures.tsv"

        tagged = []
        for condition, path in self.condition_beds.items():
            bed = pybedtools.BedTool(str(path))
            n = bed.count()
            if n == 0:
                logger.warning("No transcripts for condition %s (%s)", condition, path)
                continue
            if bed.field_count() != 12:
                raise StepError(f"{path} is not BED12 ({bed.field_count()} columns)")
            self._check_names(bed, path)
            logger.info("  %s: %d transcripts", condition, n)
            tagged.append(bed.each(self._tagger(condition)))

        if not tagged:
            raise StepError("Every condition BED is empty; nothing to combine")

        # Concatenate without merging intervals (BedTool.cat would merge
        # overlapping features and drop the BED12 columns by default).
        combined = pybedtools.BedTool(chain.from_iterable(tagged)).sort()

        tmp = bed_path.with_suffix(".bed.tmp")
        combined.saveas(str(tmp))
        os.replace(tmp, bed_path)

        num_transcripts, num_structures, num_shared = self._write_shared(bed_path, dup_path)
        logger.info("Combined %d transcripts from %d condition(s) -> %s",
                    num_transcripts, len(tagged), bed_path)
        logger.info("  %d distinct structures, %d shared by 2+ conditions -> %s",
                    num_structures, num_shared, dup_path)

        return CombinedBedOutputs(bed_path, dup_path, num_transcripts, num_structures, num_shared)

    # ── helpers ─────────────────────────────────────────────────────────────
    def _tagger(self, condition: str):
        def tag(iv):
            iv.name = tag_name(condition, iv.name, self.separator)
            return iv
        return tag

    def _check_names(self, bed: pybedtools.BedTool, path: Path) -> None:
        # A separator inside an original name would make split_name ambiguous.
        bad = [iv.name for iv in bed if self.separator in iv.name]
        if bad:
            raise StepError(
                f"{len(bad)} transcript name(s) in {path} already contain "
                f"{self.separator!r} (e.g. {bad[0]})")

    def _write_shared(self, bed_path: Path, dup_path: Path) -> tuple[int, int, int]:
        df = pd.read_table(bed_path, header=None, names=_BED12_COLS, dtype=str)
        df["condition"] = df["name"].map(lambda n: split_name(n, self.separator)[0])

        groups = df.groupby(_STRUCTURE_COLS, sort=False)
        summary = groups.agg(
            num_conditions=("condition", "nunique"),
            conditions=("condition", lambda s: ",".join(sorted(set(s)))),
            names=("name", lambda s: ",".join(s)),
        ).reset_index()
        shared = summary[summary["num_conditions"] > 1]

        tmp = dup_path.with_suffix(".tsv.tmp")
        shared[["chrom", "start", "end", "strand", "num_conditions", "conditions", "names"]] \
            .to_csv(tmp, sep="\t", index=False)
        os.replace(tmp, dup_path)

        return len(df), len(summary), len(shared)
