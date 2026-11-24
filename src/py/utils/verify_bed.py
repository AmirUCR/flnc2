# src/py/utils/verify_bed.py
from pathlib import Path

import pybedtools

from src.py.utils.custom_logger import get_logger
from src.py.utils.exceptions import StepError
from src.py.utils.pipeline_step import PipelineStep

logger = get_logger()

class VerifyBED(PipelineStep[Path]):
    def __init__(
        self,
        bed_path: str | Path,
        output_dir: str | Path
    ):
        self.bed_path = Path(bed_path)
        self.output_dir = Path(output_dir)

    def run(self) -> Path:
        logger.info("Simple checks on the BED file...")
        try:
            bed = pybedtools.BedTool(str(self.bed_path))
            if bed.field_count() != 12:
                raise StepError(f"{self.bed_path} is not a BED12 file with 12 columns.")

            total = 0
            unstranded = []
            for iv in bed:
                total += 1
                if iv.start < 0 or iv.end < 0:
                    raise StepError(f"Negative coordinates ({iv.start}, {iv.end}) for {iv.name}")
                if iv.start >= iv.end:
                    raise StepError(f"Invalid interval for {iv.name}: {iv.start} >= {iv.end}")
                if iv.strand not in ("+", "-"):
                    unstranded.append(iv.name)
        except StepError:
            raise
        except Exception as e:
            raise StepError(f"VerifyBED error: {e}. Is the input a BED12 file, and is it formatted correctly?")

        if total == 0:
            raise StepError(f"{self.bed_path} has no records.")

        if not unstranded:
            logger.info("OK")
            return self.bed_path

        if len(unstranded) == total:
            raise StepError(
                f"All {total} records in {self.bed_path} have no strand ('+' or '-'). "
                f"Flnc2 needs stranded transcripts for its overlap filters and promoter "
                f"extraction. Unstranded libraries or single-exon assemblies are the usual cause.")

        examples = ", ".join(unstranded[:5]) + (", ..." if len(unstranded) > 5 else "")
        logger.warning("Removing %d of %d record(s) with no strand: %s",
                       len(unstranded), total, examples)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        out = self.output_dir / f"{self.bed_path.stem}.stranded.bed"
        bed.filter(lambda iv: iv.strand in ("+", "-")).saveas(str(out))
        logger.info("Using %s for the rest of the pipeline", out)
        return out
