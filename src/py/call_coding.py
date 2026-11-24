import os
import re
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

import pybedtools
import pandas as pd

from src.py.utils.custom_logger import get_logger
from src.py.utils.exceptions import StepError
from src.py.utils.pipeline_step import PipelineStep
from src.py.utils.bed_to_fasta import bed_name_from_fasta_id

logger = get_logger()

@dataclass(frozen=True)
class PredictorCalls:
    name: str
    all_ids: frozenset[str]      # every transcript the tool reported on
    coding_ids: frozenset[str]   # transcripts the tool called coding

@dataclass(frozen=True)
class CallCodingOutputs:
    noncoding_bed_path: Path
    coding_bed_path: Path
    summary_path: Path
    num_coding: int
    num_noncoding: int

def _transcript_id(seq_ids: pd.Series) -> pd.Series:
    # Undo getfasta's additions, then uppercase because CPAT uppercases IDs.
    return seq_ids.astype(str).map(bed_name_from_fasta_id).str.upper()

def _unify_lgc_label(label: str) -> str:
    label = label.strip().lower()
    if "non" in label:
        return "noncoding"
    if label in ("protein coding", "protein-coding", "coding"):
        return "coding"
    raise StepError(f"Unknown LGC coding label: {label!r}")

class CallCoding(PipelineStep[CallCodingOutputs]):
    def __init__(
        self,
        bed_path: str | Path,
        cpat_best_tsv_path: str | Path | None,
        cpat_no_orf_path: str | Path | None,
        cpat_cutoff_path: str | Path,
        lgc_out_path: str | Path,
        output_dir: str | Path
    ):
        self.bed_path = Path(bed_path)
        self.cpat_best_tsv_path = Path(cpat_best_tsv_path) if cpat_best_tsv_path else None
        self.cpat_no_orf_path = Path(cpat_no_orf_path) if cpat_no_orf_path else None
        self.cpat_cutoff_path = Path(cpat_cutoff_path)
        self.lgc_out_path = Path(lgc_out_path)
        self.output_dir = Path(output_dir)

    def run(self) -> CallCodingOutputs:
        required = [self.bed_path, self.cpat_cutoff_path, self.lgc_out_path]
        required += [p for p in (self.cpat_best_tsv_path, self.cpat_no_orf_path) if p is not None]
        missing = [str(p) for p in required if not p.is_file()]
        if missing:
            raise StepError("Required file(s) not found: " + ", ".join(missing))

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # A transcript is excluded if ANY tool calls it coding (matches the
        # original paper). Transcripts absent from a tool's output, e.g. CPAT
        # finding no ORF, are implicitly noncoding, which is why we exclude a
        # coding set rather than select a noncoding one.
        predictors = [
            self._cpat_calls(),
            self._lgc_calls(),
            # add another tool here
        ]
        coding_union = frozenset().union(*(p.coding_ids for p in predictors))

        bed = pybedtools.BedTool(str(self.bed_path))
        names = [iv.name for iv in bed]
        num_input = len(names)
        if num_input == 0:
            raise StepError(f"No transcripts in {self.bed_path}")

        self._check_case_collisions(names)
        keys = {n.upper() for n in names}
        for p in predictors:
            self._check_matched_ids(p, keys, num_input, self.bed_path)

        noncoding_bed_path = self.output_dir / "predicted_noncoding.bed"
        coding_bed_path = self.output_dir / "predicted_coding.bed"
        num_noncoding = self._write_subset(
            bed, noncoding_bed_path, lambda iv: iv.name.upper() not in coding_union)
        num_coding = self._write_subset(
            bed, coding_bed_path, lambda iv: iv.name.upper() in coding_union)

        logger.info("  Coding calls (union across tools): %d", len(coding_union))
        logger.info("  Retained: %d / %d (%.1f%%)",
                    num_noncoding, num_input, 100 * num_noncoding / num_input)

        summary_path = self._write_summary(num_input, num_coding, num_noncoding)
        return CallCodingOutputs(
            noncoding_bed_path=noncoding_bed_path,
            coding_bed_path=coding_bed_path,
            summary_path=summary_path,
            num_coding=num_coding,
            num_noncoding=num_noncoding,
        )

    # ── predictors ──────────────────────────────────────────────────────────
    def _cpat_calls(self) -> PredictorCalls:
        cutoff = self._read_cpat_cutoff()
        logger.info("CPAT cutoff: %s", cutoff)

        all_ids: set[str] = set()
        coding_ids: set[str] = set()

        if self.cpat_best_tsv_path is not None:
            df = pd.read_table(self.cpat_best_tsv_path).rename(columns={"seq_ID": "seq_id"})
            for col in ("seq_id", "Coding_prob"):
                if col not in df.columns:
                    raise StepError(f"CPAT output missing column '{col}': {self.cpat_best_tsv_path}")
            ids = _transcript_id(df["seq_id"])
            all_ids |= set(ids)
            coding_ids |= set(ids[df["Coding_prob"].astype(float) >= cutoff])

        if self.cpat_no_orf_path is not None:
            no_orf = pd.read_table(self.cpat_no_orf_path, names=["seq_id"])
            all_ids |= set(_transcript_id(no_orf["seq_id"]))

        return PredictorCalls("CPAT", frozenset(all_ids), frozenset(coding_ids))

    def _lgc_calls(self) -> PredictorCalls:
        df = pd.read_table(self.lgc_out_path)
        df.columns = df.columns.str.lstrip("#").str.strip()
        for col in ("Sequence Name", "Coding Label"):
            if col not in df.columns:
                raise StepError(f"LGC output missing column '{col}': {self.lgc_out_path}")

        ids = _transcript_id(df["Sequence Name"])
        labels = df["Coding Label"].astype(str).map(_unify_lgc_label)
        return PredictorCalls(
            "LGC2.0", frozenset(ids), frozenset(ids[labels == "coding"]))

    def _read_cpat_cutoff(self) -> float:
        line = self.cpat_cutoff_path.read_text().strip().splitlines()[0]
        try:
            return float(line.rsplit(":", 1)[-1])
        except ValueError:
            raise StepError(f"Could not parse a cutoff from {self.cpat_cutoff_path}: {line!r}")

    # ── checks ──────────────────────────────────────────────────────────────
    def _check_case_collisions(self, names: list[str]) -> None:
        by_key: dict[str, set[str]] = {}
        for n in names:
            by_key.setdefault(n.upper(), set()).add(n)
        collisions = {k: v for k, v in by_key.items() if len(v) > 1}
        if collisions:
            example = ", ".join(sorted(next(iter(collisions.values()))))
            raise StepError(
                f"{len(collisions)} BED names collide when uppercased (e.g. {example})"
            )

    @staticmethod
    def _check_matched_ids(p: PredictorCalls, keys: set[str], num_input: int, bed_path: Path) -> None:
        num_matched = len(keys & p.all_ids)
        logger.info("  BED names matching a %s record: %d / %d", p.name, num_matched, num_input)
        if num_matched == 0:
            tool_examples = ", ".join(sorted(p.all_ids)[:3]) or "(none)"
            bed_examples = ", ".join(sorted(keys)[:3])
            raise StepError(
                f"No ID overlap between {p.name} output and the BED at {bed_path}. "
                f"{p.name} IDs look like: {tool_examples}; BED names look like: {bed_examples}")

    # ── output ──────────────────────────────────────────────────────────────
    @staticmethod
    def _write_subset(bed: pybedtools.BedTool, dest: Path, keep) -> int:
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        subset = bed.filter(keep).saveas(str(tmp))
        os.replace(tmp, dest)
        return pybedtools.BedTool(str(dest)).count()

    def _write_summary(self, num_input: int, num_coding: int, num_noncoding: int) -> Path:
        summary_file = self.output_dir / "predicted_coding_filter_summary.tsv"
        with summary_file.open("w") as f:
            f.write(f"# generated by {type(self).__name__} on: "
                    f"{datetime.now().astimezone().isoformat(timespec='seconds')}\n")
            f.write("# num_input: number of transcripts before coding calls.\n")
            f.write("# num_coding: number of transcripts predicted to be coding.\n")
            f.write("# num_noncoding: number of transcripts not predicted to be coding.\n")
            f.write("input\tnum_input\tnum_coding\tnum_noncoding\n")
            f.write(f"{self.bed_path.name}\t{num_input}\t{num_coding}\t{num_noncoding}\n")
        logger.info("Find a report in %s", summary_file)
        return summary_file
