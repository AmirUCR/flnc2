# src/py/merge_assemblies.py
import os
import json
import shlex
import shutil
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

import pybedtools

from src.py.utils.custom_logger import get_logger
from src.py.utils.exceptions import StepError
from src.py.utils.pipeline_step import PipelineStep
from src.py.utils.samples import Sample
from src.py.utils.cache import file_fingerprint
from src.py.utils.tools import require_files, run_tool, with_env, manifest_matches
from src.py.utils.gtf_tools import build_gtf_db, cached_gtf_db, transcripts_to_bed12
from src.py.assemble_transcripts import AssemblyOutputs

logger = get_logger()

_RESERVED_ARGS = {"--merge", "-o", "-G", "-p", "-l"}

@dataclass(frozen=True)
class ConditionMerge:
    condition: str
    gtf: Path
    bed: Path
    num_inputs: int
    num_transcripts: int

@dataclass(frozen=True)
class MergeOutputs:
    conditions: dict[str, ConditionMerge]
    summary: Path

class MergeAssemblies(PipelineStep[MergeOutputs]):
    def __init__(
        self,
        samples: list[Sample],
        assemblies: AssemblyOutputs,
        gtf_path: str | Path,
        output_dir: str | Path,
        reports_dir: str | Path,
        cache_dir: str | Path,
        threads: int = 1,
        extra_args: list[str] | str | None = None,
        stringtie_env: str | None = None,
    ):
        missing = [s.sample_id for s in samples if s.sample_id not in assemblies.samples]
        if missing:
            raise StepError("No assembly for sample(s): " + ", ".join(missing))

        # condition -> per-sample GTFs, both in sample-sheet order
        self.cond_gtfs: dict[str, list[Path]] = {}
        for s in samples:
            self.cond_gtfs.setdefault(s.condition, []).append(assemblies.samples[s.sample_id].gtf)
        if not self.cond_gtfs:
            raise StepError("No samples given")

        self.gtf_path = Path(gtf_path).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.reports_dir = Path(reports_dir).resolve()
        self.cache_dir = Path(cache_dir).resolve()
        self.threads = int(threads)
        self.extra_args = self._parse_extra_args(extra_args)
        self.stringtie_env = stringtie_env

    @staticmethod
    def _parse_extra_args(extra_args: list[str] | str | None) -> list[str]:
        if extra_args is None:
            return []
        args = shlex.split(extra_args) if isinstance(extra_args, str) else [str(a) for a in extra_args]
        clash = sorted({a for a in args if a in _RESERVED_ARGS})
        if clash:
            raise StepError(
                f"StringTie merge option(s) {', '.join(clash)} are set by Flnc2 and "
                f"can't be passed as extra arguments"
            )
        return args

    def run(self) -> MergeOutputs:
        logger.info("Merge assemblies")
        require_files(self.gtf_path, *(g for gtfs in self.cond_gtfs.values() for g in gtfs))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        ref_db = cached_gtf_db(self.gtf_path, self.cache_dir)
        num_ref = ref_db.count_features_of_type("transcript")

        results = {}
        for condition, gtfs in self.cond_gtfs.items():
            results[condition] = self._merge(condition, gtfs)
            r = results[condition]
            logger.info("  %s: %d transcripts from %d assemblies (reference has %d)",
                        condition, r.num_transcripts, r.num_inputs, num_ref)
            logger.info("  GTF: %s", r.gtf)
            logger.info("  BED: %s", r.bed)

        summary = self._write_summary(results, num_ref)
        return MergeOutputs(conditions=results, summary=summary)

    # ── per condition ───────────────────────────────────────────────────────
    def _merge(self, condition: str, gtfs: list[Path]) -> ConditionMerge:
        merged = self.output_dir / f"{condition}_merged.gtf"
        bed = self.output_dir / f"{condition}_merged.bed"
        mergelist = self.output_dir / f"{condition}_mergelist.txt"
        log = self.output_dir / f"{condition}_merge.log"
        manifest = self.output_dir / f"{condition}_merge.manifest.json"

        expected = {
            "inputs": [file_fingerprint(g) for g in gtfs],
            "reference_gtf": file_fingerprint(self.gtf_path),
            "extra_args": self.extra_args,
        }
        if merged.is_file() and bed.is_file() and manifest_matches(manifest, expected):
            logger.info("Using existing merge for %s", condition)
            return ConditionMerge(condition, merged, bed, len(gtfs), self._count(bed))

        manifest.unlink(missing_ok=True)
        # Kept on disk (not in the temp dir) as a record of what was merged.
        mergelist.write_text("".join(f"{g}\n" for g in gtfs))
        logger.info("Merging %d assemblies for %s", len(gtfs), condition)

        with tempfile.TemporaryDirectory(dir=self.output_dir, prefix=".tmp_") as tmpdir:
            tmp_gtf = Path(tmpdir) / merged.name
            cmd = ["stringtie", "--merge", *self.extra_args,
                   "-p", str(self.threads),
                   "-G", str(self.gtf_path),
                   "-l", f"MSTRG_{condition}",
                   "-o", str(tmp_gtf),
                   str(mergelist)]
            run_tool(with_env(self.stringtie_env, cmd),
                     name=f"stringtie --merge ({condition})", log_path=log)

            # Merged GTFs are small, so an in-memory DB is fine here.
            db = build_gtf_db(str(tmp_gtf))
            tmp_bed = transcripts_to_bed12(db)
            if tmp_bed.count() == 0:
                raise StepError(f"stringtie --merge produced no transcripts for {condition}. See {log}")

            os.replace(tmp_gtf, merged)
            shutil.move(tmp_bed.fn, bed)

        manifest.write_text(json.dumps(expected, indent=2))
        return ConditionMerge(condition, merged, bed, len(gtfs), self._count(bed))

    # ── helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def _count(bed: Path) -> int:
        return pybedtools.BedTool(str(bed)).count()

    def _stringtie_version(self) -> str:
        proc = subprocess.run(
            with_env(self.stringtie_env, ["stringtie", "--version"]),
            capture_output=True, text=True,
        )
        return (proc.stdout or proc.stderr).strip() or "unknown"

    def _write_summary(self, results: dict[str, ConditionMerge], num_ref: int) -> Path:
        summary_file = self.reports_dir / "merge_summary.tsv"
        with summary_file.open("w") as f:
            f.write(f"# generated by {type(self).__name__} on: "
                    f"{datetime.now().astimezone().isoformat(timespec='seconds')}\n")
            f.write(f"# stringtie: {self._stringtie_version()}\n")
            f.write(f"# reference transcripts: {num_ref}\n")
            f.write("condition\tnum_inputs\tnum_transcripts\n")
            for r in results.values():
                f.write(f"{r.condition}\t{r.num_inputs}\t{r.num_transcripts}\n")
        logger.info("Find a report in %s", summary_file)
        return summary_file
