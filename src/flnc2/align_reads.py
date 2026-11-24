# src/py/align_reads.py
import os
import re
import json
import tempfile
from pathlib import Path
from dataclasses import dataclass

from flnc2.utils.custom_logger import get_logger
from flnc2.utils.exceptions import StepError
from flnc2.utils.pipeline_step import PipelineStep
from flnc2.utils.cache import file_fingerprint
from flnc2.utils.samples import Sample
from flnc2.utils.tools import run_pipeline, run_tool, with_env

logger = get_logger()

_ALIGN_RATE = re.compile(r"([0-9.]+)% overall alignment rate")
_VALID_STRANDNESS = {"FR", "RF", "F", "R"}

# Counts used as the FPKM denominator. PE: concordant pairs. SE: reads.
_UNIQUE = {
    True:  re.compile(r"(\d+) \([\d.]+%\) aligned concordantly exactly 1 time"),
    False: re.compile(r"(\d+) \([\d.]+%\) aligned exactly 1 time"),
}
_MULTI = {
    True:  re.compile(r"(\d+) \([\d.]+%\) aligned concordantly >1 times"),
    False: re.compile(r"(\d+) \([\d.]+%\) aligned >1 times"),
}

def normalize_strandness(value: str | None, paired: bool) -> str | None:
    """Return the --rna-strandness value for this layout, or None if unstranded.
    Accepts FR/RF/F/R (any case), or None/''/'none'/'unstranded'."""
    if value is None or value.strip().lower() in ("", "none", "unstranded"):
        return None
    v = value.strip().upper()
    if v not in _VALID_STRANDNESS:
        raise StepError(f"Invalid strandedness {value!r}; expected FR, RF, F, R, or unstranded")
    # HISAT2 wants two letters for paired, one for single. Convert if needed,
    # e.g. a global "RF" override applied to a single-end sample becomes "R".
    if paired and len(v) == 1:
        return {"F": "FR", "R": "RF"}[v]
    if not paired and len(v) == 2:
        return v[0]
    return v

@dataclass(frozen=True)
class SampleAlignment:
    sample_id: str
    bam: Path
    log: Path
    strandness: str | None
    paired: bool
    alignment_rate: float | None    # percent
    unique_aligned: int | None      # pairs (PE) or reads (SE)
    multi_aligned: int | None

@dataclass(frozen=True)
class AlignmentOutputs:
    samples: dict[str, SampleAlignment]

class AlignReads(PipelineStep[AlignmentOutputs]):
    def __init__(
        self,
        samples: list[Sample],
        index_prefix: str | Path,
        strandness: dict[str, str | None],
        output_dir: str | Path,
        threads: int = 1,
        splice_sites: str | Path | None = None,
        dta: bool = False,
        sort_threads: int = 8,
        sort_mem_per_thread: str = "1G",
        min_alignment_rate: float = 50.0,
        hisat2_env: str | None = None,
    ):
        if not samples:
            raise StepError("No samples given")
        missing = [s.sample_id for s in samples if s.sample_id not in strandness]
        if missing:
            raise StepError("No strandedness for sample(s): " + ", ".join(missing))

        self.samples = samples
        self.index_prefix = Path(index_prefix)
        self.strandness = {
            s.sample_id: normalize_strandness(strandness[s.sample_id], s.paired)
            for s in samples
        }
        self.output_dir = Path(output_dir).resolve()
        self.threads = int(threads)
        self.splice_sites = Path(splice_sites) if splice_sites is not None else None
        self.dta = dta
        self.sort_threads = max(1, min(int(sort_threads), self.threads))
        self.sort_mem_per_thread = sort_mem_per_thread
        self.min_alignment_rate = float(min_alignment_rate)
        self.hisat2_env = hisat2_env

    def run(self) -> AlignmentOutputs:
        logger.info("HISAT2 alignment")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Sequential on purpose: each hisat2 process loads the full index into
        # memory, and hisat2 already scales well across threads.
        results = {s.sample_id: self._align(s) for s in self.samples}

        low = [r for r in results.values()
               if r.alignment_rate is not None and r.alignment_rate < self.min_alignment_rate]
        for r in low:
            logger.warning("  %s aligned at only %.1f%%. Wrong genome, contamination, "
                           "or adapter problems? See %s", r.sample_id, r.alignment_rate, r.log)
        return AlignmentOutputs(samples=results)

    # ── per sample ──────────────────────────────────────────────────────────
    def _align(self, sample: Sample) -> SampleAlignment:
        sid = sample.sample_id
        sdir = self.output_dir / sid
        sdir.mkdir(parents=True, exist_ok=True)
        bam = sdir / f"{sid}.sorted.bam"
        bai = Path(f"{bam}.bai")
        log = sdir / f"{sid}.hisat2.log"
        manifest = sdir / f"{sid}.alignment.manifest.json"
        strand = self.strandness[sid]

        expected = self._fingerprint(sample, strand)
        if self._is_done(bam, bai, manifest, expected):
            logger.info("  Using existing alignment for %s", sid)
            rate, unique, multi = self._parse_log(log, sample.paired)
            return SampleAlignment(sid, bam, log, strand, sample.paired, rate, unique, multi)

        manifest.unlink(missing_ok=True)
        logger.info("  Aligning %s (%s, strandness: %s)",
                    sid, "paired-end" if sample.paired else "single-end", strand or "unstranded")

        hisat2 = ["hisat2", "-p", str(self.threads), "-x", str(self.index_prefix)]
        if strand is not None:
            hisat2 += ["--rna-strandness", strand]
        if self.splice_sites is not None:
            hisat2 += ["--known-splicesite-infile", str(self.splice_sites)]
        if self.dta:
            hisat2 += ["--dta"]
        hisat2 += (["-1", str(sample.r1), "-2", str(sample.r2)] if sample.paired
                   else ["-U", str(sample.r1)])

        # Sort into a temp file and rename on success, so an interrupted run
        # never leaves a truncated BAM under the final name.
        with tempfile.TemporaryDirectory(dir=sdir, prefix=".tmp_") as tmpdir:
            tmp_bam = Path(tmpdir) / bam.name
            sort = ["samtools", "sort",
                    "-@", str(self.sort_threads), "-m", self.sort_mem_per_thread,
                    "-T", str(Path(tmpdir) / "sort"), "-o", str(tmp_bam), "-"]
            run_pipeline(
                [with_env(self.hisat2_env, hisat2), with_env(self.hisat2_env, sort)],
                name=f"hisat2 | samtools sort ({sid})", log_path=log,
            )
            run_tool(with_env(self.hisat2_env, ["samtools", "index", str(tmp_bam)]),
                     name="samtools index", log_path=sdir / f"{sid}.index.log")
            os.replace(tmp_bam, bam)
            os.replace(f"{tmp_bam}.bai", bai)

        manifest.write_text(json.dumps(expected, indent=2))
        rate, unique, multi = self._parse_log(log, sample.paired)
        return SampleAlignment(sid, bam, log, strand, sample.paired, rate, unique, multi)

    # ── resume support ──────────────────────────────────────────────────────
    def _fingerprint(self, sample: Sample, strand: str | None) -> dict:
        index_file = next(
            (Path(f"{self.index_prefix}.1.{ext}") for ext in ("ht2", "ht2l")
             if Path(f"{self.index_prefix}.1.{ext}").is_file()),
            None,
        )
        if index_file is None:
            raise StepError(f"No HISAT2 index found at prefix {self.index_prefix}")
        return {
            "r1": file_fingerprint(sample.r1),
            "r2": file_fingerprint(sample.r2) if sample.r2 else None,
            "index": file_fingerprint(index_file),
            "splice_sites": file_fingerprint(self.splice_sites) if self.splice_sites else None,
            "strandness": strand,
            "dta": self.dta,
        }

    @staticmethod
    def _is_done(bam: Path, bai: Path, manifest: Path, expected: dict) -> bool:
        if not (bam.is_file() and bai.is_file() and manifest.is_file()):
            return False
        try:
            return json.loads(manifest.read_text()) == expected
        except json.JSONDecodeError:
            return False

    @staticmethod
    def _parse_log(log: Path, paired: bool) -> tuple[float | None, int | None, int | None]:
        if not log.is_file():
            return None, None, None
        text = log.read_text()

        def first_int(pattern):
            m = pattern.search(text)
            return int(m.group(1)) if m else None

        rate = _ALIGN_RATE.search(text)
        return (float(rate.group(1)) if rate else None,
                first_int(_UNIQUE[paired]),
                first_int(_MULTI[paired]))
