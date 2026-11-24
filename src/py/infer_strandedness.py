# src/py/infer_strandedness.py
import re
import shutil
import tempfile
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.py.utils.custom_logger import get_logger
from src.py.utils.exceptions import StepError
from src.py.utils.pipeline_step import PipelineStep
from src.py.utils.samples import Sample
from src.py.utils.cache import ensure_cached
from src.py.utils.gtf_tools import cached_gtf_db, transcripts_to_bed12
from src.py.utils.tools import require_files, run_pipeline, run_tool, with_env

logger = get_logger()

# infer_experiment.py labels. "Forward" = read (or read 1) on the same strand as the gene.
_FORWARD_KEYS = {"1++,1--,2+-,2-+", "++,--"}
_REVERSE_KEYS = {"1+-,1-+,2++,2--", "+-,-+"}
_EXPLAINED = re.compile(r'explained by "([^"]+)":\s*([0-9.eE+-]+)')
_FAILED = re.compile(r"failed to determine:\s*([0-9.eE+-]+)")

@dataclass(frozen=True)
class SampleStrandedness:
    sample_id: str
    paired: bool
    frac_forward: float
    frac_reverse: float
    frac_failed: float
    strandness: str | None   # HISAT2 --rna-strandness: FR/RF (paired), F/R (single); None = unstranded
    report_path: Path

@dataclass(frozen=True)
class StrandednessOutputs:
    samples: dict[str, SampleStrandedness]
    annotation_bed: Path

class InferStrandedness(PipelineStep[StrandednessOutputs]):
    def __init__(
        self,
        samples: list[Sample],
        index_prefix: str | Path,
        gtf_path: str | Path,
        dataset: str,
        cache_dir: str | Path,
        output_dir: str | Path,
        threads: int = 1,
        splice_sites: str | Path | None = None,
        num_reads: int = 500_000,
        seed: int = 100,
        stranded_threshold: float = 0.8,
        hisat2_env: str | None = None,   # hisat2, samtools, seqtk
        rseqc_env: str | None = None,    # infer_experiment.py
    ):
        if not samples:
            raise StepError("No samples given")
        self.samples = samples
        self.index_prefix = Path(index_prefix)
        self.gtf_path = Path(gtf_path).resolve()
        self.dataset = dataset
        self.cache_dir = Path(cache_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.threads = int(threads)
        self.splice_sites = Path(splice_sites) if splice_sites is not None else None
        self.num_reads = int(num_reads)
        self.seed = int(seed)
        self.stranded_threshold = float(stranded_threshold)
        self.hisat2_env = hisat2_env
        self.rseqc_env = rseqc_env

    def run(self) -> StrandednessOutputs:
        require_files(self.gtf_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        bed = self.cache_dir / "rseqc" / f"{self.dataset}.rseqc.bed12"
        if ensure_cached(bed, {"gtf": self.gtf_path}, self._build_annotation_bed):
            logger.info("Created RSeQC annotation %s", bed)
        else:
            logger.info("Using cached RSeQC annotation %s", bed)

        # Split the thread budget across concurrent samples instead of giving
        # every sample all of it.
        max_jobs = max(1, min(len(self.samples), self.threads))
        per_job = max(1, self.threads // max_jobs)
        logger.info("Inferring strandedness for %d sample(s), %d at a time, %d thread(s) each",
                    len(self.samples), max_jobs, per_job)

        results: dict[str, SampleStrandedness] = {}
        with ThreadPoolExecutor(max_workers=max_jobs) as pool:
            futures = {pool.submit(self._process_sample, s, bed, per_job): s for s in self.samples}
            try:
                for fut in as_completed(futures):
                    r = fut.result()
                    results[r.sample_id] = r
            except BaseException:
                pool.shutdown(wait=True, cancel_futures=True)
                raise

        self._report(results)
        # Keep the sample-sheet order.
        ordered = {s.sample_id: results[s.sample_id] for s in self.samples}
        return StrandednessOutputs(samples=ordered, annotation_bed=bed)

    # ── per sample ──────────────────────────────────────────────────────────
    def _process_sample(self, sample: Sample, bed: Path, threads: int) -> SampleStrandedness:
        sid = sample.sample_id
        sdir = self.output_dir / sid
        logs = sdir / "logs"
        sdir.mkdir(parents=True, exist_ok=True)
        report = sdir / f"{sid}_strandedness.txt"

        with tempfile.TemporaryDirectory(dir=sdir, prefix=".tmp_") as tmpdir:
            tmp = Path(tmpdir)

            logger.info("Sampling %d reads for %s (%s)",
                        self.num_reads, sid, "paired-end" if sample.paired else "single-end")
            # Same seed on both mates keeps pairs in sync, as long as both
            # files list reads in the same order (true for normal FASTQ pairs).
            mates = [sample.r1] + ([sample.r2] if sample.paired else [])
            sub = []
            for n, fq in enumerate(mates, start=1):
                out = tmp / f"r{n}.fq"
                run_tool(
                    with_env(self.hisat2_env,
                             ["seqtk", "sample", f"-s{self.seed}", str(fq), str(self.num_reads)]),
                    name="seqtk", log_path=logs / f"seqtk_r{n}.log", stdout_path=out,
                )
                sub.append(out)

            logger.info("Aligning %s", sid)
            hisat2 = ["hisat2", "-p", str(threads), "-x", str(self.index_prefix), "--no-unal"]
            if self.splice_sites is not None:
                hisat2 += ["--known-splicesite-infile", str(self.splice_sites)]
            hisat2 += ["-1", str(sub[0]), "-2", str(sub[1])] if sample.paired else ["-U", str(sub[0])]

            bam = tmp / "sample.sorted.bam"
            sort = ["samtools", "sort", "-@", str(threads), "-m", "512M",
                    "-T", str(tmp / "sort"), "-o", str(bam), "-"]
            run_pipeline(
                [with_env(self.hisat2_env, hisat2), with_env(self.hisat2_env, sort)],
                name=f"hisat2 | samtools sort ({sid})", log_path=logs / "align.log",
            )
            run_tool(with_env(self.hisat2_env, ["samtools", "index", str(bam)]),
                     name="samtools index", log_path=logs / "index.log")

            logger.info("Running infer_experiment for %s", sid)
            run_tool(
                with_env(self.rseqc_env,
                         ["infer_experiment.py", "-i", str(bam), "-r", str(bed)]),
                name="infer_experiment.py", log_path=logs / "infer_experiment.log",
                stdout_path=report,
            )
        # tmp (subsampled reads, BAM) is deleted here; only the report and logs remain.

        return self._parse_report(sample, report)

    # ── parsing ─────────────────────────────────────────────────────────────
    def _parse_report(self, sample: Sample, report: Path) -> SampleStrandedness:
        text = report.read_text()
        fwd = rev = None
        for key, value in _EXPLAINED.findall(text):
            if key in _FORWARD_KEYS:
                fwd = float(value)
            elif key in _REVERSE_KEYS:
                rev = float(value)
        failed_m = _FAILED.search(text)
        if fwd is None or rev is None or failed_m is None:
            raise StepError(
                f"Could not parse infer_experiment output for {sample.sample_id}: {report}")

        determinable = fwd + rev
        if determinable == 0:
            raise StepError(
                f"No reads overlapped annotated genes for {sample.sample_id}. "
                f"Does the GTF match the genome? See {report}")

        # Compare forward vs reverse among reads that could be assigned, so a
        # large "failed" fraction doesn't hide a clearly stranded library.
        share_fwd = fwd / determinable
        if share_fwd >= self.stranded_threshold:
            strandness = "FR" if sample.paired else "F"
        elif share_fwd <= 1 - self.stranded_threshold:
            strandness = "RF" if sample.paired else "R"
        else:
            strandness = None

        return SampleStrandedness(
            sample_id=sample.sample_id, paired=sample.paired,
            frac_forward=fwd, frac_reverse=rev, frac_failed=float(failed_m.group(1)),
            strandness=strandness, report_path=report,
        )

    def _report(self, results: dict[str, SampleStrandedness]) -> None:
        for r in results.values():
            logger.info("  %s: forward %.3f, reverse %.3f, failed %.3f -> %s",
                        r.sample_id, r.frac_forward, r.frac_reverse, r.frac_failed,
                        r.strandness or "unstranded")
            share = r.frac_forward / (r.frac_forward + r.frac_reverse)
            if r.strandness is None and not 0.4 <= share <= 0.6:
                logger.warning("  %s is ambiguous (%.0f%% forward); treated as unstranded",
                               r.sample_id, 100 * share)

        # Mixed protocols in one experiment are usually a sample-sheet mistake.
        calls = {r.strandness for r in results.values()}
        if len(calls) > 1:
            logger.warning("Samples disagree on strandedness: %s",
                           ", ".join(sorted(c or "unstranded" for c in calls)))

    # # ── annotation ──────────────────────────────────────────────────────────
    def _build_annotation_bed(self, dest: Path) -> None:
        db = cached_gtf_db(self.gtf_path, self.cache_dir)
        bed = transcripts_to_bed12(db)
        shutil.move(bed.fn, dest)
