# src/py/assemble_transcripts.py
import os
import json
import shlex
import tempfile
from pathlib import Path
from dataclasses import dataclass

from src.py.utils.custom_logger import get_logger
from src.py.utils.exceptions import StepError
from src.py.utils.pipeline_step import PipelineStep
from src.py.utils.cache import file_fingerprint
from src.py.utils.tools import require_files, run_tool, with_env, manifest_matches
from src.py.align_reads import AlignmentOutputs

logger = get_logger()

# HISAT2 --rna-strandness -> StringTie library flag.
_STRINGTIE_STRAND_FLAG = {"FR": "--fr", "F": "--fr", "RF": "--rf", "R": "--rf"}

# Options this step sets itself; letting users pass them too would conflict.
_RESERVED_ARGS = {"-o", "-G", "-p", "--fr", "--rf", "--merge"}

@dataclass(frozen=True)
class SampleAssembly:
    sample_id: str
    gtf: Path
    log: Path
    strand_flag: str | None   # --fr, --rf, or None for unstranded

@dataclass(frozen=True)
class AssemblyOutputs:
    samples: dict[str, SampleAssembly]

    @property
    def gtfs(self) -> list[Path]:
        """Per-sample assemblies, in sample order. Input for stringtie --merge."""
        return [a.gtf for a in self.samples.values()]

class AssembleTranscripts(PipelineStep[AssemblyOutputs]):
    def __init__(
        self,
        alignments: AlignmentOutputs,
        output_dir: str | Path,
        gtf_path: str | Path | None = None,
        threads: int = 1,
        extra_args: list[str] | str | None = None,
        stringtie_env: str | None = None,
    ):
        if not alignments.samples:
            raise StepError("No alignments given")
        self.alignments = alignments
        self.output_dir = Path(output_dir).resolve()
        self.gtf_path = Path(gtf_path).resolve() if gtf_path is not None else None
        self.threads = int(threads)
        self.extra_args = self._parse_extra_args(extra_args)
        self.stringtie_env = stringtie_env

    @staticmethod
    def _parse_extra_args(extra_args: list[str] | str | None) -> list[str]:
        if extra_args is None:
            return []
        # Accept a YAML list or a single string like "-m 200 -c 2.5".
        args = shlex.split(extra_args) if isinstance(extra_args, str) else [str(a) for a in extra_args]
        clash = sorted({a for a in args if a in _RESERVED_ARGS})
        if clash:
            raise StepError(
                f"StringTie option(s) {', '.join(clash)} are set by Flnc2 and "
                f"can't be passed as extra arguments"
            )
        return args

    def run(self) -> AssemblyOutputs:
        logger.info("Transcript assembly")

        if self.gtf_path is not None:
            require_files(self.gtf_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            sid: self._assemble(sid, aln.bam, aln.strandness)
            for sid, aln in self.alignments.samples.items()
        }
        return AssemblyOutputs(samples=results)

    def _assemble(self, sid: str, bam: Path, strandness: str | None) -> SampleAssembly:
        require_files(bam)
        sdir = self.output_dir / sid
        sdir.mkdir(parents=True, exist_ok=True)
        out_gtf = sdir / f"{sid}_stringtie.gtf"
        log = sdir / f"{sid}.stringtie.log"
        manifest = sdir / f"{sid}.assembly.manifest.json"
        strand_flag = _STRINGTIE_STRAND_FLAG.get(strandness) if strandness else None

        expected = {
            "bam": file_fingerprint(bam),
            "gtf": file_fingerprint(self.gtf_path) if self.gtf_path else None,
            "strand_flag": strand_flag,
            "extra_args": self.extra_args,
        }
        if out_gtf.is_file() and manifest_matches(manifest, expected):
            logger.info("  Using existing assembly for %s", sid)
            return SampleAssembly(sid, out_gtf, log, strand_flag)

        manifest.unlink(missing_ok=True)
        logger.info("  Assembling %s (%s)", sid, strand_flag or "unstranded")

        with tempfile.TemporaryDirectory(dir=sdir, prefix=".tmp_") as tmpdir:
            tmp_gtf = Path(tmpdir) / out_gtf.name
            cmd = ["stringtie", *self.extra_args, "-p", str(self.threads)]
            if self.gtf_path is not None:
                cmd += ["-G", str(self.gtf_path)]
            if strand_flag is not None:
                cmd.append(strand_flag)
            cmd += ["-o", str(tmp_gtf), str(bam)]

            run_tool(with_env(self.stringtie_env, cmd), name=f"stringtie ({sid})", log_path=log)

            if not self._has_records(tmp_gtf):
                logger.warning("StringTie assembled no transcripts for %s. See %s", sid, log)
            os.replace(tmp_gtf, out_gtf)

        manifest.write_text(json.dumps(expected, indent=2))
        return SampleAssembly(sid, out_gtf, log, strand_flag)

    @staticmethod
    def _has_records(gtf: Path) -> bool:
        # StringTie writes "#" header lines even when it assembles nothing.
        if not gtf.is_file():
            return False
        with gtf.open() as f:
            return any(line.strip() and not line.startswith("#") for line in f)
