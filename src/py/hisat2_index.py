import os
import re
import json
import tempfile
from pathlib import Path
from dataclasses import dataclass

from src.py.utils.custom_logger import get_logger
from src.py.utils.exceptions import StepError
from src.py.utils.pipeline_step import PipelineStep
from src.py.utils.tools import require_files, run_tool, with_env, manifest_matches
from src.py.utils.cache import file_fingerprint, ensure_cached

logger = get_logger()

MANIFEST_NAME = "flnc2_hisat2_index_manifest.json"
_INDEX_FILE_SUFFIX = re.compile(r"\.\d+\.ht2l?$")

@dataclass(frozen=True)
class Hisat2IndexOutputs:
    index_prefix_path: Path
    splice_sites_path: Path | None   # pass to hisat2 --known-splicesite-infile; None if no GTF
    exons_path: Path | None          # only produced for use_splice_sites builds

def index_files_complete(prefix: Path) -> bool:
    # hisat2-build writes 8 files: .ht2 normally, .ht2l for large genomes.
    for ext in ("ht2", "ht2l"):
        files = [Path(f"{prefix}.{i}.{ext}") for i in range(1, 9)]
        if all(f.is_file() and f.stat().st_size > 0 for f in files):
            return True
    return False

def _nonempty(p: Path) -> bool:
    return p.is_file() and p.stat().st_size > 0

class BuildHisat2Index(PipelineStep[Hisat2IndexOutputs]):
    def __init__(
        self,
        genome_fasta_path: str | Path | None = None,
        gtf_path: str | Path | None = None,
        index_dir: str | Path | None = None,
        dataset: str | None = None,
        threads: int = 1,
        use_splice_sites: bool = False,
        hisat2_env: str | None = None,
        prebuilt_index_prefix: str | Path | None = None,
    ):
        self.threads = int(threads)
        self.use_splice_sites = use_splice_sites
        self.hisat2_env = hisat2_env
        self.gtf_path = Path(gtf_path).resolve() if gtf_path is not None else None
        self.index_dir = Path(index_dir).resolve() if index_dir is not None else None
        self.prebuilt_index_prefix = (
            self._normalize_prefix(prebuilt_index_prefix)
            if prebuilt_index_prefix is not None else None
        )

        if self.prebuilt_index_prefix is not None:
            if self.use_splice_sites:
                logger.warning(
                    "  use_splice_sites only affects building; ignored for a prebuilt index")
                self.use_splice_sites = False
            # A GTF is optional here, but if given we need somewhere writable
            # for the .ss file, since the prebuilt index dir may be read-only.
            if self.gtf_path is not None and self.index_dir is None:
                raise StepError(
                    "gtf_path was given with prebuilt_index_prefix, so index_dir is "
                    "required as the place to write the splice-site file"
                )
            self.dataset = dataset or (self.gtf_path.stem if self.gtf_path else None)
        else:
            missing = [name for name, value in (
                ("genome_fasta_path", genome_fasta_path),
                ("gtf_path", gtf_path),
                ("index_dir", index_dir),
                ("dataset", dataset),
            ) if value is None]
            if missing:
                raise StepError(
                    "Building a HISAT2 index requires: " + ", ".join(missing)
                    + " (or provide prebuilt_index_prefix)"
                )
            self.genome_fasta_path = Path(genome_fasta_path).resolve()
            self.dataset = dataset
            self.index_prefix_path = self.index_dir / dataset
            self.manifest = self.index_dir / MANIFEST_NAME

        if self.index_dir is not None and self.dataset is not None:
            self.splice_sites_path = self.index_dir / f"{self.dataset}.ss"
            self.splice_sites_manifest = self.index_dir / f"{self.dataset}.ss.manifest.json"
            self.exons_path = self.index_dir / f"{self.dataset}.exon"
            self.log_dir = self.index_dir / "logs"

    def run(self) -> Hisat2IndexOutputs:
        logger.info("HISAT2 Indexing")

        if self.gtf_path is not None:
            require_files(self.gtf_path)
            self.index_dir.mkdir(parents=True, exist_ok=True)

        # Splice sites first: cheap, independent of the index source, and
        # needed by the build when use_splice_sites is on.
        splice_sites = self._ensure_splice_sites() if self.gtf_path is not None else None

        if self.prebuilt_index_prefix is not None:
            prefix = self._use_prebuilt()
            exons = None
        else:
            prefix = self._ensure_built_index()
            exons = self.exons_path if self.use_splice_sites else None

        return Hisat2IndexOutputs(
            index_prefix_path=prefix,
            splice_sites_path=splice_sites,
            exons_path=exons,
        )

    # ── splice sites (always, when a GTF is available) ──────────────────────
    def _ensure_splice_sites(self) -> Path:
        expected = {"gtf_path": file_fingerprint(self.gtf_path)}
        if _nonempty(self.splice_sites_path) and manifest_matches(
            self.splice_sites_manifest, expected
        ):
            logger.info("  Using cached splice sites at %s", self.splice_sites_path)
            return self.splice_sites_path

        self.splice_sites_manifest.unlink(missing_ok=True)
        self._extract(
            "hisat2_extract_splice_sites.py", self.splice_sites_path, "extract_splice_sites")
        self.splice_sites_manifest.write_text(json.dumps(expected, indent=2))
        return self.splice_sites_path

    # ── prebuilt index ──────────────────────────────────────────────────────
    @staticmethod
    def _normalize_prefix(prefix: str | Path) -> Path:
        # Accept "idx/mm39", or a file like "idx/mm39.1.ht2" and strip it to the prefix.
        p = Path(prefix).expanduser().resolve()
        return p.with_name(_INDEX_FILE_SUFFIX.sub("", p.name))

    def _use_prebuilt(self) -> Path:
        prefix = self.prebuilt_index_prefix
        if not index_files_complete(prefix):
            raise StepError(
                f"No complete HISAT2 index at prefix {prefix}. "
                f"Expected {prefix}.1.ht2 ... {prefix}.8.ht2 (or .ht2l)."
            )
        logger.info("  Using user-provided HISAT2 index at %s", prefix)
        return prefix

    # ── built index ─────────────────────────────────────────────────────────
    def _ensure_built_index(self) -> Path:
        require_files(self.genome_fasta_path)
        expected = {
            "genome_fasta_path": file_fingerprint(self.genome_fasta_path),
            "gtf_path": file_fingerprint(self.gtf_path),
            "use_splice_sites": self.use_splice_sites,
        }

        if self._index_cache_is_valid(expected):
            logger.info("  Using cached HISAT2 index at %s", self.index_prefix_path)
            return self.index_prefix_path

        if self.manifest.exists():
            logger.warning("  HISAT2 index inputs changed since last build, rebuilding")
        # Invalidate first, so a crash mid-rebuild can't leave a manifest
        # that vouches for a half-replaced index.
        self.manifest.unlink(missing_ok=True)

        if self.use_splice_sites:
            self._extract("hisat2_extract_exons.py", self.exons_path, "extract_exons")
        self._build_index()

        # Written last: its presence means everything above completed.
        self.manifest.write_text(json.dumps(expected, indent=2))
        logger.info("  HISAT2 index ready at %s", self.index_prefix_path)
        return self.index_prefix_path

    def _index_cache_is_valid(self, expected: dict) -> bool:
        if not manifest_matches(self.manifest, expected):
            return False
        if not index_files_complete(self.index_prefix_path):
            return False
        if self.use_splice_sites and not _nonempty(self.exons_path):
            return False
        return True

    # ── shared helpers ──────────────────────────────────────────────────────
    def _extract(self, script: str, dest: Path, log_name: str) -> None:
        logger.info("Generating %s", dest.name)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        run_tool(
            with_env(self.hisat2_env, [script, str(self.gtf_path)]),
            name=script,
            log_path=self.log_dir / f"{log_name}.log",
            stdout_path=tmp,
        )
        if tmp.stat().st_size == 0:
            tmp.unlink()
            raise StepError(f"{script} produced no output from {self.gtf_path}")
        os.replace(tmp, dest)

    def _build_index(self) -> None:
        logger.info("  Running hisat2-build with %d threads (this can take hours)", self.threads)

        # Build into a temp dir on the same filesystem, then move the finished
        # files into place, so an interrupted build never leaves partial files
        # under the real prefix.
        with tempfile.TemporaryDirectory(dir=self.index_dir, prefix=".build_") as tmpdir:
            tmp_prefix = Path(tmpdir) / self.dataset
            cmd = ["hisat2-build", "-p", str(self.threads)]
            if self.use_splice_sites:
                cmd += ["--ss", str(self.splice_sites_path), "--exon", str(self.exons_path)]
            cmd += [str(self.genome_fasta_path), str(tmp_prefix)]

            run_tool(
                with_env(self.hisat2_env, cmd),
                name="hisat2-build",
                log_path=self.log_dir / "hisat2_build.log",
            )

            # Remove stale index files from a previous build (either extension).
            for old in self.index_dir.glob(f"{self.dataset}.*.ht2*"):
                old.unlink()
            for built in Path(tmpdir).glob(f"{self.dataset}.*.ht2*"):
                os.replace(built, self.index_dir / built.name)

        if not index_files_complete(self.index_prefix_path):
            raise StepError(
                f"hisat2-build finished but index files are incomplete at {self.index_prefix_path}"
            )
        