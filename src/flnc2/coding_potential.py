import os
import stat
import shutil
import tempfile
import subprocess
from pathlib import Path
from dataclasses import dataclass

from flnc2.utils.custom_logger import get_logger
from flnc2.utils.exceptions import StepError
from flnc2.utils.pipeline_step import PipelineStep
from flnc2.utils.tools import require_files, run_tool, with_env, ensure_executable, count_fasta_records

logger = get_logger()

# species alias -> (CPAT model prefix, LGC species name)
SPECIES_MODELS = {
    "mouse":        ("Mouse", "Mus_musculus"),
    "mus_musculus": ("Mouse", "Mus_musculus"),
    "human":        ("Human", "Homo_sapiens"),
    "homo_sapiens": ("Human", "Homo_sapiens"),
}

@dataclass(frozen=True)
class CodingPotentialOutputs:
    cpat_best_tsv_path: Path | None   # None if CPAT found no ORFs
    cpat_no_orf_path: Path | None     # None if every sequence had an ORF
    cpat_cutoff_path: Path
    lgc_out_path: Path
    num_sequences: int

class CodingPotential(PipelineStep[CodingPotentialOutputs]):
    def __init__(
        self,
        pureloci_fa_path: str | Path,
        external_dir: str | Path,
        output_dir: str | Path,
        species: str,
        cpat_env: str | None = None,   # conda env with CPAT; None = use PATH
        lgc_env: str | None = None,    # conda env with Python 2.7 for LGC
    ):
        # resolve() matters: tools run in a temp cwd, so relative paths would break
        self.pureloci_fa_path = Path(pureloci_fa_path).resolve()
        self.external_dir = Path(external_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.cpat_env = cpat_env
        self.lgc_env = lgc_env

        key = species.lower()
        if key not in SPECIES_MODELS:
            raise StepError(
                f"Unsupported species '{species}'. "
                f"Supported: {', '.join(sorted(SPECIES_MODELS))}"
            )
        self.cpat_prefix, self.lgc_species = SPECIES_MODELS[key]

        cpat_dir = self.external_dir / "cpat"
        self.cpat_hexamer = cpat_dir / f"{self.cpat_prefix}_Hexamer.tsv"
        self.cpat_logit = cpat_dir / f"{self.cpat_prefix}_logitModel.RData"
        self.cpat_cutoff = cpat_dir / f"{self.cpat_prefix}_cutoff.txt"

        lgc_dir = self.external_dir / "lgc2"
        self.lgc_script = lgc_dir / "LGC-2.0.py"
        self.orffinder = lgc_dir / "ORFfinder"

    def run(self) -> CodingPotentialOutputs:
        require_files(
            self.pureloci_fa_path, self.cpat_hexamer, self.cpat_logit, self.cpat_cutoff,
            self.lgc_script, self.orffinder,
        )
        if self.cpat_env is None and shutil.which("cpat") is None:
            raise StepError("cpat not found on PATH (or set cpat_env).")

        num_seqs = count_fasta_records(self.pureloci_fa_path)
        if num_seqs == 0:
            raise StepError(f"No sequences in {self.pureloci_fa_path}")

        # Both tools drop junk files (CPAT_run_info.log, simp*.temp) into their
        # working directory. Running them in a throwaway cwd replaces the old
        # cleanup trap: the junk disappears with the directory.
        with tempfile.TemporaryDirectory(prefix="flnc2_cp_") as workdir:
            best_tsv = self._run_cpat(Path(workdir), num_seqs)
            lgc_out_path = self._run_lgc(Path(workdir), num_seqs)

        no_orf = self.output_dir / "cpat" / "cpat_out.no_ORF.txt"
        return CodingPotentialOutputs(
            cpat_best_tsv_path=best_tsv,
            cpat_no_orf_path=no_orf if no_orf.is_file() and no_orf.stat().st_size > 0 else None,
            cpat_cutoff_path=self.cpat_cutoff,
            lgc_out_path=lgc_out_path,
            num_sequences=num_seqs,
        )

    # ── CPAT ────────────────────────────────────────────────────────────────
    def _run_cpat(self, workdir: Path, num_seqs: int) -> Path | None:
        logger.info("CPAT on %d sequences", num_seqs)
        out_dir = self.output_dir / "cpat"
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = out_dir / "cpat_out"

        run_tool(
            with_env(self.cpat_env, [
                "cpat",
                "-x", str(self.cpat_hexamer),
                "-d", str(self.cpat_logit),
                "-g", str(self.pureloci_fa_path),
                "-o", str(prefix),
            ]),
            name="CPAT", log_path=out_dir / "cpat.log", cwd=workdir,
        )

        best_tsv = out_dir / "cpat_out.ORF_prob.best.tsv"
        if best_tsv.is_file() and best_tsv.stat().st_size > 0:
            with best_tsv.open() as f:
                n = sum(1 for _ in f) - 1   # minus header
            logger.info("  ORFs found for %d transcripts -> %s", n, out_dir)
            return best_tsv

        logger.warning("  No %s produced", best_tsv.name)
        return None

    # ── LGC 2.0 ─────────────────────────────────────────────────────────────
    def _run_lgc(self, workdir: Path, num_seqs: int) -> Path:
        logger.info("LGC2.0 on %d sequences", num_seqs)
        out_dir = self.output_dir / "lgc2.0"
        out_dir.mkdir(parents=True, exist_ok=True)
        lgc_out_path = out_dir / "lgc_out_path.txt"

        ensure_executable(self.orffinder)

        run_tool(
            with_env(self.lgc_env, [
                "python", str(self.lgc_script),
                "-S", self.lgc_species,
                "--orffinder_path", str(self.orffinder),
                str(self.pureloci_fa_path),
                str(lgc_out_path),
            ]),
            name="LGC2.0", log_path=out_dir / "lgc.log", cwd=workdir,
        )

        if not lgc_out_path.is_file():
            raise StepError(f"LGC2.0 finished but produced no {lgc_out_path}")
        logger.info("  LGC2.0 output -> %s", lgc_out_path)
        return lgc_out_path
