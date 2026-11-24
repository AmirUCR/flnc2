# src/py/utils/tools.py
import os
import stat
import json
import shlex
import shutil
import pyfaidx
import contextlib
import subprocess
from pathlib import Path
from functools import lru_cache
from typing import Mapping, Optional

from src.py.utils.exceptions import StepError

def with_env(env: str | None, cmd: list[str]) -> list[str]:
    """Prefix a command with `conda run` if an env name is given."""
    if env is None:
        return cmd
    return ["conda", "run", "--no-capture-output", "-n", env, *cmd]

def run_tool(
    cmd: list[str],
    name: str,
    log_path: Path,
    cwd: Path | None = None,
    stdout_path: Path | None = None,
) -> None:
    """Run an external tool. stderr (and stdout, unless redirected to
    stdout_path) goes to log_path. Raises StepError on nonzero exit."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_f:
        if stdout_path is not None:
            with stdout_path.open("w") as out_f:
                proc = subprocess.run(cmd, cwd=cwd, stdout=out_f, stderr=log_f)
        else:
            proc = subprocess.run(cmd, cwd=cwd, stdout=log_f, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
                how = (f"was killed by signal {-proc.returncode}" if proc.returncode < 0
                    else f"failed with exit status {proc.returncode}")
                raise StepError(f"{name} {how}. See {log_path}")
                
def require_files(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise StepError("Required file(s) not found: " + ", ".join(missing))

def ensure_executable(path: Path) -> None:
    mode = path.stat().st_mode
    if not mode & stat.S_IXUSR:
        path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

def count_fasta_records(path: Path) -> int:
    with path.open() as f:
        return sum(1 for line in f if line.startswith(">"))

def dataset_name(fasta_path: str | Path) -> str:
    """mm39.fa.gz -> mm39, GRCh38.primary_assembly.genome.fasta -> GRCh38.primary_assembly.genome"""
    name = Path(fasta_path).name
    for suffix in (".gz", ".bgz"):
        name = name.removesuffix(suffix)
    for suffix in (".fa", ".fasta", ".fna"):
        name = name.removesuffix(suffix)
    return name

def manifest_matches(manifest: Path, expected: dict) -> bool:
        if not manifest.is_file():
            return False
        try:
            return json.loads(manifest.read_text()) == expected
        except json.JSONDecodeError:
            return False

def run_pipeline(
    cmds: list[list[str]],
    name: str,
    log_path: Path,
    stdout_path: Path | None = None,
) -> None:
    """Run cmds[0] | cmds[1] | ... ; all stderr goes to log_path."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as stack:
        log_f = stack.enter_context(log_path.open("w"))
        final_out = stack.enter_context(stdout_path.open("wb")) if stdout_path else log_f

        procs = []
        prev_stdout = None
        for i, cmd in enumerate(cmds):
            last = i == len(cmds) - 1
            p = subprocess.Popen(
                cmd,
                stdin=prev_stdout,
                stdout=final_out if last else subprocess.PIPE,
                stderr=log_f,
            )
            if prev_stdout is not None:
                prev_stdout.close()   # so upstream gets SIGPIPE if downstream dies
            prev_stdout = p.stdout
            procs.append(p)

        codes = [p.wait() for p in procs]

    failed = [f"{shlex.join(c)[:60]}... (exit {rc})" for c, rc in zip(cmds, codes) if rc != 0]
    if failed:
        raise StepError(f"{name} failed: {'; '.join(failed)}. See {log_path}")

@lru_cache(maxsize=1)
def _conda_envs() -> tuple:
    """Paths of all conda envs, or an empty tuple if conda is unavailable."""
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not conda:
        return ()
    try:
        result = subprocess.run(
            [conda, "env", "list", "--json"],
            capture_output=True, text=True, check=True,
        )
        return tuple(json.loads(result.stdout).get("envs", []))
    except (subprocess.CalledProcessError, json.JSONDecodeError, OSError):
        return ()

def conda_env_path(name: str) -> Optional[str]:
    """Return the path of the conda env with this name (or path), or None."""
    target = os.path.abspath(os.path.expanduser(name))
    for env_path in _conda_envs():
        if os.path.basename(env_path) == name or os.path.abspath(env_path) == target:
            return env_path
    return None

def conda_env_exists(name: str) -> bool:
    return conda_env_path(name) is not None

def which_in_conda_env(binary_name: str, env_path: str) -> Optional[str]:
    """Like shutil.which, but only searches inside the given conda env."""
    if os.name == "nt":
        dirs = [env_path, os.path.join(env_path, "Scripts"), os.path.join(env_path, "Library", "bin")]
    else:
        dirs = [os.path.join(env_path, "bin")]
    return shutil.which(binary_name, path=os.pathsep.join(dirs))

def require_on_path_iterable(requirements: Mapping[str, Optional[str]]) -> None:
    """Check each executable in its conda env, or on the current PATH if no env is given."""
    missing = []
    for binary_name, env in requirements.items():
        if env in (None, ""):
            if shutil.which(binary_name) is None:
                missing.append(f"'{binary_name}' (not on your PATH)")
        else:
            env_path = conda_env_path(env)
            if env_path is None or which_in_conda_env(binary_name, env_path) is None:
                missing.append(f"'{binary_name}' (not in conda env '{env}')")

    if missing:
        raise StepError("Error: Required executables not found:\n  " + "\n  ".join(missing))

def require_fai(genome_fasta: Path) -> Path:
    """Return <genome>.fai, creating it if missing and the directory is writable."""
    fai = Path(f"{genome_fasta}.fai")
    if fai.is_file():
        return fai
    if not os.access(genome_fasta.parent, os.W_OK):
        raise StepError(
            f"{fai} is missing and {genome_fasta.parent} is not writable. "
            f"Run 'samtools faidx {genome_fasta}' where you have write access, "
            f"or copy the genome somewhere writable.")
    pyfaidx.Faidx(str(genome_fasta))   # writes the .fai
    return fai
