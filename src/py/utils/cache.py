# src/py/utils/cache.py
import os
import json
from pathlib import Path
from typing import Callable

from src.py.utils.exceptions import StepError

def file_fingerprint(p: Path) -> dict:
    s = p.stat()
    return {"path": str(p.resolve()), "size": s.st_size, "mtime_ns": s.st_mtime_ns}

def ensure_cached(
    dest: Path,
    inputs: dict[str, Path],
    build: Callable[[Path], None],
    extra: dict | None = None,
) -> bool:
    """Build `dest` via build(tmp_path) unless it exists and its inputs are
    unchanged. Returns True if it was (re)built."""
    manifest = dest.with_name(dest.name + ".manifest.json")
    expected = {k: file_fingerprint(p) for k, p in inputs.items()} | (extra or {})

    if dest.is_file() and dest.stat().st_size > 0 and manifest.is_file():
        try:
            if json.loads(manifest.read_text()) == expected:
                return False
        except json.JSONDecodeError:
            pass

    dest.parent.mkdir(parents=True, exist_ok=True)
    manifest.unlink(missing_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.unlink(missing_ok=True)   # leftover from an interrupted build
    build(tmp)

    if not tmp.is_file() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise StepError(f"Building {dest.name} produced no output")
    os.replace(tmp, dest)
    manifest.write_text(json.dumps(expected, indent=2))
    return True
