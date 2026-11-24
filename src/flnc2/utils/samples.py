# src/py/utils/samples.py
import re
from pathlib import Path
from dataclasses import dataclass

import pandas as pd

from flnc2.utils.exceptions import StepError

_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")   # used as a directory name

@dataclass(frozen=True)
class Sample:
    sample_id: str
    condition: str
    r1: Path
    r2: Path | None = None

    @property
    def paired(self) -> bool:
        return self.r2 is not None

def load_samples(path: str | Path) -> list[Sample]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise StepError(f"Cannot find samples file: {path}")

    df = pd.read_table(path, dtype=str, comment="#").fillna("")
    missing = {"sample_id", "condition", "r1"} - set(df.columns)
    if missing:
        raise StepError(f"Samples file missing column(s): {', '.join(sorted(missing))}")
    if "r2" not in df.columns:
        df["r2"] = ""

    def resolve(value: str) -> Path:
        p = Path(value.strip()).expanduser()
        return (p if p.is_absolute() else path.parent / p).resolve()

    samples: list[Sample] = []
    errors: list[str] = []
    seen: set[str] = set()

    for i, row in enumerate(df.itertuples(index=False), start=2):   # line 1 is the header
        sid = row.sample_id.strip()
        if not _SAFE_ID.match(sid):
            errors.append(f"line {i}: invalid sample_id {sid!r} (use letters, digits, . _ -)")
            continue
        if sid in seen:
            errors.append(f"line {i}: duplicate sample_id {sid!r}")
            continue
        seen.add(sid)

        cond = row.condition.strip()
        if not _SAFE_ID.match(cond):
            errors.append(
                f"line {i}: {sid} has invalid condition {cond!r} (use letters, digits, . _ -)")
            continue

        if not row.r1.strip():
            errors.append(f"line {i}: {sid} has no r1")
            continue
        r1 = resolve(row.r1)
        r2 = resolve(row.r2) if row.r2.strip() else None

        for label, p in (("r1", r1), ("r2", r2)):
            if p is not None and not p.is_file():
                errors.append(f"line {i}: {sid} {label} not found: {p}")
        if r2 is not None and r1 == r2:
            errors.append(f"line {i}: {sid} r1 and r2 are the same file")

        samples.append(Sample(sid, cond, r1, r2))

    by_lower: dict[str, set[str]] = {}
    for s in samples:
        by_lower.setdefault(s.condition.lower(), set()).add(s.condition)
    clashes = [", ".join(sorted(v)) for v in by_lower.values() if len(v) > 1]
    if clashes:
        errors.append("conditions differing only in case: " + "; ".join(clashes))

    if errors:
        raise StepError("Problems in samples file:\n  " + "\n  ".join(errors))
    if not samples:
        raise StepError(f"No samples in {path}")
    return samples
