# src/py/utils/py_bedtools.py
from pathlib import Path
from itertools import chain
from dataclasses import dataclass

import pybedtools

def concat_beds(beds: list[Path], dest: Path) -> Path:
    """Concatenate and sort BEDs, keeping every column and name as is."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    combined = pybedtools.BedTool(
        chain.from_iterable(pybedtools.BedTool(str(b)) for b in beds)).sort()
    combined.saveas(str(dest))
    return dest

@dataclass(frozen=True)
class ImpureLociResult:
    notoverlap: pybedtools.BedTool   # query with no same-strand overlap with exclude
    overlap: pybedtools.BedTool      # query overlapping exclude (flags impure loci)
    pure: pybedtools.BedTool         # notoverlap minus anything sharing a locus with overlap

def remove_impure_loci(
    query: pybedtools.BedTool, exclude: pybedtools.BedTool
) -> ImpureLociResult:
    notoverlap = query.intersect(exclude, s=True, v=True, wa=True)
    overlap = query.intersect(exclude, s=True, u=True)
    pure = notoverlap.intersect(overlap, s=True, v=True, wa=True)
    return ImpureLociResult(notoverlap=notoverlap, overlap=overlap, pure=pure)
