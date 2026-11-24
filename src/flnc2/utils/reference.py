# src/flnc2/utils/reference.py
import gzip
import os
import shutil
import urllib.request
from pathlib import Path

from flnc2.utils.custom_logger import get_logger
from flnc2.utils.exceptions import StepError

logger = get_logger()

_GENCODE_MOUSE_M10 = {
    "genome": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M10/"
              "GRCm38.primary_assembly.genome.fa.gz",
    "gtf":    "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M10/"
              "gencode.vM10.primary_assembly.annotation.gtf.gz",
}

# species (as accepted by --species) -> {"genome": url, "gtf": url}
REFERENCES = {
    "mouse": _GENCODE_MOUSE_M10,
    "mus_musculus": _GENCODE_MOUSE_M10,
}


def reference_url(species: str, kind: str) -> str:
    if species not in REFERENCES:
        raise StepError(f"--get_genome has no reference for species '{species}'. "
                        f"Available: {', '.join(sorted(REFERENCES))}. "
                        f"Provide --genome and --gtf yourself.")
    return REFERENCES[species][kind]


def reference_path(species: str, kind: str, cache_dir: Path) -> Path:
    """Where the unzipped file lives: cache_dir/<file name without .gz>."""
    name = reference_url(species, kind).rsplit("/", 1)[-1]
    return Path(cache_dir) / name.removesuffix(".gz")


def fetch_gunzipped(url: str, dest: Path) -> Path:
    """Download a .gz file and decompress it to dest while streaming.

    Written to a .tmp file and renamed on success, so an interrupted download
    never leaves a truncated file under the final name.
    """
    if dest.is_file() and dest.stat().st_size > 0:
        logger.info("  Using cached %s", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    logger.info("  Downloading %s (this can take several minutes)", url)
    try:
        with urllib.request.urlopen(url, timeout=60) as response, \
                gzip.GzipFile(fileobj=response) as unzipped, \
                tmp.open("wb") as out:
            shutil.copyfileobj(unzipped, out, length=1 << 20)
    except (OSError, EOFError) as e:   # network errors and corrupt gzip data
        tmp.unlink(missing_ok=True)
        raise StepError(f"Could not download {url}: {e}")
    os.replace(tmp, dest)
    logger.info("  Saved %s", dest)
    return dest
