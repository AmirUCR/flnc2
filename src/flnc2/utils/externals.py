# src/flnc2/utils/externals.py
"""Third-party files Flnc2 needs but does not ship: CPAT models, LGC 2.0, ORFfinder.

They are published as one archive attached to a Flnc2 GitHub release, downloaded
once into <cache_dir>/externals/v<version>/, checked against a pinned sha256,
and unpacked. Build the archive from your current src/ext with:

    tar -czf flnc2-externals-v1.tar.gz -C src/ext cpat lgc2
    sha256sum flnc2-externals-v1.tar.gz

Bump EXTERNALS_VERSION whenever the archive changes, so existing caches are
not reused for different files.
"""
import hashlib
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from flnc2.utils.custom_logger import get_logger
from flnc2.utils.exceptions import StepError

logger = get_logger()

EXTERNALS_VERSION = "1"
EXTERNALS_URL = (f"https://github.com/AmirUCR/flnc2/releases/download/"
                 f"externals-v{EXTERNALS_VERSION}/flnc2-externals-v{EXTERNALS_VERSION}.tar.gz")
EXTERNALS_SHA256 = "657bf105d806cb5f8a8b41e1dbbd0595cec79fba23183ebe77f1419e79e5a315"

# Files every externals archive must contain, relative to its root.
REQUIRED = (
    "cpat/Human_Hexamer.tsv",
    "cpat/Human_logitModel.RData",
    "cpat/Human_cutoff.txt",
    "cpat/Mouse_Hexamer.tsv",
    "cpat/Mouse_logitModel.RData",
    "cpat/Mouse_cutoff.txt",
    "lgc2/LGC-2.0.py",
    "lgc2/ORFfinder",
)


def externals_dir(cache_dir: Path) -> Path:
    return Path(cache_dir) / "externals" / f"v{EXTERNALS_VERSION}"


def missing_files(ext_dir: Path) -> list[str]:
    return [rel for rel in REQUIRED if not (Path(ext_dir) / rel).is_file()]


def ensure_externals(cache_dir: Path) -> Path:
    """Return the externals directory, downloading it first if incomplete."""
    dest = externals_dir(cache_dir)
    if not missing_files(dest):
        logger.info("  Using cached external tools at %s", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("  Downloading external tools (CPAT models, LGC 2.0, ORFfinder)")

    # Download and unpack in a temp dir next to the destination, then move the
    # finished directory into place, so an interrupted run leaves nothing behind.
    with tempfile.TemporaryDirectory(dir=dest.parent, prefix=".download_") as tmp:
        archive = Path(tmp) / "externals.tar.gz"
        digest = hashlib.sha256()
        try:
            with urllib.request.urlopen(EXTERNALS_URL, timeout=60) as response, \
                    archive.open("wb") as out:
                while chunk := response.read(1 << 20):
                    digest.update(chunk)
                    out.write(chunk)
        except OSError as e:
            raise StepError(f"Could not download {EXTERNALS_URL}: {e}")

        if digest.hexdigest() != EXTERNALS_SHA256:
            raise StepError(f"Checksum mismatch for {EXTERNALS_URL}: expected "
                            f"{EXTERNALS_SHA256}, got {digest.hexdigest()}")

        unpacked = Path(tmp) / "unpacked"
        with tarfile.open(archive) as tar:
            tar.extractall(unpacked, filter="data")   # rejects absolute paths and ../

        missing = missing_files(unpacked)
        if missing:
            raise StepError(f"Externals archive is missing: {', '.join(missing)}")

        orffinder = unpacked / "lgc2" / "ORFfinder"
        orffinder.chmod(orffinder.stat().st_mode | 0o111)

        if dest.exists():
            shutil.rmtree(dest)   # an incomplete earlier download
        os.replace(unpacked, dest)

    logger.info("  External tools ready at %s", dest)
    return dest
