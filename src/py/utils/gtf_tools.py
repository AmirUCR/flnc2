# src/py/utils/gtf_tools.py
import hashlib
import tempfile
from pathlib import Path

import gffutils
import pybedtools

from src.py.utils.custom_logger import get_logger
from src.py.utils.cache import ensure_cached

logger = get_logger()

def build_gtf_db(gtf_path: str | Path, db_path: str | Path = ":memory:") -> gffutils.FeatureDB:
    """Always build a fresh DB. For a cached, reusable DB use cached_gtf_db().

    GENCODE/Ensembl GTFs already have gene+transcript features, so inference
    off is both correct and much faster.
    """
    logger.info(f"Creating GTF database for {Path(gtf_path).name}")
    return gffutils.create_db(
        str(gtf_path),
        dbfn=str(db_path),
        force=True,
        keep_order=True,
        merge_strategy="create_unique",
        disable_infer_genes=True,
        disable_infer_transcripts=True,
    )

def cached_gtf_db(gtf_path: str | Path, cache_dir: str | Path) -> gffutils.FeatureDB:
    """Build the gffutils DB once per GTF and reuse it until the GTF changes."""
    gtf_path = Path(gtf_path).resolve()
    # Path hash in the name, so two different GTFs with the same file name
    # don't keep invalidating each other's cache.
    tag = hashlib.sha1(str(gtf_path).encode()).hexdigest()[:8]
    dest = Path(cache_dir) / "gffutils" / f"{gtf_path.name}.{tag}.gffutils.db"
    if not ensure_cached(dest, {"gtf": gtf_path}, lambda tmp: build_gtf_db(gtf_path, tmp)):
        logger.info(f"Using cached GTF database for {gtf_path.name}")
    return gffutils.FeatureDB(str(dest))

def _transcript_biotype(tx) -> str | None:
    # GENCODE uses transcript_type; Ensembl uses transcript_biotype.
    for key in ("transcript_type", "transcript_biotype"):
        if key in tx.attributes:
            return tx.attributes[key][0]
    return None

def _gene_biotype(gene) -> str | None:
    # GENCODE uses gene_type; Ensembl uses gene_biotype.
    for key in ("gene_type", "gene_biotype"):
        if key in gene.attributes:
            return gene.attributes[key][0]
    return None

def protein_coding_genes(db: gffutils.FeatureDB):
    """Yield gene features whose biotype is protein_coding."""
    for g in db.features_of_type("gene"):
        if _gene_biotype(g) == "protein_coding":
            yield g

def transcripts_to_bed12(db: gffutils.FeatureDB, keep_biotypes=None) -> pybedtools.BedTool:
    keep = set(keep_biotypes) if keep_biotypes is not None else None

    with tempfile.NamedTemporaryFile("w", suffix=".bed", delete=False) as out:
        bed_path = out.name
        for tx in db.features_of_type("transcript"):
            if keep is not None and _transcript_biotype(tx) not in keep:
                continue

            exons = sorted(db.children(tx, featuretype="exon"), key=lambda e: e.start)
            if not exons:
                continue

            # GTF is 1-based inclusive; BED is 0-based half-open.
            chrom_start = tx.start - 1
            chrom_end = tx.end
            # BED12 lists blocks in ascending genomic order regardless of strand.
            block_sizes = [e.end - e.start + 1 for e in exons]
            block_starts = [(e.start - 1) - chrom_start for e in exons]

            out.write("\t".join(str(v) for v in (
                tx.seqid, chrom_start, chrom_end,
                tx.attributes["transcript_id"][0],
                0, tx.strand,
                chrom_start, chrom_start,          # no CDS: thick region empty
                0,                                 # itemRgb
                len(exons),
                ",".join(map(str, block_sizes)) + ",",
                ",".join(map(str, block_starts)) + ",",
            )) + "\n")

    return pybedtools.BedTool(bed_path)
