# src/py/filter_by_expression.py
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.signal import argrelextrema

from flnc2.utils.custom_logger import get_logger
from flnc2.utils.exceptions import StepError
from flnc2.utils.pipeline_step import PipelineStep
from flnc2.utils.gtf_tools import build_gtf_db
from flnc2.utils.tools import require_files

logger = get_logger()

_BED12_COLS = ["chrom", "chromStart", "chromEnd", "name", "score", "strand",
               "thickStart", "thickEnd", "itemRgb", "blockCount", "blockSizes", "blockStarts"]

def bw_nrd0(x: np.ndarray) -> float:
    """R's bw.nrd0 (Silverman's rule of thumb), the default for density()."""
    sd = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    lo = min(sd, iqr / 1.349)
    if lo == 0:
        lo = sd or abs(x[0]) or 1.0
    return 0.9 * lo * len(x) ** (-0.2)

def r_density(x: np.ndarray, n: int = 512, cut: int = 3):
    """Mimic R's density(): bw.nrd0 bandwidth, grid extended cut*bw past the data."""
    bw = bw_nrd0(x)
    # gaussian_kde's bw_method is a multiple of the data sd, not an absolute bandwidth.
    kde = gaussian_kde(x, bw_method=bw / np.std(x, ddof=1))
    grid = np.linspace(x.min() - cut * bw, x.max() + cut * bw, n)
    return grid, kde(grid), bw

@dataclass(frozen=True)
class ExpressionFilterOutputs:
    bed_path: Path
    cutoffs: dict[str, float]
    num_genes_before: int
    num_genes_after: int
    num_tx_before: int
    num_tx_after: int

class FilterByExpression(PipelineStep[ExpressionFilterOutputs]):
    """Per-sample density-mode FPKM cutoff, capped at max_cutoff, as in Flnc1's
    LIB/FPKM_Cutoff.R. A gene is kept if it clears its own sample's cutoff in
    any sample of this condition. Transcripts are kept if their gene is."""

    def __init__(
        self,
        bed_path: str | Path,
        merged_gtf_path: str | Path,
        fpkm_path: str | Path,
        output_dir: str | Path,
        label: str,
        max_cutoff: float = 0.01,
        fit_max_fpkm: float = 0.5,
        min_points: int = 100,
    ):
        self.bed_path = Path(bed_path)
        self.merged_gtf_path = Path(merged_gtf_path)
        self.fpkm_path = Path(fpkm_path)
        self.output_dir = Path(output_dir)
        self.label = label
        self.max_cutoff = float(max_cutoff)
        self.fit_max_fpkm = float(fit_max_fpkm)
        self.min_points = int(min_points)

    def run(self) -> ExpressionFilterOutputs:
        logger.info("Filtering by expression")
        require_files(self.bed_path, self.merged_gtf_path, self.fpkm_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        bed = pd.read_table(self.bed_path, header=None, names=_BED12_COLS, dtype=str)
        db = build_gtf_db(self.merged_gtf_path)
        tx2gene = {tx.attributes["transcript_id"][0]: tx.attributes["gene_id"][0]
                   for tx in db.features_of_type("transcript")}
        bed["gene_id"] = bed["name"].map(tx2gene)
        n_orphan = bed["gene_id"].isna().sum()
        if n_orphan:
            logger.warning("  %d transcripts have no gene_id in %s", n_orphan, self.merged_gtf_path)

        # The fit uses only genes of transcripts still in the BED, as in the R script.
        fpkm = pd.read_table(self.fpkm_path, index_col="Geneid")
        fpkm_bed = fpkm.loc[fpkm.index.intersection(bed["gene_id"].dropna().unique())]
        logger.info("  Genes in FPKM matrix: %d | genes in BED: %d", len(fpkm), len(fpkm_bed))
        if fpkm_bed.empty:
            raise StepError(f"No gene_id overlap between {self.fpkm_path} and {self.bed_path}")

        cutoffs = {s: self._cutoff(s, fpkm_bed[s].to_numpy()) for s in fpkm_bed.columns}
        passes = pd.DataFrame({s: fpkm_bed[s] >= c for s, c in cutoffs.items()})
        expressed_genes = set(fpkm_bed.index[passes.any(axis=1)])

        kept = bed[bed["gene_id"].isin(expressed_genes)].drop(columns="gene_id")
        logger.info("  Expressed genes: %d -> %d", len(fpkm_bed), len(expressed_genes))
        logger.info("  Expressed transcripts: %d -> %d", len(bed), len(kept))

        out_bed = self.output_dir / "expressed.bed"
        kept.to_csv(out_bed, sep="\t", index=False, header=False)

        result = ExpressionFilterOutputs(
            bed_path=out_bed, cutoffs=cutoffs,
            num_genes_before=len(fpkm_bed), num_genes_after=len(expressed_genes),
            num_tx_before=len(bed), num_tx_after=len(kept),
        )
        self._write_summary(result)
        return result

    def _cutoff(self, sample: str, values: np.ndarray) -> float:
        x = values[(values > 0) & (values < self.fit_max_fpkm)]
        if len(x) < self.min_points:
            logger.info("  %s: only %d genes in (0, %g), using %g",
                        sample, len(x), self.fit_max_fpkm, self.max_cutoff)
            return self.max_cutoff

        grid, y, bw = r_density(x)
        peaks = argrelextrema(y, np.greater)[0]
        if len(peaks) == 0:
            logger.info("  %s: no density mode found, using %g", sample, self.max_cutoff)
            return self.max_cutoff

        mode = grid[peaks[0]]
        cutoff = min(mode, self.max_cutoff)
        logger.info("  %s: bw=%.5f mode=%.4f cutoff=%.4f n=%d", sample, bw, mode, cutoff, len(x))
        self._plot(x, grid, y, peaks, cutoff, sample)
        return cutoff

    def _plot(self, x, grid, y, peaks, cutoff, sample) -> None:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(grid, y, linewidth=2)
        marks = list(peaks[:2]) + list(argrelextrema(y, np.less)[0][:1])
        for i in marks:
            q = 100 * (1 - np.mean(x > grid[i]))
            ax.annotate(f"x={grid[i]:.3f}\nquantile={q:.1f}%", (grid[i], y[i]), fontsize=9)
        ax.axvline(cutoff, color="#DD8452", linestyle="--", linewidth=1.5,
                   label=f"cutoff={cutoff:.4f}")
        ax.set_xlabel("FPKM")
        ax.set_ylabel("Density")
        ax.set_title(f"{self.label} / {sample}")
        ax.legend(fontsize=9)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.grid(alpha=0.3, linestyle="--")
        ax.tick_params(axis="both", which="both", length=0)
        fig.tight_layout()
        plots = self.output_dir / "fpkm_plots"
        plots.mkdir(exist_ok=True)
        fig.savefig(plots / f"fpkm_cutoff_{self.label}_{sample}.png", dpi=150)
        plt.close(fig)

    def _write_summary(self, r: ExpressionFilterOutputs) -> None:
        path = self.output_dir / "expression_filter_summary.tsv"
        with path.open("w") as f:
            f.write("# genes_before/after: genes of BED transcripts, before/after the FPKM cutoff.\n")
            f.write("# tx_before/after: transcripts, before/after the FPKM cutoff.\n")
            f.write("condition\tgenes_before\tgenes_after\ttx_before\ttx_after\n")
            f.write(f"{self.label}\t{r.num_genes_before}\t{r.num_genes_after}\t"
                    f"{r.num_tx_before}\t{r.num_tx_after}\n")
        logger.info("  Find a report in %s", path)
