# src/py/quantify_counts.py
import os
import json
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

import pandas as pd

from flnc2.utils.custom_logger import get_logger
from flnc2.utils.exceptions import StepError
from flnc2.utils.pipeline_step import PipelineStep
from flnc2.utils.cache import file_fingerprint
from flnc2.utils.samples import Sample
from flnc2.utils.tools import require_files, run_tool, with_env, manifest_matches
from flnc2.align_reads import AlignmentOutputs
from flnc2.merge_assemblies import MergeOutputs

logger = get_logger()

# HISAT2 --rna-strandness -> featureCounts -s
_FC_STRAND = {None: "0", "FR": "1", "F": "1", "RF": "2", "R": "2"}
_LAYOUTS = ("SE", "PE")

@dataclass(frozen=True)
class ConditionCounts:
    condition: str
    counts: Path
    summary: Path
    num_genes: int
    layouts: dict[str, str]   # sample_id -> "SE" | "PE"

@dataclass(frozen=True)
class CountsOutputs:
    conditions: dict[str, ConditionCounts]
    summary: Path

class QuantifyCounts(PipelineStep[CountsOutputs]):
    def __init__(
        self,
        samples: list[Sample],
        alignments: AlignmentOutputs,
        merges: MergeOutputs,
        output_dir: str | Path,
        reports_dir: str | Path,
        threads: int = 1,
        extra_attributes: tuple[str, ...] = ("gene_name",),
        subread_env: str | None = None,
    ):
        missing = [s.sample_id for s in samples if s.sample_id not in alignments.samples]
        if missing:
            raise StepError("No alignment for sample(s): " + ", ".join(missing))
        conditions = {s.condition for s in samples}
        missing = sorted(conditions - set(merges.conditions))
        if missing:
            raise StepError("No merged assembly for condition(s): " + ", ".join(missing))

        # condition -> samples, in sample-sheet order
        self.by_condition: dict[str, list[Sample]] = {}
        for s in samples:
            self.by_condition.setdefault(s.condition, []).append(s)

        self.alignments = alignments
        self.merges = merges
        self.output_dir = Path(output_dir).resolve()
        self.reports_dir = Path(reports_dir).resolve()
        self.threads = int(threads)
        self.extra_attributes = tuple(extra_attributes)
        self.subread_env = subread_env

    def run(self) -> CountsOutputs:
        logger.info("Quantification")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        n_pe = sum(s.paired for group in self.by_condition.values() for s in group)
        n_total = sum(len(g) for g in self.by_condition.values())
        logger.info("  Layouts: %d paired-end, %d single-end", n_pe, n_total - n_pe)

        results = {
            cond: self._count(cond, group, self.merges.conditions[cond].gtf)
            for cond, group in self.by_condition.items()
        }
        summary = self._write_summary(results)
        return CountsOutputs(conditions=results, summary=summary)

    # ── per condition ───────────────────────────────────────────────────────
    def _count(self, condition: str, samples: list[Sample], gtf: Path) -> ConditionCounts:
        counts_path = self.output_dir / f"{condition}_counts.txt"
        summary_path = self.output_dir / f"{condition}_counts.summary"
        manifest = self.output_dir / f"{condition}_counts.manifest.json"
        layouts = {s.sample_id: "PE" if s.paired else "SE" for s in samples}

        bams = {s.sample_id: self.alignments.samples[s.sample_id].bam for s in samples}
        strands = {s.sample_id: _FC_STRAND[self.alignments.samples[s.sample_id].strandness]
                   for s in samples}
        require_files(gtf, *bams.values())

        expected = {
            "gtf": file_fingerprint(gtf),
            "bams": {sid: file_fingerprint(b) for sid, b in bams.items()},
            "strands": strands,
            "layouts": layouts,
            "extra_attributes": list(self.extra_attributes),
        }
        if counts_path.is_file() and summary_path.is_file() \
                and manifest_matches(manifest, expected):
            logger.info("  Using existing counts for %s", condition)
            return ConditionCounts(condition, counts_path, summary_path,
                                   self._num_genes(counts_path), layouts)

        manifest.unlink(missing_ok=True)
        tables, summaries = [], []

        with tempfile.TemporaryDirectory(dir=self.output_dir, prefix=".tmp_") as tmpdir:
            # featureCounts applies -p/--countReadPairs/-C to the whole call, so
            # SE and PE samples are counted separately and joined afterwards.
            for layout in _LAYOUTS:
                group = [s for s in samples if layouts[s.sample_id] == layout]
                if not group:
                    continue
                part = Path(tmpdir) / f"{layout}.txt"
                group_bams = [str(bams[s.sample_id]) for s in group]

                logger.info("  featureCounts: %s [%s] (%d BAMs) against %s",
                            condition, layout, len(group), gtf.name)
                cmd = [
                    "featureCounts",
                    "-T", str(self.threads),
                    # One -s value per BAM, so samples with different
                    # strandedness can share a call.
                    "-s", ",".join(strands[s.sample_id] for s in group),
                    "-a", str(gtf),
                    "-g", "gene_id",
                    "-t", "exon",
                ]
                if self.extra_attributes:
                    cmd += ["--extraAttributes", ",".join(self.extra_attributes)]
                if layout == "PE":
                    # -p: paired input; --countReadPairs: count fragments, not
                    # reads; -C: drop fragments whose mates hit different chromosomes.
                    cmd += ["-p", "--countReadPairs", "-C"]
                cmd += ["-o", str(part), *group_bams]

                run_tool(with_env(self.subread_env, cmd),
                         name=f"featureCounts ({condition}, {layout})",
                         log_path=self.output_dir / "logs" / f"{condition}_{layout}.log")

                # featureCounts names sample columns by the BAM path as given.
                rename = {str(bams[s.sample_id]): s.sample_id for s in group}
                tables.append(self._read_counts(part, rename))
                summaries.append(self._read_summary(Path(f"{part}.summary"), rename))

            counts = self._join_counts(tables, condition)
            summary = pd.concat(
                [summaries[0]] + [s.drop(columns="Status") for s in summaries[1:]], axis=1)

            # Restore sample-sheet column order across the SE/PE groups.
            order = [s.sample_id for s in samples]
            annot_cols = [c for c in counts.columns if c not in layouts]
            counts = counts[annot_cols + order]
            summary = summary[["Status"] + order]

            tmp_counts = Path(tmpdir) / counts_path.name
            tmp_summary = Path(tmpdir) / summary_path.name
            self._write_counts(tmp_counts, counts, layouts, strands)
            summary.to_csv(tmp_summary, sep="\t", index=False)
            os.replace(tmp_counts, counts_path)
            os.replace(tmp_summary, summary_path)

        (self.reports_dir / summary_path.name).write_bytes(summary_path.read_bytes())
        manifest.write_text(json.dumps(expected, indent=2))

        num_genes = len(counts)
        logger.info("    %s: %d genes -> %s", condition, num_genes, counts_path)
        return ConditionCounts(condition, counts_path, summary_path, num_genes, layouts)

    # ── table handling ──────────────────────────────────────────────────────
    @staticmethod
    def _read_counts(path: Path, rename: dict[str, str]) -> pd.DataFrame:
        # Line 1 is "# Program:featureCounts ..."; line 2 is the header.
        df = pd.read_table(path, skiprows=1, dtype=str).rename(columns=rename)
        for sid in rename.values():
            df[sid] = df[sid].astype(int)
        return df

    @staticmethod
    def _read_summary(path: Path, rename: dict[str, str]) -> pd.DataFrame:
        return pd.read_table(path, dtype=str).rename(columns=rename)

    @staticmethod
    def _join_counts(tables: list[pd.DataFrame], condition: str) -> pd.DataFrame:
        base = tables[0]
        annot_cols = [c for c in base.columns if not pd.api.types.is_integer_dtype(base[c])]
        for t in tables[1:]:
            # Same GTF for every call, so annotation rows must match exactly.
            # Checking (rather than merging on Geneid) catches a mismatch
            # instead of silently producing NaNs.
            if not t[annot_cols].equals(base[annot_cols]):
                raise StepError(f"featureCounts SE and PE tables disagree on genes for {condition}")
        sample_cols = [t.drop(columns=annot_cols) for t in tables[1:]]
        return pd.concat([base] + sample_cols, axis=1)

    @staticmethod
    def _write_counts(path: Path, df: pd.DataFrame,
                      layouts: dict[str, str], strands: dict[str, str]) -> None:
        present = [l for l in _LAYOUTS if l in layouts.values()]
        with path.open("w") as f:
            f.write(f"# merged from featureCounts runs: {', '.join(present)}\n")
            f.write("# sample layouts: "
                    + " ".join(f"{sid}={layouts[sid]}" for sid in df.columns if sid in layouts)
                    + "\n")
            f.write("# PE columns are FRAGMENT counts (--countReadPairs); "
                    "SE columns are READ counts\n")
            f.write("# sample strandedness (-s): "
                    + " ".join(f"{sid}={strands[sid]}" for sid in df.columns if sid in strands)
                    + "\n")
            df.to_csv(f, sep="\t", index=False)

    @staticmethod
    def _num_genes(path: Path) -> int:
        return len(pd.read_table(path, comment="#", usecols=[0]))

    # ── report ──────────────────────────────────────────────────────────────
    def _featurecounts_version(self) -> str:
        proc = subprocess.run(with_env(self.subread_env, ["featureCounts", "-v"]),
                              capture_output=True, text=True)
        return " ".join((proc.stdout + proc.stderr).split()) or "unknown"

    def _write_summary(self, results: dict[str, ConditionCounts]) -> Path:
        summary_file = self.reports_dir / "featurecounts_summary.tsv"
        with summary_file.open("w") as f:
            f.write(f"# generated by {type(self).__name__} on: "
                    f"{datetime.now().astimezone().isoformat(timespec='seconds')}\n")
            f.write(f"# featureCounts: {self._featurecounts_version()}\n")
            f.write("# PE calls add: -p --countReadPairs -C (fragment counts); SE calls count reads\n")
            f.write("condition\tnum_genes\tlayouts\n")
            for r in results.values():
                present = ",".join(l for l in _LAYOUTS if l in r.layouts.values())
                f.write(f"{r.condition}\t{r.num_genes}\t{present}\n")
            f.write("#\n# per-sample layout and strandedness\n# sample\tlayout\tstrand\n")
            for cond, samples in self.by_condition.items():
                for s in samples:
                    strand = _FC_STRAND[self.alignments.samples[s.sample_id].strandness]
                    f.write(f"# {s.sample_id}\t{results[cond].layouts[s.sample_id]}\t-s {strand}\n")
        logger.info("  Find a report in %s", summary_file)
        return summary_file
