# src/py/prepare_training_data.py
import re
from pathlib import Path
from dataclasses import dataclass

import pandas as pd

from flnc2.utils.custom_logger import get_logger
from flnc2.utils.exceptions import StepError
from flnc2.utils.pipeline_step import PipelineStep
from flnc2.utils.tools import require_files
from flnc2.combine_conditions_beds import NAME_SEPARATOR, split_name

logger = get_logger()

_SAFE_GROUP = re.compile(r"^[A-Za-z0-9._-]+$")   # used as a directory name

@dataclass(frozen=True)
class TrainingDataOutputs:
    labels_path: Path               # name -> label
    group_beds: dict[str, Path]     # group -> BED12; one "all" entry when ungrouped
    grouped: bool

class PrepareTrainingData(PipelineStep[TrainingDataOutputs]):
    """BED12 + one label column -> BED12(s) for filtering, plus a label table.

    Names must be unique. If every name has the form 'group|name', rows are
    split by group so locus-level filters act within each group, as the
    per-condition filtering does in the reads route.
    """

    def __init__(self, bed_path: str | Path, output_dir: str | Path, label_col: str):
        self.bed_path = Path(bed_path)
        self.output_dir = Path(output_dir)
        self.label_col = label_col

    def run(self) -> TrainingDataOutputs:
        require_files(self.bed_path)
        df = pd.read_table(self.bed_path, header=None, comment="#", dtype=str)
        if not df.empty and df.iloc[0, 0] == "chrom":   # optional header row
            df = df.iloc[1:].reset_index(drop=True)
        if df.shape[1] != 13:
            raise StepError(f"{self.bed_path} should be BED12 plus one label column "
                            f"(13 columns), found {df.shape[1]}")
        if df.empty:
            raise StepError(f"No rows in {self.bed_path}")

        bed, names, labels = df.iloc[:, :12], df[3], df[12].str.strip()

        bad = labels[~labels.isin(["0", "1"])]
        if len(bad):
            raise StepError(f"Labels must be 0 or 1; found {bad.unique()[:5].tolist()}")

        dupes = names[names.duplicated()].unique()
        if len(dupes):
            raise StepError(
                f"{len(dupes)} duplicate name(s), e.g. {dupes[0]}. If a transcript appears "
                f"in several samples, make names unique, e.g. GSE103849_fko{NAME_SEPARATOR}{dupes[0]}")

        tagged = names.str.contains(NAME_SEPARATOR, regex=False)
        if tagged.any() and not tagged.all():
            raise StepError(f"Either all names or none should have the form "
                            f"'group{NAME_SEPARATOR}name'; {tagged.sum()} of {len(names)} do")
        grouped = bool(tagged.all())

        self.output_dir.mkdir(parents=True, exist_ok=True)
        labels_path = self.output_dir / "labels.tsv"
        pd.DataFrame({"name": names, self.label_col: labels.astype(int)}) \
            .to_csv(labels_path, sep="\t", index=False)

        if grouped:
            groups = names.map(lambda n: split_name(n)[0])
            unsafe = [g for g in groups.unique() if not _SAFE_GROUP.match(g)]
            if unsafe:
                raise StepError(f"Invalid group name(s) {unsafe[:5]} (use letters, digits, . _ -)")
            group_beds = {}
            for group, rows in bed.groupby(groups, sort=False):
                path = self.output_dir / f"{group}.bed"
                rows.to_csv(path, sep="\t", header=False, index=False)
                group_beds[group] = path
        else:
            path = self.output_dir / "all.bed"
            bed.to_csv(path, sep="\t", header=False, index=False)
            group_beds = {"all": path}

        n_pos = int((labels == "1").sum())
        logger.info("Training input: %d transcripts (%d positive, %d negative), %s",
                    len(df), n_pos, len(df) - n_pos,
                    f"{len(group_beds)} group(s)" if grouped else "no groups")
        return TrainingDataOutputs(labels_path, group_beds, grouped)
