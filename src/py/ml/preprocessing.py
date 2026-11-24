# src/py/ml/preprocessing.py
import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import asdict, dataclass

from src.py.utils.exceptions import StepError
from src.py.count_kmers import kmer_names

def kmer_feature_cols(k: int) -> list[str]:
    names = kmer_names(k)
    return names + [f"{c}_flank" for c in names]

def locus_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("locus_") and c != "locus_type"]

@dataclass
class Preprocessing:
    """Everything needed to turn a feature table into model input, fitted on
    training rows only. Mirrors the scaling in the training script."""
    label_col: str
    feature_cols: list[str]       # final column order fed to the model
    kmer_cols: list[str]
    kmer_min: list[float]
    kmer_max: list[float]
    length_lower: float           # mean - 3 sd of log_exon_lengths
    length_upper: float           # mean + 3 sd
    length_min: float             # min/max after clipping
    length_max: float
    exon_count_min: float
    exon_count_max: float

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Preprocessing":
        return cls(**json.loads(Path(path).read_text()))

def fit_preprocessing(
    df: pd.DataFrame,
    kmer_cols: list[str],
    locus_cols: list[str],
    label_col: str
) -> Preprocessing:
    log_len = df["log_exon_lengths"]
    lower = log_len.mean() - 3 * log_len.std()
    upper = log_len.mean() + 3 * log_len.std()
    clipped = log_len.clip(lower, upper)
    return Preprocessing(
        label_col=label_col,
        feature_cols=kmer_cols + ["length_scaled", "exon_count_scaled"] + locus_cols,
        kmer_cols=kmer_cols,
        kmer_min=df[kmer_cols].min().astype(float).tolist(),
        kmer_max=df[kmer_cols].max().astype(float).tolist(),
        length_lower=float(lower), length_upper=float(upper),
        length_min=float(clipped.min()), length_max=float(clipped.max()),
        exon_count_min=float(df["blockCount"].min()),
        exon_count_max=float(df["blockCount"].max()),
    )

def _minmax(x, lo, hi):
    # Same as sklearn's MinMaxScaler: a zero range divides by 1 instead of 0.
    rng = np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float)
    rng = np.where(rng == 0, 1.0, rng)
    return (x - lo) / rng

def transform(df: pd.DataFrame, p: Preprocessing) -> np.ndarray:
    needed = set(p.kmer_cols) | {"log_exon_lengths", "blockCount"} | {
        c for c in p.feature_cols if c not in ("length_scaled", "exon_count_scaled")}
    missing = sorted(needed - set(df.columns))
    if missing:
        raise StepError(f"Feature table is missing {len(missing)} column(s) the model "
                        f"needs, e.g. {', '.join(missing[:5])}")

    x = df.copy()
    x[p.kmer_cols] = _minmax(x[p.kmer_cols].to_numpy(dtype=float), p.kmer_min, p.kmer_max)
    clipped = x["log_exon_lengths"].clip(p.length_lower, p.length_upper)
    x["length_scaled"] = _minmax(clipped, p.length_min, p.length_max)
    x["exon_count_scaled"] = _minmax(x["blockCount"].astype(float),
                                     p.exon_count_min, p.exon_count_max)
    return x[p.feature_cols].to_numpy(dtype=np.float32)
