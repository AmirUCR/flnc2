# src/py/utils/configurator.py
import os
import re
import sys
import time
import json
import argparse
from datetime import datetime
from pathlib import Path

import yaml
import pandas as pd

from src.py.coding_potential import SPECIES_MODELS
from src.py.utils.custom_logger import get_logger, set_quiet
from src.py.utils.exceptions import StepError
from src.py.utils.reference import fetch_gunzipped, reference_path, reference_url
from src.py.utils.samples import load_samples
from src.py.utils.tools import require_on_path_iterable, conda_env_exists

logger = get_logger()

DEFAULT_CONFIG = "config.yaml"
MODES = ("inference", "train")
STRANDEDNESS = ("F", "R", "FR", "RF", "unstranded")
BED12_COLS = ("chrom", "chromStart", "chromEnd", "name", "score", "strand",
              "thickStart", "thickEnd", "itemRgb", "blockCount", "blockSizes", "blockStarts")
ENV_FLAGS = ("hisat2_env", "rseqc_env", "stringtie_env", "subread_env", "cpat_env", "lgc_env")

# Options where an empty value ("" in YAML) means "not set".
_OPTIONAL = ("experiment_name", "bed", "samples", "gtf", "genome", "train_bed",
             "hisat2_index", "resume_dir", "model_dir", "cache_dir", "output_dir",
             "strandedness", "stringtie_assembly_args", "stringtie_merge_args", *ENV_FLAGS)

def sanitize_filename(filename, max_length=255) -> str:
    if not filename:
        return 'FLNC2_TEST_RUN'

    # Remove disallowed characters (e.g., /, \0, *, ?, ", <, >, |)
    sanitized = re.sub(r'[\/\0\*\?"<>\|]', '', str(filename))

    # Replace whitespace characters with underscores
    sanitized = re.sub(r'\s+', '_', sanitized)

    # Remove leading periods to avoid hidden files, unless it's a special case (e.g., ".", "..")
    if sanitized.startswith('.') and sanitized not in ['.', '..']:
        sanitized = sanitized.lstrip('.')

    # Böse böse...
    if sanitized == '.':
        logger.warning("Böse böse...")
        sanitized = '._'
    elif sanitized == '..':
        logger.warning("Böse böse...")
        sanitized = '.._'

    if not sanitized:
        return 'FLNC2_TEST_RUN'

    # Shorten the filename to comply with filesystem limits
    if len(sanitized.encode('utf-8')) > max_length:
        # Shorten while trying to preserve file extension
        name, dot, extension = sanitized.rpartition('.')
        if dot and extension and len(extension) <= max_length - 1:
            # Shorten name part, leave room for dot and extension
            name = name[:max_length - len(extension) - 2]
            sanitized = f"{name}.{extension}"
        else:
            # If there's no extension or it's too long, just truncate the name
            sanitized = sanitized[:max_length]

    return sanitized


def coerce_bool(v) -> bool:
    if isinstance(v, bool):
        return v

    if v is None:
        raise StepError("Expected boolean (True/False), got None.")

    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f"):
        return False

    raise StepError(f"Expected boolean (True/False), got: {v}. "
                    f"Accepted values: True/False, 1/0, yes/no, y/n, t/f")


class Configurator:
    def __init__(self) -> None:
        self.start_time = time.time()
        self.samples_path: Path | None = None   # args.samples becomes list[Sample]
        self.to_download: dict[str, str] = {}   # "genome"/"gtf" -> URL, for --get_genome

    def begruessung(self) -> None:
        logger.info("Flnc2: Welcome!")

    # ──────────────────────────────────────────────────────────────────────────
    # Parsing
    # ──────────────────────────────────────────────────────────────────────────
    def parse_configurations(self) -> None:
        # 1. Pre-parse only --config
        config_parser = argparse.ArgumentParser(add_help=False)
        config_parser.add_argument(
            "-c", "--config", type=str, default=None,
            help=f"Config file to use. Defaults to {DEFAULT_CONFIG} if it exists.",
        )
        config_args, remaining_args = config_parser.parse_known_args()

        # 2. Load YAML if present. A missing default config is fine; a missing
        #    config the user named explicitly is an error.
        config_path = Path(config_args.config or DEFAULT_CONFIG).expanduser()
        config_defaults = {}
        if config_path.is_file():
            try:
                with config_path.open() as f:
                    config_defaults = yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                m = re.search(r"line (\d+)", str(e))
                line_number = m.group(1) if m else "?"
                raise StepError(f"Problem reading {config_path} (line {line_number}): {e}")
            if not isinstance(config_defaults, dict):
                raise StepError(f"{config_path} should contain 'option: value' pairs.")
        elif config_args.config is not None:
            raise StepError(f"Cannot find config file {config_path}")

        parser = argparse.ArgumentParser(
            prog="Flnc2",
            description="Flnc2: Identification of lncRNA from RNA-seq data using deep learning",
            parents=[config_parser],
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        parser.add_argument("--help-examples", action="store_true", help="Show usage examples and exit.")

        g_general = parser.add_argument_group("General")
        g_inputs  = parser.add_argument_group("Inputs")
        g_output  = parser.add_argument_group("Output")
        g_ml      = parser.add_argument_group("Machine Learning")
        g_ext     = parser.add_argument_group("External Tools Settings")

        g_general.add_argument("--soundcheck", action="store_true", help="Run all configuration checks, then exit without running anything.")
        g_general.add_argument("--quiet", action="store_true", help="Only show warnings and errors.")
        g_general.add_argument("--experiment_name", type=str, default=None, help="Name of the experiment.")
        g_general.add_argument("--threads", type=int, default=1, help="Number of threads to use throughout the program.")
        g_general.add_argument("--species", type=str, default="mouse", help="Which species are we working with? Valid are ['human', 'mouse']")
        g_general.add_argument("--min_length", type=int, default=200, help="Minimum lncRNA transcript length. Transcripts shorter than this are discarded.")
        g_general.add_argument("--min_exon_length", type=int, default=2, help="Minimum transcript exon length. Transcripts composed of at least one exon shorter than this are discarded.")
        g_general.add_argument("--canonical_re", type=str, default="^(chr)?([0-9]+|X|Y|M|MT)$", help="Canonical chromosomes regular expression. Default covers both UCSC/GENCODE and Ensembl.")
        g_general.add_argument("--promoter_length", type=int, default=1_000, help="Length of the region upstream of each transcript's TSS used for promoter k-mer features. Must match the value the model was trained with.")
        g_general.add_argument("--k", type=int, default=6, help="K value used to create kmer buckets as part of feature extraction. Must match the value the model was trained with.")

        g_inputs.add_argument("--bed", type=str, default=None, help="Inference only. Path to the BED12 file containing transcript coordinates.")
        g_inputs.add_argument("--gtf", type=str, default=None, help="Path to the GTF file.")
        g_inputs.add_argument("--genome", type=str, default=None, help="Path to the genome file.")
        g_inputs.add_argument("--get_genome", type=coerce_bool, default=False, help="Download the GENCODE mouse M10 (GRCm38) genome and GTF into --cache_dir for whichever of --genome/--gtf is not set. Skipped if already downloaded.")
        g_inputs.add_argument("--samples", type=str, default=None, help="Inference only. Samples file pointing to .fastq.gz read files.")

        g_output.add_argument("--cache_dir", type=str, default=Path("data/cache"), help="Cached files are stored here.")
        g_output.add_argument("--output_dir", type=str, default=Path("data/output"), help="Output base directory.")
        g_output.add_argument("--resume_dir", type=str, default=None, help="Point to a directory to let Flnc2 know it should resume work using the files already present.")

        g_ml.add_argument("--mode", type=str, default="inference", help="'inference' (predict lncRNAs) or 'train'.")
        g_ml.add_argument("--model_dir", type=str, default=Path("data/cache/ml"), help="Inference only. Directory containing final_model.keras and preprocessing.json, as written by --mode train.")
        g_ml.add_argument("--train_bed", type=str, default=None, help="Train only. BED12 with a header row and one extra label column named by --label_col (13 columns total).")
        g_ml.add_argument("--epochs", type=int, default=None, help="Train only. Number of training epochs. tools/cv_harness.py reports a recommended value.")
        g_ml.add_argument("--label_col", type=str, default="H3K4me3", help="Train only. Name of the 0/1 label column in --train_bed.")

        g_ext.add_argument("--external_dir", type=str, default=Path("src/ext"), help="Location of external tools (e.g., CPAT, LGC2).")
        g_ext.add_argument("--hisat2_env", type=str, default=None, help="Conda environment in which HISAT2, samtools and seqtk live. Leave empty if they are installed in the same env.")
        g_ext.add_argument("--rseqc_env", type=str, default=None, help="Conda environment in which RSeQC lives. Leave empty if it is installed in the same env. Used for infer_experiment.py.")
        g_ext.add_argument("--stringtie_env", type=str, default=None, help="Conda environment in which StringTie lives. Leave empty if it is installed in the same env.")
        g_ext.add_argument("--subread_env", type=str, default=None, help="Conda environment in which featureCounts/Subread lives. Leave empty if it is installed in the same env.")
        g_ext.add_argument("--cpat_env", type=str, default=None, help="Conda environment in which CPAT lives. Leave empty if it is installed in the same env.")
        g_ext.add_argument("--lgc_env", type=str, default=None, help="Conda environment with Python 2.7 for LGC2.0. Leave empty if it is installed in the same env.")

        g_ext.add_argument("--hisat2_use_splice_sites", type=coerce_bool, default=False, help="Building with --ss on a mammalian genome needs on the order of 160 GB of RAM per the HISAT2 documentation, compared to roughly 8 GB for a plain index. On a typical workstation, the build with splice sites will be out-of-memory-killed.")
        g_ext.add_argument("--hisat2_index", type=str, default=None, help="Provide the path to your own HISAT2 index to skip building one.")
        g_ext.add_argument("--stranded_threshold", type=float, default=0.8, help="A sample is called stranded if at least this fraction of assignable reads agree. Used to determine HISAT2 --rna-strandness.")
        g_ext.add_argument("--strandedness", type=str, default=None, help="Strand-specific RNA-seq orientation for HISAT2 and featureCounts. If omitted, Flnc2 determines it for you. Options are 'F', 'R', 'FR', 'RF', 'unstranded'")
        g_ext.add_argument("--stringtie_assembly_args", type=str, default=None, help="Options passed to stringtie assembly step.")
        g_ext.add_argument("--stringtie_merge_args", type=str, default=None, help="Options passed to stringtie --merge.")

        # A typo in config.yaml (e.g. "threds: 64") would otherwise be silently ignored.
        known = {action.dest for action in parser._actions}
        unknown = sorted(set(config_defaults) - known)
        if unknown:
            raise StepError(f"Unknown option(s) in {config_path}: {', '.join(unknown)}. "
                            f"Check the spelling against `flnc2 --help`.")

        parser.set_defaults(**config_defaults)
        args = parser.parse_args(remaining_args)
        args.config = str(config_path) if config_path.is_file() else None

        # store_true doesn't convert YAML values, so `quiet: "false"` would be truthy.
        args.quiet = coerce_bool(args.quiet)
        set_quiet(args.quiet)

        if args.help_examples:
            print("Examples:")
            print("  flnc2 -c config.yaml")
            sys.exit(0)

        self.args = args

    # ──────────────────────────────────────────────────────────────────────────
    # Validation. Everything is checked before any directory is created, so a
    # bad option fails immediately instead of after hours of work.
    # ──────────────────────────────────────────────────────────────────────────
    def check_and_fix_configurations(self) -> argparse.Namespace:
        a = self.args
        for name in _OPTIONAL:
            if getattr(a, name, None) == "":
                setattr(a, name, None)

        if a.config:
            logger.info("  Unspecified command-line arguments default to the values in %s.", a.config)
        else:
            logger.info("  No config file found; using command-line arguments and built-in defaults.")

        self._check_general()
        self._check_inputs()
        self._check_external_tools()
        self._check_ml()

        if a.soundcheck:
            logger.info("  All checks passed. Flnc2 is ready to go.")
            sys.exit(0)

        self._prepare_output()
        for name, url in self.to_download.items():
            fetch_gunzipped(url, getattr(a, name))
        self.log_args()
        return a

    def _check_general(self) -> None:
        a = self.args
        a.experiment_name = sanitize_filename(a.experiment_name)
        logger.info("  Starting pipeline with name %s.", a.experiment_name)

        a.mode = str(a.mode).strip().lower()
        if a.mode not in MODES:
            raise StepError(f"--mode must be 'inference' or 'train', got {a.mode!r}")
        logger.info("  Mode: %s", a.mode)

        a.threads = self._int("threads", minimum=1)
        available = os.cpu_count() or 1
        if a.threads > available:
            logger.warning("  --threads %d exceeds the %d available CPUs; using %d.",
                           a.threads, available, available)
            a.threads = available
        logger.info("  Using %d threads.", a.threads)

        a.species = str(a.species).strip().lower()
        if a.species not in SPECIES_MODELS:
            raise StepError(f"Unsupported --species {a.species!r}. "
                            f"Supported: {', '.join(sorted(SPECIES_MODELS))}")

        # Resolved here because --get_genome needs it; created in _prepare_output.
        a.cache_dir = Path(a.cache_dir or "data/cache").expanduser().resolve()

        a.min_length = self._int("min_length", minimum=1)
        a.min_exon_length = self._int("min_exon_length", minimum=1)
        a.promoter_length = self._int("promoter_length", minimum=1)
        a.k = self._int("k", minimum=1, maximum=8)   # 4**k columns; k=8 is already 65,536

        try:
            re.compile(a.canonical_re)
        except (re.error, TypeError) as e:
            raise StepError(f"Invalid --canonical_re {a.canonical_re!r}: {e}")

    def _check_inputs(self) -> None:
        a = self.args
        a.get_genome = coerce_bool(a.get_genome)

        if a.get_genome:
            # Fill in whichever of --genome/--gtf is unset; user paths always win.
            filled = []
            for name in ("genome", "gtf"):
                if getattr(a, name) is None:
                    path = reference_path(a.species, name, a.cache_dir)
                    setattr(a, name, path)
                    filled.append(name)
                    if not path.is_file():
                        self.to_download[name] = reference_url(a.species, name)
            if len(filled) == 1:
                other = "gtf" if filled[0] == "genome" else "genome"
                logger.warning("  --get_genome supplies only the %s; make sure your --%s "
                               "is the same assembly (GRCm38, GENCODE M10).", filled[0], other)

        for name, what in (("gtf", "GTF file"), ("genome", "genome FASTA")):
            if name in self.to_download:
                logger.info("  %s will be downloaded to %s", what, getattr(a, name))
                continue
            if getattr(a, name) is None:
                raise StepError(f"A {what} (--{name}) is required. Provide one, or set "
                                f"--get_genome to download the GENCODE mouse M10 reference.")
            setattr(a, name, self._existing_file(name, what, required=True))
            logger.info("  Using %s at %s", what, getattr(a, name))

        if a.mode == "train":
            for name in ("bed", "samples"):
                if getattr(a, name) is not None:
                    logger.warning("  --%s is ignored in --mode train (training reads --train_bed).", name)
                    setattr(a, name, None)
            if a.train_bed is None:
                raise StepError("--mode train needs --train_bed (BED12 plus a 0/1 label column).")
            a.train_bed = self._existing_file("train_bed", "training BED", required=True)
            logger.info("  Using training BED at %s", a.train_bed)
            return

        # Inference
        if a.train_bed is not None:
            logger.warning("  --train_bed is ignored in --mode inference.")
            a.train_bed = None
        if a.bed is not None and a.samples is not None:
            raise StepError("--bed is mutually exclusive with --samples. Provide only one type of "
                            "input; check both the command line and the config file.")
        if a.bed is None and a.samples is None:
            raise StepError("Neither --bed nor --samples was provided. Provide one type of input, "
                            "on the command line or in the config file.")

        if a.bed is not None:
            a.bed = self._existing_file("bed", "BED file", required=True)
            logger.info("  Using BED file at %s", a.bed)
            return

        samples_path = self._existing_file("samples", "samples file", required=True)
        self.samples_path = samples_path
        logger.info("  Loading %s...", samples_path)
        a.samples = load_samples(samples_path)   # list[Sample]
        logger.info("    %d samples in %d condition(s)",
                    len(a.samples), len({s.condition for s in a.samples}))

        if a.strandedness is not None:
            s = str(a.strandedness).strip()
            s = "unstranded" if s.lower() in ("unstranded", "none") else s.upper()
            if s not in STRANDEDNESS:
                raise StepError(f"Invalid --strandedness {a.strandedness!r}. "
                                f"Valid choices are {', '.join(STRANDEDNESS)}.")
            a.strandedness = s
            logger.info("  Using strandedness %s", s)
        else:
            try:
                a.stranded_threshold = float(a.stranded_threshold)
            except (TypeError, ValueError):
                raise StepError(f"--stranded_threshold must be a number, got {a.stranded_threshold!r}")
            if not 0.5 < a.stranded_threshold <= 1:
                raise StepError(f"--stranded_threshold must be in (0.5, 1], got {a.stranded_threshold}")

    def _check_external_tools(self) -> None:
        a = self.args
        reads_route = a.mode == "inference" and a.samples is not None

        for flag in ENV_FLAGS:
            env = getattr(a, flag)
            if env is not None and not conda_env_exists(env):
                raise StepError(f"No such conda env '{env}' for --{flag}.")

        # CPAT and LGC models, needed by the lncRNA filtering in every route.
        ext = Path(a.external_dir).expanduser()
        cpat_prefix = SPECIES_MODELS[a.species][0]
        needed = [
            ext / "cpat" / f"{cpat_prefix}_Hexamer.tsv",
            ext / "cpat" / f"{cpat_prefix}_logitModel.RData",
            ext / "cpat" / f"{cpat_prefix}_cutoff.txt",
            ext / "lgc2" / "LGC-2.0.py",
            ext / "lgc2" / "ORFfinder",
        ]
        missing = [str(p) for p in needed if not p.is_file()]
        if missing:
            raise StepError(f"--external_dir {ext} is missing: {', '.join(missing)}")
        a.external_dir = ext

        a.hisat2_use_splice_sites = coerce_bool(a.hisat2_use_splice_sites)

        # Only check the tools this run will actually call.
        tools = {"bedtools": None, "cpat": a.cpat_env}
        if reads_route:
            tools.update({
                "hisat2": a.hisat2_env,
                "hisat2_extract_splice_sites.py": a.hisat2_env,
                "samtools": a.hisat2_env,
                "stringtie": a.stringtie_env,
                "featureCounts": a.subread_env,
            })
            if a.strandedness is None:
                tools["seqtk"] = a.hisat2_env
                tools["infer_experiment.py"] = a.rseqc_env

            if a.hisat2_index is not None:
                idx = Path(a.hisat2_index).expanduser()
                prefix = idx.with_name(re.sub(r"\.\d+\.ht2l?$", "", idx.name))
                if not any(prefix.parent.glob(f"{prefix.name}.1.ht2*")):
                    raise StepError(f"No HISAT2 index found for --hisat2_index {idx} "
                                    f"(expected {prefix}.1.ht2 ... {prefix}.8.ht2).")
                a.hisat2_index = idx
            else:
                tools["hisat2-build"] = a.hisat2_env
                if a.hisat2_use_splice_sites:
                    tools["hisat2_extract_exons.py"] = a.hisat2_env

        require_on_path_iterable(tools)

    def _check_ml(self) -> None:
        a = self.args
        if a.mode == "train":
            if a.epochs is None:
                raise StepError("--mode train needs --epochs >= 1. The CV harness "
                                "(tools/cv_harness.py) reports a recommended value.")
            a.epochs = self._int("epochs", minimum=1)
            self._check_train_labels()
            return

        if a.model_dir is None:
            raise StepError("--mode inference needs --model_dir.")
        a.model_dir = Path(a.model_dir).expanduser().resolve()
        if not a.model_dir.is_dir():
            raise StepError(f"--model_dir does not exist or is not a directory: {a.model_dir}")
        missing = [n for n in ("final_model.keras", "preprocessing.json")
                   if not (a.model_dir / n).is_file()]
        if missing:
            raise StepError(f"--model_dir {a.model_dir} is missing {', '.join(missing)}. "
                            f"Point it at the model/ directory written by --mode train.")
        logger.info("  Using model dir at %s", a.model_dir)

    def _check_train_labels(self) -> None:
        """--train_bed: header row, BED12 + label_col (13 columns), labels only 0/1."""
        a = self.args
        label_col = str(a.label_col).strip() if a.label_col is not None else ""
        if not label_col:
            raise StepError("--label_col must name the label column in --train_bed.")
        if label_col in BED12_COLS:
            raise StepError(f"--label_col {label_col!r} is a BED12 column name; "
                            f"it must name the extra 13th column.")
        a.label_col = label_col

        try:
            df = pd.read_table(a.train_bed, dtype=str, keep_default_na=False)
        except pd.errors.EmptyDataError:
            raise StepError(f"{a.train_bed} is empty.")
        except pd.errors.ParserError as e:
            raise StepError(f"Could not parse {a.train_bed}: {e}")
        df.columns = [str(c).lstrip("#").strip() for c in df.columns]

        if label_col not in df.columns:
            raise StepError(
                f"--label_col '{label_col}' not found in {a.train_bed}. The file needs a header "
                f"row naming its 13 columns (BED12 followed by '{label_col}'). "
                f"First line read as: {', '.join(df.columns[:13])}")
        if len(df.columns) != 13 or df.columns[12] != label_col:
            raise StepError(
                f"{a.train_bed} should have 13 columns, BED12 followed by '{label_col}'. "
                f"Found {len(df.columns)}: {', '.join(df.columns)}")
        if df.empty:
            raise StepError(f"{a.train_bed} has a header but no rows.")

        labels = df[label_col].str.strip()
        bad = labels[~labels.isin(["0", "1"])]
        if len(bad):
            rows = ", ".join(str(i + 1) for i in bad.index[:5]) + (", ..." if len(bad) > 5 else "")
            values = ", ".join(repr(v) for v in bad.unique()[:5])
            raise StepError(
                f"'{label_col}' must contain only 0 and 1. Found {len(bad)} other value(s) "
                f"({values}) in data row(s) {rows} of {a.train_bed}.")

        n_pos = int((labels == "1").sum())
        n_neg = len(labels) - n_pos
        logger.info("  Training labels in '%s': %d positive, %d negative", label_col, n_pos, n_neg)
        if n_pos == 0 or n_neg == 0:
            logger.warning("  '%s' contains only class %s. Fitting a model needs both 0 and 1, "
                           "so training will fail at the final step.",
                           label_col, "1" if n_neg == 0 else "0")

    def _prepare_output(self) -> None:
        a = self.args
        a.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("  Using cache dir %s", a.cache_dir)

        if a.resume_dir is not None:
            path = Path(a.resume_dir).expanduser().resolve()
            if not path.is_dir():
                raise StepError(f"--resume_dir does not exist or is not a directory: {path}")
            a.output_dir = path
            logger.info("  Using existing output dir at %s (via --resume_dir)", path)
        else:
            a.output_dir = self.create_output_dir(a.output_dir or "data/output", a.experiment_name)

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────
    def _int(self, name: str, minimum: int | None = None, maximum: int | None = None) -> int:
        """Coerce an option to int. YAML can deliver strings ("64") or floats."""
        value = getattr(self.args, name)
        try:
            if isinstance(value, bool):   # bool is an int subclass; True here is a config mistake
                raise ValueError
            v = int(value)
            if isinstance(value, float) and value != v:   # 2.5 should not silently become 2
                raise ValueError
        except (TypeError, ValueError):
            raise StepError(f"--{name} must be an integer, got {value!r}")
        if minimum is not None and v < minimum:
            raise StepError(f"--{name} must be >= {minimum}, got {v}")
        if maximum is not None and v > maximum:
            raise StepError(f"--{name} must be <= {maximum}, got {v}")
        return v

    def _existing_file(self, name: str, what: str, required: bool = False) -> Path | None:
        value = getattr(self.args, name)
        if value is None:
            if required:
                raise StepError(f"A {what} (--{name}) is required.")
            return None
        path = Path(value).expanduser()
        if not path.is_file():
            raise StepError(f"Cannot find {what} {path} (--{name}). "
                            f"Does it exist, and is the path spelled correctly?")
        return path

    def log_args(self) -> Path:
        """Write the options this run used to <output_dir>/<experiment>_config_used.yaml.

        Values are recorded after validation: config file and command-line
        overrides merged, defaults filled in, paths expanded.
        """
        a = self.args
        skip = {"soundcheck", "help_examples", "config"}
        used = {}
        for key, value in vars(a).items():
            if key in skip:
                continue
            if key == "samples" and value is not None:
                value = self.samples_path   # the sample sheet, not the parsed Sample objects
            used[key] = self._plain(value)

        path = Path(a.output_dir) / f"{a.experiment_name}_config_used.yaml"
        with path.open("w") as f:
            f.write(f"# Options used for Flnc2 experiment {a.experiment_name}\n")
            f.write(f"# Written {datetime.now().astimezone().isoformat(timespec='seconds')}\n")
            f.write(f"# Config file: {a.config or 'none (command line and built-in defaults)'}\n")
            yaml.safe_dump(used, f, sort_keys=False, default_flow_style=False)
        logger.info("  Options used written to %s", path)
        return path

    @staticmethod
    def _plain(value):
        """Convert a value to something yaml.safe_dump accepts."""
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, (list, tuple)):
            return [Configurator._plain(v) for v in value]
        if isinstance(value, dict):
            return {str(k): Configurator._plain(v) for k, v in value.items()}
        return str(value)   # Path and anything else

    def build_env(self, args) -> dict[str, str]:
        """String-only environment for bash scripts. Python steps should use `args`."""
        env = {}
        for key, value in vars(args).items():
            env_key = f"FLNC2_{key.upper()}"
            if value is None:
                env[env_key] = ""                           # set but empty, safe under `set -u`
            elif isinstance(value, bool):                   # must come before int
                env[env_key] = "1" if value else ""         # bash: [ -n "$VAR" ]
            elif isinstance(value, (int, float, str, Path)):
                env[env_key] = str(value)
            elif isinstance(value, (list, tuple)) and all(
                isinstance(v, (int, float, str, Path)) for v in value
            ):
                env[env_key] = " ".join(map(str, value))
            else:
                env[env_key] = json.dumps(value, default=str)  # dicts, lists of dicts
        return env

    def log_time(self, total_time_elapsed: float = 0) -> None:
        elapsed = time.time() - self.start_time + total_time_elapsed
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        hours, minutes = int(hours), int(minutes)

        if hours:
            took = f"{hours} hours, {minutes} minutes, {seconds:.2f} seconds"
        elif minutes:
            took = f"{minutes} minutes, {seconds:.2f} seconds"
        else:
            took = f"{seconds:.2f} seconds"
        logger.info("Flnc2 experiment took %s.", took)

        out = getattr(getattr(self, "args", None), "output_dir", None)
        if isinstance(out, Path) and out.is_dir():
            with (out / "run_time.txt").open("a") as f:
                f.write(f"Flnc2 experiment took {took}.\n")

    def create_output_dir(self, output_dir: str | Path, experiment_name: str) -> Path:
        base = Path(output_dir).expanduser().resolve() / experiment_name
        dir_name, i = base, 1
        while dir_name.exists():
            dir_name = base.with_name(f"{base.name}_{i}")
            i += 1
        logger.info("  Creating directory %s", dir_name)
        dir_name.mkdir(parents=True)
        return dir_name
