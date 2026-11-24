# src/py/pipelines.py
from pathlib import Path
from dataclasses import dataclass

from src.py.utils.custom_logger import get_logger
from src.py.utils.verify_bed import VerifyBED
from src.py.utils.tools import dataset_name

from src.py.hisat2_index import BuildHisat2Index, Hisat2IndexOutputs
from src.py.infer_strandedness import InferStrandedness, StrandednessOutputs
from src.py.align_reads import AlignReads, AlignmentOutputs
from src.py.assemble_transcripts import AssembleTranscripts, AssemblyOutputs
from src.py.merge_assemblies import MergeAssemblies, MergeOutputs
from src.py.quantify_counts import QuantifyCounts, CountsOutputs
from src.py.filter_non_lncrna import FilterNonLncRNA, FilterNonLncRNAOutputs
from src.py.coding_potential import CodingPotential, CodingPotentialOutputs
from src.py.call_coding import CallCoding, CallCodingOutputs
from src.py.filter_predicted_coding import FilterPredictedCoding, FilterPredictedCodingOutput
from src.py.compute_fpkm import ComputeFPKM, FPKMOutputs
from src.py.filter_by_expression import FilterByExpression, ExpressionFilterOutputs
from src.py.combine_conditions_beds import CombineConditionBeds, CombinedBedOutputs
from src.py.classify_locus import LocusOutputs, ClassifyLocus
from src.py.utils.bed_to_fasta import ExtractTranscriptSequences, TranscriptFastaOutputs, parse_fasta_name
from src.py.extract_promoter_sequence import ExtractPromoterSequences, PromoterFastaOutputs, parse_promoter_name
from src.py.count_kmers import CountKmers, KmerOutputs
from src.py.transcript_features import TranscriptFeatures, TranscriptFeaturesOutputs
from src.py.build_feature_table import BuildFeatureTable, FeatureTableOutputs
from src.py.predict import Predict, PredictionOutputs
from src.py.train_model import TrainModel, TrainOutputs
from src.py.prepare_training_data import PrepareTrainingData, TrainingDataOutputs
from src.py.utils.py_bedtools import concat_beds

logger = get_logger()

@dataclass(frozen=True)
class LncRNAOutputs:
    filtered: FilterNonLncRNAOutputs
    coding_potential: CodingPotentialOutputs
    coding_calls: CallCodingOutputs
    pure_noncoding: FilterPredictedCodingOutput

# ── shared pieces ───────────────────────────────────────────────────────────
def run_lncrna_filtering(bed_path: Path, args, output_dir: Path, verify: bool = True) -> LncRNAOutputs:
    """BED12 of candidate transcripts -> high-confidence noncoding transcripts.

    Writes fixed file names inside output_dir, so give each call its own dir.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verify:
        bed_path = VerifyBED(bed_path=bed_path, output_dir=output_dir).run()

    filtered = FilterNonLncRNA(
        bed_path=bed_path,
        gtf_path=args.gtf,
        output_dir=output_dir,
        cache_dir=Path(args.cache_dir),
        min_length=args.min_length,
        min_exon_length=args.min_exon_length,
        canonical_re=args.canonical_re,
    ).run()

    pureloci_fa = ExtractTranscriptSequences(
        bed_path=filtered.bed_path,
        genome_fasta_path=args.genome,
        output_dir=output_dir,
        output_name="pureloci.fa",
        embed_blocks=False,
    ).run()

    coding_potential = CodingPotential(
        pureloci_fa_path=pureloci_fa.fasta_path,
        external_dir=args.external_dir,
        output_dir=output_dir,
        species=args.species,
        cpat_env=args.cpat_env or None,
        lgc_env=args.lgc_env or None,
    ).run()

    coding_calls = CallCoding(
        bed_path=filtered.bed_path,
        cpat_best_tsv_path=coding_potential.cpat_best_tsv_path,
        cpat_no_orf_path=coding_potential.cpat_no_orf_path,
        cpat_cutoff_path=coding_potential.cpat_cutoff_path,
        lgc_out_path=coding_potential.lgc_out_path,
        output_dir=output_dir,
    ).run()

    pure_noncoding = FilterPredictedCoding(
        noncoding_bed_path=coding_calls.noncoding_bed_path,
        coding_bed_path=coding_calls.coding_bed_path,
        output_dir=output_dir,
    ).run()

    return LncRNAOutputs(filtered, coding_potential, coding_calls, pure_noncoding)


def build_features(candidates_bed: Path, args, out: Path) -> FeatureTableOutputs:
    """Candidate lncRNA BED -> one feature table, the same for both modes."""
    loci: LocusOutputs = ClassifyLocus(
        bed_path=candidates_bed,
        gtf_path=args.gtf,
        cache_dir=Path(args.cache_dir),
        output_dir=out / "locus",
    ).run()

    transcripts_fa: TranscriptFastaOutputs = ExtractTranscriptSequences(
        bed_path=loci.bed_path,
        genome_fasta_path=args.genome,
        output_dir=out / "sequences",
    ).run()

    promoters_fa: PromoterFastaOutputs = ExtractPromoterSequences(
        bed_path=loci.bed_path,
        genome_fasta_path=args.genome,
        output_dir=out / "sequences",
        flank_length=args.promoter_length,
    ).run()

    transcript_kmers: KmerOutputs = CountKmers(
        fasta_path=transcripts_fa.fasta_path,
        bed_path=loci.bed_path,
        output_path=out / "kmers" / "transcript_kmers.tsv",
        k=args.k,
        name_from_header=lambda h: parse_fasta_name(h)[0],
    ).run()

    promoter_kmers: KmerOutputs = CountKmers(
        fasta_path=promoters_fa.fasta_path,
        bed_path=loci.bed_path,
        output_path=out / "kmers" / "flank_kmers.tsv",
        k=args.k,
        name_from_header=lambda h: parse_promoter_name(h)[0],
    ).run()

    transcript_features: TranscriptFeaturesOutputs = TranscriptFeatures(
        bed_path=loci.bed_path,
        output_path=out / "features" / "transcript_features.tsv",
    ).run()

    return BuildFeatureTable(
        locus_table_path=loci.table_path,
        transcript_features_path=transcript_features.table_path,
        transcript_kmers_path=transcript_kmers.table_path,
        promoter_kmers_path=promoter_kmers.table_path,
        output_path=out / "features" / "final_df.tsv",
    ).run()


# ── input routes: each ends in a feature table ──────────────────────────────
def run_bed_pipeline(args) -> FeatureTableOutputs:
    out = Path(args.output_dir)
    lnc = run_lncrna_filtering(bed_path=args.bed, args=args, output_dir=out / "lncrna")
    return build_features(lnc.pure_noncoding.bed_path, args, out)


def run_reads_pipeline(args) -> FeatureTableOutputs:
    """Reads -> alignment -> assembly -> counts -> per-condition filtering -> features."""
    out = Path(args.output_dir)
    cache = Path(args.cache_dir)

    prebuilt = args.hisat2_index   # Path or None
    hisat2_index_output: Hisat2IndexOutputs = BuildHisat2Index(
        prebuilt_index_prefix=prebuilt,
        genome_fasta_path=args.genome,
        gtf_path=args.gtf,
        index_dir=cache / "hisat2",
        dataset=dataset_name(args.genome),
        threads=args.threads,
        use_splice_sites=bool(args.hisat2_use_splice_sites) and prebuilt is None,
        hisat2_env=args.hisat2_env or None,
    ).run()

    if args.strandedness is not None:
        strand_by_sample = {s.sample_id: args.strandedness for s in args.samples}
    else:
        strandedness: StrandednessOutputs = InferStrandedness(
            samples=args.samples,
            index_prefix=hisat2_index_output.index_prefix_path,
            gtf_path=args.gtf,
            dataset=dataset_name(args.genome),
            cache_dir=cache,
            splice_sites=hisat2_index_output.splice_sites_path,
            output_dir=out / "strandedness",
            threads=args.threads,
            stranded_threshold=args.stranded_threshold,
            hisat2_env=args.hisat2_env or None,
            rseqc_env=args.rseqc_env or None,
        ).run()
        strand_by_sample = {sid: r.strandness for sid, r in strandedness.samples.items()}

    alignments: AlignmentOutputs = AlignReads(
        samples=args.samples,
        index_prefix=hisat2_index_output.index_prefix_path,
        splice_sites=hisat2_index_output.splice_sites_path,
        dta=True,
        strandness=strand_by_sample,
        output_dir=out / "bam",
        threads=args.threads,
        hisat2_env=args.hisat2_env or None,
    ).run()

    assemblies: AssemblyOutputs = AssembleTranscripts(
        alignments=alignments,
        output_dir=out / "assembly",
        gtf_path=Path(args.gtf),
        threads=args.threads,
        extra_args=args.stringtie_assembly_args,
        stringtie_env=args.stringtie_env or None,
    ).run()

    merges: MergeOutputs = MergeAssemblies(
        samples=args.samples,
        assemblies=assemblies,
        gtf_path=Path(args.gtf),
        output_dir=out / "merge",
        reports_dir=out / "reports",
        cache_dir=cache,
        threads=args.threads,
        extra_args=args.stringtie_merge_args,
        stringtie_env=args.stringtie_env or None,
    ).run()

    counts: CountsOutputs = QuantifyCounts(
        samples=args.samples,
        alignments=alignments,
        merges=merges,
        output_dir=out / "counts",
        reports_dir=out / "reports",
        threads=args.threads,
        subread_env=args.subread_env or None,
    ).run()

    expressed: dict[str, ExpressionFilterOutputs] = {}
    for condition, merge in merges.conditions.items():
        logger.info("lncRNA filtering for condition %s", condition)
        cond_dir = out / "lncrna" / condition
        lnc = run_lncrna_filtering(bed_path=merge.bed, args=args, output_dir=cond_dir)
        fpkm: FPKMOutputs = ComputeFPKM(
            counts_path=counts.conditions[condition].counts,
            alignments=alignments,
            output_path=out / "counts" / f"{condition}_gene_fpkm.tsv",
        ).run()
        expressed[condition] = FilterByExpression(
            bed_path=lnc.pure_noncoding.bed_path,
            merged_gtf_path=merge.gtf,
            fpkm_path=fpkm.fpkm_path,
            output_dir=cond_dir / "expression",
            label=condition,
        ).run()

    combined: CombinedBedOutputs = CombineConditionBeds(
        condition_beds={cond: e.bed_path for cond, e in expressed.items()},
        output_dir=out / "combined",
    ).run()

    return build_features(combined.bed_path, args, out)


# ── modes: what to do with the feature table ────────────────────────────────
def run_inference(features: FeatureTableOutputs, args) -> PredictionOutputs:
    model_dir = Path(args.model_dir)
    return Predict(
        feature_table_path=features.table_path,
        model_path=model_dir / "final_model.keras",
        preprocessing_path=model_dir / "preprocessing.json",
        output_path=Path(args.output_dir) / "predictions.tsv",
    ).run()

def run_training(args) -> TrainOutputs:
    out = Path(args.output_dir)
    data: TrainingDataOutputs = PrepareTrainingData(
        bed_path=args.train_bed,
        output_dir=out / "training_input",
        label_col=args.label_col,
    ).run()

    # Filter each group on its own, as the reads route does per condition.
    filtered = []
    for group, bed in data.group_beds.items():
        logger.info("lncRNA filtering for group %s", group)
        lnc = run_lncrna_filtering(
            bed_path=bed,
            args=args,
            output_dir=out / "lncrna" / group if data.grouped else out / "lncrna",
        )
        filtered.append(lnc.pure_noncoding.bed_path)

    candidates = concat_beds(filtered, out / "lncrna" / "training_candidates.bed")
    features = build_features(candidates, args, out)

    return TrainModel(
        feature_table_path=features.table_path,
        labels_path=data.labels_path,
        output_dir=out / "model",
        k=args.k,
        epochs=args.epochs,
        label_col=args.label_col,
    ).run()
