#!/usr/bin/env python
# tools/cv_harness.py
"""Cross-validation for Flnc2 models.

Input: training_table.tsv from `flnc2 --mode train` (features + label).
With --group-by-name, names of the form 'group|name' define the folds
(leave-one-group-out). Otherwise stratified k-fold; note that identical
transcripts under different names can then land in both train and eval.

Reports per-fold metrics and a recommended --epochs (median best epoch).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

from src.py.combine_conditions_beds import split_name
from src.py.ml.model import fit_model, plot_metric
from src.py.ml.preprocessing import (fit_preprocessing, kmer_feature_cols,
                                     locus_feature_cols, transform)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table", type=Path)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--label_col", default="H3K4me3")
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--group_by_name", action="store_true")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--max_epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=47)
    args = ap.parse_args()

    df = pd.read_table(args.table)
    y = df[args.label_col].to_numpy(dtype="float32")
    kmer_cols, locus_cols = kmer_feature_cols(args.k), locus_feature_cols(df)

    if args.group_by_name:
        groups = df["name"].map(lambda n: split_name(n)[0]).to_numpy()
        splits = list(LeaveOneGroupOut().split(df, y, groups))
        labels = [groups[ev[0]] for _, ev in splits]
    else:
        skf = StratifiedKFold(args.n_splits, shuffle=True, random_state=args.seed)
        splits = list(skf.split(df, y))
        labels = [f"split{i}" for i in range(1, len(splits) + 1)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results, oof = [], []

    for fold, ((tr, ev), held_out) in enumerate(zip(splits, labels), start=1):
        print(f"Fold {fold}/{len(splits)}: holding out {held_out}")
        if len(np.unique(y[ev])) < 2:
            print("  SKIP: held-out set has only one class")
            continue

        fold_dir = args.output_dir / f"fold_{fold}_{held_out}"
        (fold_dir / "plots").mkdir(parents=True, exist_ok=True)

        # Fitted on this fold's training rows only.
        prep = fit_preprocessing(df.iloc[tr], kmer_cols, locus_cols, args.label_col)
        X_tr, X_ev = transform(df.iloc[tr], prep), transform(df.iloc[ev], prep)
        model, history = fit_model(X_tr, y[tr], epochs=args.max_epochs,
                                   batch_size=args.batch_size, seed=args.seed,
                                   validation=(X_ev, y[ev]),
                                   log_path=fold_dir / "training_log.csv")
        model.save(fold_dir / "model.keras")
        prep.save(fold_dir / "preprocessing.json")
        for metric, title in (("loss", "Loss"), ("auroc", "AUROC"), ("auprc", "AUPRC")):
            plot_metric(history, metric, f"val_{metric}", f"{title} ({held_out} held out)",
                        title, fold_dir / "plots" / f"{metric}.png")

        y_pred = model.predict(X_ev, verbose=0).ravel()
        res = {
            "fold": fold, "held_out": held_out, "n_train": len(tr), "n_eval": len(ev),
            "eval_pos_rate": float(y[ev].mean()),
            "auroc": roc_auc_score(y[ev], y_pred),
            "auprc": average_precision_score(y[ev], y_pred),
            "accuracy": accuracy_score(y[ev], y_pred > 0.5),
            "best_epoch": int(np.argmax(history.history["val_auroc"]) + 1),
        }
        results.append(res)
        print(f"  AUROC {res['auroc']:.4f} | AUPRC {res['auprc']:.4f} | best epoch {res['best_epoch']}")
        oof.append(pd.DataFrame({"name": df["name"].iloc[ev].values, "held_out": held_out,
                                 "y_true": y[ev], "y_pred": y_pred}))

    if not results:
        raise SystemExit("Every held-out set had a single class; nothing to evaluate.")

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_dir / "cv_results.tsv", sep="\t", index=False)
    oof_df = pd.concat(oof, ignore_index=True)
    oof_df.to_csv(args.output_dir / "oof_predictions.tsv", sep="\t", index=False)

    summary = {m: {"mean": float(results_df[m].mean()), "sd": float(results_df[m].std(ddof=1))}
               for m in ("auroc", "auprc", "accuracy")}
    summary["pooled_auroc"] = float(roc_auc_score(oof_df["y_true"], oof_df["y_pred"]))
    summary["recommended_epochs"] = int(results_df["best_epoch"].median())
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + results_df.to_string(index=False))
    for m in ("auroc", "auprc", "accuracy"):
        print(f"{m.upper():8s}: {summary[m]['mean']:.4f} ± {summary[m]['sd']:.4f}")
    print(f"Pooled AUROC: {summary['pooled_auroc']:.4f}")
    print(f"Recommended --epochs: {summary['recommended_epochs']}")


if __name__ == "__main__":
    main()
