"""
tune_threshold.py - post-hoc decision-threshold sweep for spike detection.

The default 0.5 threshold gives weak F1 even when AUC-ROC is high - a known
symptom of class imbalance plus the train/test positive-rate shift seen in
this dataset (train ~0.15% positives vs test ~2.4%). This script sweeps
thresholds in (0, 1) for each (model, horizon), reports the F1-optimal
threshold with its precision/recall, and compares against the default 0.5.

Caveat: tuning is performed on the TEST set, so the resulting F1 values are
an UPPER BOUND. For deployment you should tune on the validation split (add
val proba to the *_pred.npz files in each train script). Here it serves as
a diagnostic to confirm the AUC/F1 disconnect is a thresholding issue, not
a discrimination issue.

Run:  python ml_pipeline/tune_threshold.py
Inputs:  artifacts/W{W}/results/{baseline,xgb,lstm}_pred.npz
Outputs: artifacts/W{W}/results/threshold_tuning.json
         artifacts/W{W}/results/threshold_tuning.md
"""

import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

from sklearn.metrics import (precision_recall_curve, f1_score,
                             precision_score, recall_score)


MODELS = ["baseline", "xgb", "lstm"]
DEFAULT_THRESHOLD = 0.5


def _metrics_at(threshold, y_true, proba):
    y_pred = (proba >= threshold).astype(int)
    return {
        "threshold":  round(float(threshold), 4),
        "f1":         round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision":  round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":     round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "n_pos_pred": int(y_pred.sum()),
    }


def _sweep_f1(y_true, proba):
    """F1-optimal threshold using PR-curve unique-probability candidates."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # precision_recall_curve returns one extra precision/recall (the (1,0) point)
    # with no corresponding threshold - align lengths.
    p = precision[:-1]
    r = recall[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = 2 * p * r / (p + r)
        f1 = np.nan_to_num(f1, nan=0.0)
    if f1.size == 0:
        return None
    return float(thresholds[int(np.argmax(f1))])


def tune_one(model_name):
    """Load pred npz, sweep threshold per horizon, return dict or None."""
    pred_path = C.RESULTS_DIR / f"{model_name}_pred.npz"
    if not pred_path.exists():
        print(f"  [skip] {pred_path.name} not found")
        return None
    z = np.load(pred_path, allow_pickle=True)

    out = {}
    for H in C.HORIZONS:
        key_proba = f"clf_proba_h{H}"
        key_y     = f"y_clf_h{H}"
        if key_proba not in z.files or key_y not in z.files:
            continue
        proba  = z[key_proba].astype(np.float64)
        y_true = z[key_y].astype(int)

        default = _metrics_at(DEFAULT_THRESHOLD, y_true, proba)
        n_pos   = int(y_true.sum())

        if n_pos == 0:
            out[str(H)] = {"n_total": int(len(y_true)), "n_pos": 0,
                           "note": "no positives in test set",
                           "default": default}
            continue

        best_thr = _sweep_f1(y_true, proba)
        if best_thr is None:
            out[str(H)] = {"n_total": int(len(y_true)), "n_pos": n_pos,
                           "default": default}
            continue

        optimal = _metrics_at(best_thr, y_true, proba)
        out[str(H)] = {
            "n_total":  int(len(y_true)),
            "n_pos":    n_pos,
            "pos_rate": round(n_pos / len(y_true), 5),
            "default":  default,
            "optimal":  optimal,
            "f1_gain":  round(optimal["f1"] - default["f1"], 4),
        }
    return out


def _md_row(model, H, b):
    d = b["default"]; o = b["optimal"]
    return (f"| {model} | {H} | {d['f1']:.3f} | {d['precision']:.3f} | {d['recall']:.3f} "
            f"| {o['threshold']:.3f} | {o['f1']:.3f} | {o['precision']:.3f} | {o['recall']:.3f} "
            f"| +{b['f1_gain']:.3f} |")


def write_markdown(all_results, out_path):
    lines = [
        "# Threshold tuning (post-hoc, on test set)",
        "",
        "F1-optimal decision threshold per (model, horizon). **Caveat:** tuned",
        "on the test set, so these are UPPER BOUNDS on what threshold tuning",
        "alone can buy you. For deployment, tune on the validation split.",
        "",
        "| model | H (min) | F1@0.5 | P@0.5 | R@0.5 | thr\\* | F1\\* | P\\* | R\\* | ΔF1 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for model, res in all_results.items():
        if res is None:
            continue
        for H in C.HORIZONS:
            cell = res.get(str(H))
            if not cell or "optimal" not in cell:
                continue
            lines.append(_md_row(model, H, cell))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    print("=" * 70)
    print(" THRESHOLD TUNING - F1-optimal threshold per (model, horizon)")
    print(f" reading from: {C.RESULTS_DIR}")
    print("=" * 70)

    if not C.RESULTS_DIR.exists():
        print(f"  [error] results dir does not exist: {C.RESULTS_DIR}")
        print(f"          run train_baseline.py / train_xgb.py / train_lstm.py first")
        return

    all_results = {}
    for m in MODELS:
        print(f"\n  {m}")
        r = tune_one(m)
        if r is None:
            continue
        all_results[m] = r
        for H in C.HORIZONS:
            cell = r.get(str(H))
            if not cell or "optimal" not in cell:
                if cell and "note" in cell:
                    print(f"    H={H:2d}min  {cell['note']}")
                continue
            d = cell["default"]; o = cell["optimal"]
            print(f"    H={H:2d}min  F1@0.5={d['f1']:.3f} -> "
                  f"F1*={o['f1']:.3f} @ thr={o['threshold']:.3f}  "
                  f"(P={o['precision']:.3f}  R={o['recall']:.3f})")

    out_json = C.RESULTS_DIR / "threshold_tuning.json"
    out_md   = C.RESULTS_DIR / "threshold_tuning.md"
    out_json.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    write_markdown(all_results, out_md)
    print(f"\n  wrote {out_json.name}")
    print(f"  wrote {out_md.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
