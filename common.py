"""
common.py — shared helpers used by every training script.

Keeping feature extraction and metric computation in one place guarantees the
baseline, XGBoost and LSTM are scored *identically* — which is the whole point
of the model comparison in the paper.
"""

import json
import random
from pathlib import Path

import numpy as np

import config as C


# ──────────────────────────────────────────────────────────────────────────────
#  Reproducibility
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = C.RANDOM_SEED):
    """Seed python, numpy and (if installed) torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ──────────────────────────────────────────────────────────────────────────────
#  Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_horizon(H: int) -> dict:
    """Load one horizon's windowed tensors into a plain dict."""
    path = C.WINDOWS_DIR / f"h{H}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run preprocess.py first.")
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


# ──────────────────────────────────────────────────────────────────────────────
#  Tabular feature extraction (used by baseline + XGBoost)
# ──────────────────────────────────────────────────────────────────────────────

def extract_tabular_features(X: np.ndarray) -> np.ndarray:
    """Turn windowed sequences into a fixed-length feature vector.

    X: (n, W, F) scaled windows
    -> (n, W*F + 5*F): the flattened window augmented with, per metric,
       the mean / std / min / max / linear-slope over the window.

    Both non-sequential models (baseline, XGBoost) receive this same vector,
    so any performance gap between them reflects the model, not the features.
    """
    n, W, F = X.shape
    flat = X.reshape(n, W * F)

    mean = X.mean(axis=1)
    std  = X.std(axis=1)
    mn   = X.min(axis=1)
    mx   = X.max(axis=1)

    # least-squares slope of each metric across the window
    t = np.arange(W, dtype=np.float64)
    t_c = t - t.mean()
    denom = (t_c ** 2).sum()
    slope = (X * t_c[None, :, None]).sum(axis=1) / denom        # (n, F)

    return np.concatenate([flat, mean, std, mn, mx, slope],
                          axis=1).astype(np.float32)


def tabular_feature_names(feature_cols, W: int):
    """Names aligned with extract_tabular_features (for XGBoost importances)."""
    names = [f"{c}@w{w}" for w in range(W) for c in feature_cols]
    for stat in ("mean", "std", "min", "max", "slope"):
        names += [f"{c}_{stat}" for c in feature_cols]
    return names


# ──────────────────────────────────────────────────────────────────────────────
#  Metrics
# ──────────────────────────────────────────────────────────────────────────────

def regression_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                                 r2_score)
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return {
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2":   float(r2_score(y_true, y_pred)),
        "n":    int(len(y_true)),
    }


def classification_metrics(y_true, y_pred, y_proba) -> dict:
    from sklearn.metrics import (f1_score, precision_score, recall_score,
                                 roc_auc_score, confusion_matrix)
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    out = {
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "f1_macro":  float(f1_score(y_true, y_pred, average="macro",
                                    zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),          # [[TN, FP], [FN, TP]]
        "n":          int(len(y_true)),
        "n_positive": int(y_true.sum()),
    }
    try:
        out["auc_roc"] = float(roc_auc_score(y_true, y_proba))
    except ValueError:                            # only one class present
        out["auc_roc"] = None
    return out


# ──────────────────────────────────────────────────────────────────────────────
#  Result IO
# ──────────────────────────────────────────────────────────────────────────────

def save_json(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  wrote {path}")


def print_horizon_line(H, reg, clf):
    """One-line console summary for a finished (horizon) cell."""
    print(f"    H={H:>2}min | "
          f"REG  MAE {reg['mae']:6.3f}  RMSE {reg['rmse']:6.3f}  R2 {reg['r2']:+.3f}"
          f"  ||  CLF  F1 {clf['f1']:.3f}  P {clf['precision']:.3f}  "
          f"R {clf['recall']:.3f}  AUC {clf['auc_roc'] if clf['auc_roc'] is not None else float('nan'):.3f}")
