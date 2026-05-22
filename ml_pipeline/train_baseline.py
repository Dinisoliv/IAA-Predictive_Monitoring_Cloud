"""
train_baseline.py — Linear / Logistic Regression baseline.
================================================================================
The minimum-viable model. If XGBoost and the LSTM cannot beat this, then either
the task is trivially easy or the feature engineering is inadequate — so this
anchors the whole comparison.

  * Regression      : LinearRegression on the tabular feature vector.
  * Classification  : LogisticRegression(class_weight="balanced") — handles the
                      ~2.4 % positive rate without resampling.
  * Both use a StandardScaler (fitted on train only) for numerical stability.
  * Fitted on the TRAIN split; VAL is unused (the baseline has no hyper-params);
    TEST is used only for the reported numbers.

Run:
    python ml_pipeline/train_baseline.py

Outputs:
    artifacts/results/baseline.json        (all metrics, every horizon + task)
    artifacts/results/baseline_pred.npz    (test predictions, for the figures)
================================================================================
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from common import (set_seed, load_horizon, extract_tabular_features,
                    regression_metrics, classification_metrics,
                    save_json, print_horizon_line)

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def main():
    set_seed()
    t0 = time.time()
    print("=" * 70)
    print(" BASELINE — Linear / Logistic Regression")
    print("=" * 70)

    results = {"model": "baseline", "window_size": C.WINDOW_SIZE,
               "features": "tabular (flattened window + mean/std/min/max/slope)",
               "horizons": {}}
    preds = {}

    for H in C.HORIZONS:
        d = load_horizon(H)
        Xtr = extract_tabular_features(d["X_train"])
        Xte = extract_tabular_features(d["X_test"])

        # ── Regression ────────────────────────────────────────────────────────
        reg = make_pipeline(StandardScaler(), LinearRegression())
        reg.fit(Xtr, d["y_reg_train"])
        yhat = reg.predict(Xte)
        reg_m = regression_metrics(d["y_reg_test"], yhat)

        # ── Classification ────────────────────────────────────────────────────
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000,
                               random_state=C.RANDOM_SEED),
        )
        clf.fit(Xtr, d["y_clf_train"])
        proba = clf.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        clf_m = classification_metrics(d["y_clf_test"], pred, proba)

        results["horizons"][str(H)] = {"regression": reg_m,
                                       "classification": clf_m}
        preds[f"reg_h{H}"]       = yhat.astype(np.float32)
        preds[f"clf_proba_h{H}"] = proba.astype(np.float32)
        preds[f"y_reg_h{H}"]     = d["y_reg_test"].astype(np.float32)
        preds[f"y_clf_h{H}"]     = d["y_clf_test"].astype(np.int8)
        preds[f"m_test_h{H}"]    = d["m_test"]
        print_horizon_line(H, reg_m, clf_m)

    results["runtime_s"] = round(time.time() - t0, 1)
    save_json(results, C.RESULTS_DIR / "baseline.json")
    np.savez_compressed(C.RESULTS_DIR / "baseline_pred.npz", **preds)
    print(f"\n  done in {results['runtime_s']}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
