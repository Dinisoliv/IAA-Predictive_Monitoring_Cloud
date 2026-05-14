"""
train_xgb.py — Gradient-boosted trees (XGBoost).
================================================================================
The primary non-sequential model. Receives the *same* tabular feature vector as
the baseline, so any gain over the baseline is attributable to the model, not
the features.

  * Small grid search over learning_rate / max_depth / subsample, with
    early stopping on the VALIDATION split (n_estimators chosen automatically).
  * Regression     : XGBRegressor, selected by lowest validation RMSE.
  * Classification : XGBClassifier with scale_pos_weight = n_neg / n_pos
                     (handles the ~2.4 % positive rate), selected by best
                     validation F1.
  * TEST split is touched only once, for the final reported numbers.
  * Gain-based feature importances are saved for the paper's analysis of which
    metrics / temporal statistics drive the forecast.

Run:
    python ml_pipeline/train_xgb.py

Outputs:
    artifacts/results/xgb.json         (metrics + best params + importances)
    artifacts/results/xgb_pred.npz     (test predictions, for the figures)
================================================================================
"""

import sys
import time
import itertools
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from common import (set_seed, load_horizon, extract_tabular_features,
                    tabular_feature_names, regression_metrics,
                    classification_metrics, save_json, print_horizon_line)

import xgboost as xgb

# ── Small hyper-parameter grid (edit to widen the search) ──────────────────────
GRID = {
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth":     [4, 6, 8],
    "subsample":     [0.7, 1.0],
}                                              # 3 * 3 * 2 = 18 combinations
N_ESTIMATORS_MAX   = 1000                      # capped by early stopping
EARLY_STOP_ROUNDS  = 30
COMMON = dict(tree_method="hist", n_jobs=-1,
              colsample_bytree=0.8, random_state=C.RANDOM_SEED)


def _combos():
    keys = list(GRID)
    for vals in itertools.product(*(GRID[k] for k in keys)):
        yield dict(zip(keys, vals))


def _fit_one(estimator_cls, params, Xtr, ytr, Xva, yva):
    """Fit a single XGBoost model with early stopping on the validation set."""
    model = estimator_cls(
        n_estimators=N_ESTIMATORS_MAX,
        early_stopping_rounds=EARLY_STOP_ROUNDS,
        **params, **COMMON,
    )
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return model


def _grid_search(estimator_cls, Xtr, ytr, Xva, yva, score_fn, maximise,
                 extra_params=None):
    """Return (best_model, best_params, best_val_score)."""
    best = (None, None, -np.inf if maximise else np.inf)
    for params in _combos():
        p = dict(params)
        if extra_params:
            p.update(extra_params)
        model = _fit_one(estimator_cls, p, Xtr, ytr, Xva, yva)
        if estimator_cls is xgb.XGBClassifier:
            val_pred_proba = model.predict_proba(Xva)[:, 1]
            val_pred = (val_pred_proba >= 0.5).astype(int)
            score = score_fn(yva, val_pred)
        else:
            score = score_fn(yva, model.predict(Xva))
        better = score > best[2] if maximise else score < best[2]
        if better:
            best = (model, p, score)
    return best


def main():
    set_seed()
    t0 = time.time()
    print("=" * 70)
    print(" XGBOOST — gradient-boosted trees")
    print(f" grid: {len(list(_combos()))} combos x early-stopping  "
          f"(per horizon, per task)")
    print("=" * 70)

    from sklearn.metrics import mean_squared_error, f1_score
    # rmse helper that works across sklearn versions (no `squared=` kwarg)
    def rmse(y_true, y_pred):
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    feat_names = tabular_feature_names(list(C.FEATURE_COLS), C.WINDOW_SIZE)

    results = {"model": "xgboost", "window_size": C.WINDOW_SIZE,
               "grid": GRID, "horizons": {}}
    preds = {}

    for H in C.HORIZONS:
        d = load_horizon(H)
        Xtr = extract_tabular_features(d["X_train"])
        Xva = extract_tabular_features(d["X_val"])
        Xte = extract_tabular_features(d["X_test"])

        # ── Regression: pick lowest validation RMSE ──────────────────────────
        reg_model, reg_params, _ = _grid_search(
            xgb.XGBRegressor, Xtr, d["y_reg_train"], Xva, d["y_reg_val"],
            score_fn=rmse, maximise=False,
            extra_params={"objective": "reg:squarederror"},
        )
        yhat = reg_model.predict(Xte)
        reg_m = regression_metrics(d["y_reg_test"], yhat)
        reg_m["best_params"] = {k: v for k, v in reg_params.items()
                                if k in GRID}
        reg_m["best_n_estimators"] = int(reg_model.best_iteration + 1)

        # ── Classification: pick best validation F1 ──────────────────────────
        ytr_c = d["y_clf_train"]
        n_pos = int(ytr_c.sum())
        n_neg = int(len(ytr_c) - n_pos)
        spw = n_neg / max(n_pos, 1)
        clf_model, clf_params, _ = _grid_search(
            xgb.XGBClassifier, Xtr, ytr_c, Xva, d["y_clf_val"],
            score_fn=lambda yt, yp: f1_score(yt, yp, zero_division=0),
            maximise=True,
            extra_params={"objective": "binary:logistic",
                          "scale_pos_weight": spw, "eval_metric": "logloss"},
        )
        proba = clf_model.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        clf_m = classification_metrics(d["y_clf_test"], pred, proba)
        clf_m["best_params"] = {k: v for k, v in clf_params.items()
                                if k in GRID}
        clf_m["best_n_estimators"] = int(clf_model.best_iteration + 1)
        clf_m["scale_pos_weight"] = round(spw, 2)

        # ── Feature importances (gain) ───────────────────────────────────────
        reg_imp = reg_model.get_booster().get_score(importance_type="gain")
        clf_imp = clf_model.get_booster().get_score(importance_type="gain")
        # xgboost keys are f0, f1, ... -> map back to readable names
        def _named(imp):
            return {feat_names[int(k[1:])]: round(float(v), 4)
                    for k, v in imp.items()}
        reg_m["feature_importance_gain"] = _named(reg_imp)
        clf_m["feature_importance_gain"] = _named(clf_imp)

        results["horizons"][str(H)] = {"regression": reg_m,
                                       "classification": clf_m}
        preds[f"reg_h{H}"]       = yhat.astype(np.float32)
        preds[f"clf_proba_h{H}"] = proba.astype(np.float32)
        preds[f"y_reg_h{H}"]     = d["y_reg_test"].astype(np.float32)
        preds[f"y_clf_h{H}"]     = d["y_clf_test"].astype(np.int8)
        preds[f"m_test_h{H}"]    = d["m_test"]
        print_horizon_line(H, reg_m, clf_m)

    results["runtime_s"] = round(time.time() - t0, 1)
    save_json(results, C.RESULTS_DIR / "xgb.json")
    np.savez_compressed(C.RESULTS_DIR / "xgb_pred.npz", **preds)
    print(f"\n  done in {results['runtime_s']}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
