"""
train_xgb.py - Gradient-boosted trees (XGBoost), GPU-enabled.

Primary non-sequential model. Same tabular feature vector as the baseline, so
any gain over the baseline is the model, not the features.
  * 36-combo grid search (learning_rate / max_depth / subsample /
    colsample_bytree) with early stopping on the VALIDATION split.
  * Regression     : XGBRegressor, picked by lowest validation RMSE.
  * Classification : XGBClassifier, scale_pos_weight = n_neg/n_pos, picked by F1.
  * Device         : auto-detects CUDA via torch; falls back to CPU.
  * Gain importances saved for the paper's feature-importance analysis.

Run:  python ml_pipeline/train_xgb.py     (needs: pip install -U "xgboost>=2.0")
Outputs: artifacts/results/xgb.json , artifacts/results/xgb_pred.npz
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


def _detect_device():
    """CUDA if a CUDA build of torch sees a GPU, else CPU. Override below."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


XGB_DEVICE = _detect_device()              # force "cuda" / "cpu" here if needed

GRID = {
    "learning_rate":    [0.05, 0.1, 0.2],
    "max_depth":        [4, 6, 8],
    "subsample":        [0.7, 1.0],
    "colsample_bytree": [0.7, 1.0],
}                                          # 3*3*2*2 = 36 combinations
N_ESTIMATORS_MAX  = 1000                   # capped by early stopping
EARLY_STOP_ROUNDS = 30
COMMON = dict(tree_method="hist", device=XGB_DEVICE, n_jobs=-1,
              random_state=C.RANDOM_SEED)


def _combos():
    keys = list(GRID)
    for vals in itertools.product(*(GRID[k] for k in keys)):
        yield dict(zip(keys, vals))


def _fit_one(estimator_cls, params, Xtr, ytr, Xva, yva):
    """Fit one XGBoost model with early stopping on the validation set."""
    model = estimator_cls(n_estimators=N_ESTIMATORS_MAX,
                          early_stopping_rounds=EARLY_STOP_ROUNDS,
                          **params, **COMMON)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    # training ran on GPU; switch the booster to CPU for prediction so the
    # NumPy eval/test arrays don't trigger a device-mismatch fallback warning
    # (predictions are identical — the trained trees are device-agnostic)
    if XGB_DEVICE != "cpu":
        model.get_booster().set_param({"device": "cpu"})
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
            val_proba = model.predict_proba(Xva)[:, 1]
            score = score_fn(yva, (val_proba >= 0.5).astype(int))
        else:
            score = score_fn(yva, model.predict(Xva))
        better = score > best[2] if maximise else score < best[2]
        if better:
            best = (model, p, score)
    return best


def _named_importance(model, feat_names):
    """Map xgboost's f0/f1/... gain importances back to readable names."""
    imp = model.get_booster().get_score(importance_type="gain")
    return {feat_names[int(k[1:])]: round(float(v), 4) for k, v in imp.items()}


def main():
    set_seed()
    t0 = time.time()
    print("=" * 70)
    print(" XGBOOST - gradient-boosted trees")
    print(f" device: {XGB_DEVICE}   grid: {len(list(_combos()))} combos "
          f"x early-stopping  (per horizon, per task)")
    if XGB_DEVICE == "cpu":
        print(" (CUDA not detected; set XGB_DEVICE='cuda' to force GPU)")
    print("=" * 70)

    from sklearn.metrics import mean_squared_error, f1_score

    def rmse(y_true, y_pred):
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    feat_names = tabular_feature_names(list(C.FEATURE_COLS), C.WINDOW_SIZE)
    results = {"model": "xgboost", "window_size": C.WINDOW_SIZE,
               "device": XGB_DEVICE, "grid": GRID, "horizons": {}}
    preds = {}

    for H in C.HORIZONS:
        d = load_horizon(H)
        Xtr = extract_tabular_features(d["X_train"])
        Xva = extract_tabular_features(d["X_val"])
        Xte = extract_tabular_features(d["X_test"])

        # regression - pick lowest validation RMSE
        reg_model, reg_params, _ = _grid_search(
            xgb.XGBRegressor, Xtr, d["y_reg_train"], Xva, d["y_reg_val"],
            score_fn=rmse, maximise=False,
            extra_params={"objective": "reg:squarederror"})
        yhat = reg_model.predict(Xte)
        reg_m = regression_metrics(d["y_reg_test"], yhat)
        reg_m["best_params"] = {k: v for k, v in reg_params.items() if k in GRID}
        reg_m["best_n_estimators"] = int(reg_model.best_iteration + 1)
        reg_m["feature_importance_gain"] = _named_importance(reg_model, feat_names)

        # classification - pick best validation F1
        ytr_c = d["y_clf_train"]
        n_pos = int(ytr_c.sum())
        spw = (len(ytr_c) - n_pos) / max(n_pos, 1)
        clf_model, clf_params, _ = _grid_search(
            xgb.XGBClassifier, Xtr, ytr_c, Xva, d["y_clf_val"],
            score_fn=lambda yt, yp: f1_score(yt, yp, zero_division=0),
            maximise=True,
            extra_params={"objective": "binary:logistic",
                          "scale_pos_weight": spw, "eval_metric": "logloss"})
        proba = clf_model.predict_proba(Xte)[:, 1]
        clf_m = classification_metrics(d["y_clf_test"],
                                       (proba >= 0.5).astype(int), proba)
        clf_m["best_params"] = {k: v for k, v in clf_params.items() if k in GRID}
        clf_m["best_n_estimators"] = int(clf_model.best_iteration + 1)
        clf_m["scale_pos_weight"] = round(spw, 2)
        clf_m["feature_importance_gain"] = _named_importance(clf_model, feat_names)

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
