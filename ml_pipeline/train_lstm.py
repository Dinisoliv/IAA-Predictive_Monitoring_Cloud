"""
train_lstm.py — LSTM with a validation-based hyperparameter search (PyTorch).
================================================================================
Captures temporal dependencies that feature aggregation throws away. Receives
the RAW windowed sequence (W x F), no manual feature engineering.

HYPERPARAMETER SEARCH
  A grid over {hidden_size, num_layers, learning_rate, dropout} is evaluated on
  the VALIDATION split. This is what makes the paper's claim — "start simple,
  add complexity only if validation justifies it" — actually true.

    * SEARCH_HORIZON = 15  -> search once (per task) at H=15, reuse the winning
                              config to train final models at every horizon.
                              Good GPU-time / rigour balance. (default)
    * SEARCH_HORIZON = None -> search independently for every horizon x task.
                               Most rigorous, ~Nconfigs x 8 trainings.

  During the search, training windows are subsampled (LSTM_SEARCH_SUBSAMPLE) for
  speed; the FINAL models train on the full split.

  Every config's validation score is logged into lstm.json so the search itself
  is reportable.

DEVICE
  Auto-detects CUDA -> MPS -> CPU. On the remote GPU box this runs on CUDA.

  * Regression     : target scaled to [0,1] (CPU%/100), predictions rescaled.
  * Classification : BCEWithLogitsLoss, pos_weight = n_neg / n_pos (~2.4 % pos).
  * Early stopping : on validation loss.

Run:
    python ml_pipeline/train_lstm.py        (needs: pip install torch)

Outputs:
    artifacts/results/lstm.json       (metrics + hp-search log + learning curves)
    artifacts/results/lstm_pred.npz   (test predictions, for the figures)
    artifacts/models/lstm_<task>_h<H>.pt
================================================================================
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from common import (set_seed, load_horizon, regression_metrics,
                    classification_metrics, save_json, print_horizon_line)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ── Hyperparameter grid (edit here) ───────────────────────────────────────────
HIDDEN_SIZES    = [64, 128]
NUM_LAYERS_OPTS = [1, 2]
LEARNING_RATES  = [1e-3, 5e-4]
DROPOUTS        = [0.0, 0.2]          # only applied when num_layers >= 2

# ── Training budget ───────────────────────────────────────────────────────────
BATCH_SIZE  = 512
MAX_EPOCHS  = 40
PATIENCE    = 6

# ── Search strategy ───────────────────────────────────────────────────────────
SEARCH_HORIZON         = 15           # int -> search there & reuse; None -> per cell
LSTM_SEARCH_SUBSAMPLE  = 150_000      # train windows used DURING the search; None=all
LSTM_MAX_TRAIN_WINDOWS = None         # cap for FINAL training; None = use all


# ──────────────────────────────────────────────────────────────────────────────
#  Device / model
# ──────────────────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class LSTMForecaster(nn.Module):
    """N-layer LSTM -> dense head. One scalar output (value or logit)."""
    def __init__(self, n_features, hidden, layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
#  Data plumbing
# ──────────────────────────────────────────────────────────────────────────────

def make_loader(X, y, shuffle, device, seed=C.RANDOM_SEED):
    ds = TensorDataset(torch.from_numpy(np.ascontiguousarray(X)),
                       torch.from_numpy(np.ascontiguousarray(y)))
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, generator=g,
                      pin_memory=(device.type == "cuda"))


def subsample(X, y, cap, seed=C.RANDOM_SEED):
    if cap is None or len(X) <= cap:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), cap, replace=False)
    return X[idx], y[idx]


# ──────────────────────────────────────────────────────────────────────────────
#  Train / predict
# ──────────────────────────────────────────────────────────────────────────────

def train_one(cfg, Xtr, ytr, Xva, yva, loss_fn, device):
    """Train a single config with early stopping. Returns (model, history)."""
    set_seed()                                          # same init for fair compare
    model = LSTMForecaster(Xtr.shape[2], cfg["hidden"],
                           cfg["layers"], cfg["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    tr_loader = make_loader(Xtr, ytr, shuffle=True,  device=device)
    va_loader = make_loader(Xva, yva, shuffle=False, device=device)

    best_val, best_state, bad = float("inf"), None, 0
    history = {"train_loss": [], "val_loss": []}

    for _ in range(MAX_EPOCHS):
        model.train()
        tr_sum = tr_n = 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tr_sum += loss.item() * len(xb); tr_n += len(xb)

        model.eval()
        va_sum = va_n = 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                va_sum += loss_fn(model(xb), yb).item() * len(xb); va_n += len(xb)

        tr_loss, va_loss = tr_sum / tr_n, va_sum / va_n
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val_loss"] = best_val
    history["epochs_run"] = len(history["train_loss"])
    return model, history


def predict(model, X, device):
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(np.ascontiguousarray(X))),
                        batch_size=BATCH_SIZE, shuffle=False)
    out = []
    with torch.no_grad():
        for (xb,) in loader:
            out.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(out)


def build_grid():
    """All configs; dropout is forced to 0 for 1-layer models, then deduped."""
    grid, seen = [], set()
    for h in HIDDEN_SIZES:
        for nl in NUM_LAYERS_OPTS:
            for lr in LEARNING_RATES:
                for dr in DROPOUTS:
                    d = dr if nl > 1 else 0.0
                    key = (h, nl, lr, d)
                    if key in seen:
                        continue
                    seen.add(key)
                    grid.append({"hidden": h, "layers": nl,
                                 "lr": lr, "dropout": d})
    return grid


def search(Xtr, ytr, Xva, yva, loss_fn, device, grid, label):
    """Grid search on the validation split. Returns (best_cfg, search_log)."""
    Xs, ys = subsample(Xtr, ytr, LSTM_SEARCH_SUBSAMPLE)
    print(f"    [{label}] searching {len(grid)} configs "
          f"on {len(Xs):,} train / {len(Xva):,} val windows")
    log, best_cfg, best_vl = [], None, float("inf")
    for i, cfg in enumerate(grid, 1):
        _, hist = train_one(cfg, Xs, ys, Xva, yva, loss_fn, device)
        vl = hist["best_val_loss"]
        log.append({**cfg, "val_loss": round(vl, 6),
                    "epochs": hist["epochs_run"]})
        tag = ""
        if vl < best_vl:
            best_cfg, best_vl, tag = cfg, vl, "  <- best"
        print(f"      {i:>2}/{len(grid)}  h{cfg['hidden']} L{cfg['layers']} "
              f"lr{cfg['lr']:.0e} d{cfg['dropout']}  val={vl:.5f}{tag}")
    return best_cfg, log


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def _clf_pos_weight(y, device):
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    return torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)


def main():
    set_seed()
    t0 = time.time()
    device = get_device()
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    grid = build_grid()

    print("=" * 70)
    print(" LSTM — recurrent network with validation-based HP search")
    print(f" device: {device}   grid: {len(grid)} configs   "
          f"search_horizon: {SEARCH_HORIZON}")
    if device.type != "cuda":
        print(" WARNING: CUDA not detected — this will be slow. Check the GPU box.")
    print("=" * 70)

    results = {"model": "lstm", "window_size": C.WINDOW_SIZE,
               "device": str(device),
               "grid": {"hidden": HIDDEN_SIZES, "layers": NUM_LAYERS_OPTS,
                        "lr": LEARNING_RATES, "dropout": DROPOUTS},
               "search_horizon": SEARCH_HORIZON,
               "search_log": {}, "horizons": {}}
    preds = {}

    # ── Optionally search once at SEARCH_HORIZON and reuse the winners ────────
    reg_cfg = clf_cfg = None
    if SEARCH_HORIZON is not None:
        d = load_horizon(SEARCH_HORIZON)
        reg_cfg, reg_log = search(
            d["X_train"], (d["y_reg_train"] / 100.0).astype(np.float32),
            d["X_val"],   (d["y_reg_val"]   / 100.0).astype(np.float32),
            nn.MSELoss(), device, grid, f"reg@H{SEARCH_HORIZON}")
        clf_cfg, clf_log = search(
            d["X_train"], d["y_clf_train"].astype(np.float32),
            d["X_val"],   d["y_clf_val"].astype(np.float32),
            nn.BCEWithLogitsLoss(pos_weight=_clf_pos_weight(d["y_clf_train"], device)),
            device, grid, f"clf@H{SEARCH_HORIZON}")
        results["search_log"] = {"regression": reg_log, "classification": clf_log}
        results["selected_config"] = {"regression": reg_cfg,
                                      "classification": clf_cfg}
        print(f"\n  selected  reg={reg_cfg}\n            clf={clf_cfg}\n")

    # ── Train final models for every horizon ─────────────────────────────────
    for H in C.HORIZONS:
        d = load_horizon(H)
        Xtr, Xva, Xte = d["X_train"], d["X_val"], d["X_test"]

        # ---- regression ----
        ytr = (d["y_reg_train"] / 100.0).astype(np.float32)
        yva = (d["y_reg_val"]   / 100.0).astype(np.float32)
        if SEARCH_HORIZON is None:
            reg_cfg, reg_log = search(Xtr, ytr, Xva, yva, nn.MSELoss(),
                                      device, grid, f"reg@H{H}")
            results["search_log"].setdefault("regression", {})[str(H)] = reg_log
        Xtr_f, ytr_f = subsample(Xtr, ytr, LSTM_MAX_TRAIN_WINDOWS)
        reg_model, hist_r = train_one(reg_cfg, Xtr_f, ytr_f, Xva, yva,
                                      nn.MSELoss(), device)
        yhat = predict(reg_model, Xte, device) * 100.0
        reg_m = regression_metrics(d["y_reg_test"], yhat)
        reg_m["config"] = reg_cfg
        reg_m["learning_curve"] = hist_r
        torch.save(reg_model.state_dict(), C.MODELS_DIR / f"lstm_reg_h{H}.pt")

        # ---- classification ----
        ytr_c = d["y_clf_train"].astype(np.float32)
        yva_c = d["y_clf_val"].astype(np.float32)
        pos_w = _clf_pos_weight(d["y_clf_train"], device)
        if SEARCH_HORIZON is None:
            clf_cfg, clf_log = search(Xtr, ytr_c, Xva, yva_c,
                                      nn.BCEWithLogitsLoss(pos_weight=pos_w),
                                      device, grid, f"clf@H{H}")
            results["search_log"].setdefault("classification", {})[str(H)] = clf_log
        Xtr_c, ytr_cs = subsample(Xtr, ytr_c, LSTM_MAX_TRAIN_WINDOWS)
        clf_model, hist_c = train_one(clf_cfg, Xtr_c, ytr_cs, Xva, yva_c,
                                      nn.BCEWithLogitsLoss(pos_weight=pos_w),
                                      device)
        logits = predict(clf_model, Xte, device)
        proba = 1.0 / (1.0 + np.exp(-logits))
        pred = (proba >= 0.5).astype(int)
        clf_m = classification_metrics(d["y_clf_test"], pred, proba)
        clf_m["config"] = clf_cfg
        clf_m["pos_weight"] = round(float(pos_w.item()), 2)
        clf_m["learning_curve"] = hist_c
        torch.save(clf_model.state_dict(), C.MODELS_DIR / f"lstm_clf_h{H}.pt")

        results["horizons"][str(H)] = {"regression": reg_m,
                                       "classification": clf_m}
        preds[f"reg_h{H}"]       = yhat.astype(np.float32)
        preds[f"clf_proba_h{H}"] = proba.astype(np.float32)
        preds[f"y_reg_h{H}"]     = d["y_reg_test"].astype(np.float32)
        preds[f"y_clf_h{H}"]     = d["y_clf_test"].astype(np.int8)
        preds[f"m_test_h{H}"]    = d["m_test"]
        print_horizon_line(H, reg_m, clf_m)

    results["runtime_s"] = round(time.time() - t0, 1)
    save_json(results, C.RESULTS_DIR / "lstm.json")
    np.savez_compressed(C.RESULTS_DIR / "lstm_pred.npz", **preds)
    print(f"\n  done in {results['runtime_s']}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
