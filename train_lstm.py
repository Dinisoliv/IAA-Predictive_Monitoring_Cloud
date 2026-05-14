"""
train_lstm.py — Long Short-Term Memory network (PyTorch).
================================================================================
Captures temporal dependencies that feature aggregation throws away. Receives
the RAW windowed sequence (W x 5), no manual feature engineering.

  * Architecture   : 1 LSTM layer (hidden=64) -> dense head. Kept deliberately
                     small — Intel Iris is not a CUDA/MPS device, so this runs
                     on CPU; a lean model keeps each cell to a few minutes.
  * Regression     : target scaled to [0,1] (CPU% / 100) for training stability,
                     predictions rescaled to % before scoring. MSE loss.
  * Classification : BCEWithLogitsLoss with pos_weight = n_neg / n_pos for the
                     ~2.4 % positive rate.
  * Early stopping : on validation loss, patience 5, max 30 epochs.
  * If a cell is too slow on your machine, set LSTM_MAX_TRAIN_WINDOWS below to
    e.g. 200_000 to subsample the training windows.

Run:
    python ml_pipeline/train_lstm.py        (needs: pip install torch)

Outputs:
    artifacts/results/lstm.json       (metrics + learning curves)
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

# ── LSTM hyper-parameters (edit here) ─────────────────────────────────────────
HIDDEN_SIZE   = 64
NUM_LAYERS    = 1
LEARNING_RATE = 1e-3
BATCH_SIZE    = 512
MAX_EPOCHS    = 30
PATIENCE      = 5
LSTM_MAX_TRAIN_WINDOWS = None      # e.g. 200_000 to subsample; None = use all


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")      # Intel Iris lands here


class LSTMForecaster(nn.Module):
    """1-layer LSTM -> dense head. One scalar output (value or logit)."""
    def __init__(self, n_features: int, hidden: int, layers: int):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)              # out: (B, W, hidden)
        return self.head(out[:, -1, :]).squeeze(-1)   # last timestep -> (B,)


def make_loader(X, y, shuffle, seed=C.RANDOM_SEED):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, generator=g)


def train_model(model, train_loader, val_loader, loss_fn, device):
    """Train with early stopping on validation loss. Returns (model, history)."""
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val, best_state, bad = float("inf"), None, 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(MAX_EPOCHS):
        model.train()
        tr_sum = tr_n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tr_sum += loss.item() * len(xb); tr_n += len(xb)

        model.eval()
        va_sum = va_n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
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
    loader = DataLoader(TensorDataset(torch.from_numpy(X)),
                        batch_size=BATCH_SIZE, shuffle=False)
    out = []
    with torch.no_grad():
        for (xb,) in loader:
            out.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(out)


def subsample(X, y, seed=C.RANDOM_SEED):
    if LSTM_MAX_TRAIN_WINDOWS is None or len(X) <= LSTM_MAX_TRAIN_WINDOWS:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), LSTM_MAX_TRAIN_WINDOWS, replace=False)
    return X[idx], y[idx]


def main():
    set_seed()
    t0 = time.time()
    device = get_device()
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" LSTM — 1-layer recurrent network")
    print(f" device: {device}   hidden={HIDDEN_SIZE}  batch={BATCH_SIZE}  "
          f"max_epochs={MAX_EPOCHS}")
    if device.type == "cpu":
        print(" (running on CPU — Intel Iris is not a CUDA/MPS device)")
    print("=" * 70)

    n_features = len(C.FEATURE_COLS)
    results = {"model": "lstm", "window_size": C.WINDOW_SIZE,
               "device": str(device), "arch": {"hidden": HIDDEN_SIZE,
               "layers": NUM_LAYERS}, "horizons": {}}
    preds = {}

    for H in C.HORIZONS:
        d = load_horizon(H)
        Xtr, Xva, Xte = d["X_train"], d["X_val"], d["X_test"]

        # ── Regression (target scaled to [0,1] for training) ─────────────────
        ytr = (d["y_reg_train"] / 100.0).astype(np.float32)
        yva = (d["y_reg_val"]   / 100.0).astype(np.float32)
        Xtr_s, ytr_s = subsample(Xtr, ytr)
        model = LSTMForecaster(n_features, HIDDEN_SIZE, NUM_LAYERS).to(device)
        model, hist_r = train_model(
            model,
            make_loader(Xtr_s, ytr_s, shuffle=True),
            make_loader(Xva, yva, shuffle=False),
            nn.MSELoss(), device)
        yhat = predict(model, Xte, device) * 100.0          # back to %
        reg_m = regression_metrics(d["y_reg_test"], yhat)
        reg_m["learning_curve"] = hist_r
        torch.save(model.state_dict(), C.MODELS_DIR / f"lstm_reg_h{H}.pt")

        # ── Classification (BCEWithLogits + pos_weight) ──────────────────────
        ytr_c = d["y_clf_train"].astype(np.float32)
        yva_c = d["y_clf_val"].astype(np.float32)
        n_pos = int(ytr_c.sum()); n_neg = len(ytr_c) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
        Xtr_c, ytr_cs = subsample(Xtr, ytr_c)
        model_c = LSTMForecaster(n_features, HIDDEN_SIZE, NUM_LAYERS).to(device)
        model_c, hist_c = train_model(
            model_c,
            make_loader(Xtr_c, ytr_cs, shuffle=True),
            make_loader(Xva, yva_c, shuffle=False),
            nn.BCEWithLogitsLoss(pos_weight=pos_weight), device)
        logits = predict(model_c, Xte, device)
        proba = 1.0 / (1.0 + np.exp(-logits))
        pred = (proba >= 0.5).astype(int)
        clf_m = classification_metrics(d["y_clf_test"], pred, proba)
        clf_m["learning_curve"] = hist_c
        clf_m["pos_weight"] = round(float(pos_weight.item()), 2)
        torch.save(model_c.state_dict(), C.MODELS_DIR / f"lstm_clf_h{H}.pt")

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
