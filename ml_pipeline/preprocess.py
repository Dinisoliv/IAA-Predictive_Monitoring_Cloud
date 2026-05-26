"""
preprocess.py — cleaning, resampling, windowing and chronological split.
================================================================================
Pipeline (matches Sections 3.1-3.3 of Deliverable 2, with the corrections agreed
after inspecting the data):

  1. Load dataset_forecasting.csv (100 machines, ~6.15M raw 10 s rows).
  2. Per machine, in chronological order:
       a. Clean   — disk_io_percent codes -1/101 -> NaN; clip every metric to
                    [0,100], out-of-range -> NaN.
       b. Resample— average raw 10 s samples into a fixed 1-minute grid; bins
                    with no samples become NaN (these are the offline gaps).
       c. Interp  — linearly fill gaps of < 5 consecutive missing minutes;
                    longer gaps stay NaN.
       d. Split   — first 60 % of the machine timeline = train, next 20 % = val,
                    last 20 % = test (strictly chronological, no shuffling).
  3. Fit a global per-feature min-max scaler ON THE TRAINING ROWS ONLY, apply to
     all splits. The target (CPU %) is kept in original units so MAE/RMSE read
     as percentage points.
  4. Build sliding windows of length W for each horizon H. A window is kept only
     if every input row AND the target row are fully observed (no NaN) and the
     whole span lies inside a single split (no window crosses train/val/test or
     a long gap).
  5. Save one .npz per horizon plus a scaler.npz and a summary.json.

Run:
    python ml_pipeline/preprocess.py

Outputs (artifacts/windows/):
    h5.npz  h15.npz  h30.npz  h60.npz   scaler.npz   summary.json
================================================================================
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

# make `import config` work no matter where the script is launched from
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C


# ──────────────────────────────────────────────────────────────────────────────
#  Per-machine processing
# ──────────────────────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Invalidate sensor-error codes and out-of-range values (-> NaN)."""
    df = df.copy()
    # disk_io_percent documented error codes
    df.loc[df["disk_io_percent"].isin(C.DISK_INVALID_CODES), "disk_io_percent"] = np.nan
    # every metric must live in [0, 100]
    for col in C.FEATURE_COLS:
        df.loc[(df[col] < 0) | (df[col] > 100), col] = np.nan
    return df


def resample_to_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Average the raw ~10 s samples into a fixed RESAMPLE_S grid.

    Returns a DataFrame indexed by bin-start-second, one row per grid step,
    with NaN where the machine produced no samples in that bin.
    """
    df = df.copy()
    df["bin"] = (df["time_stamp"] // C.RESAMPLE_S) * C.RESAMPLE_S
    grid = df.groupby("bin")[C.FEATURE_COLS].mean()
    # reindex onto the complete, regular grid so gaps become explicit NaN rows
    full = np.arange(grid.index.min(),
                     grid.index.max() + C.RESAMPLE_S,
                     C.RESAMPLE_S)
    return grid.reindex(full)


def _nan_run_lengths(isna: np.ndarray) -> np.ndarray:
    """For a boolean mask, return an array where every True position carries the
    length of the consecutive True-run it belongs to (False positions -> 0)."""
    n = len(isna)
    runlen = np.zeros(n, dtype=np.int32)
    if not isna.any():
        return runlen
    padded = np.concatenate([[0], isna.astype(np.int8), [0]])
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    for s, e in zip(starts, ends):
        runlen[s:e] = e - s
    return runlen


def interpolate_short_gaps(grid: pd.DataFrame) -> pd.DataFrame:
    """Linearly fill runs of STRICTLY FEWER than 5 consecutive missing minutes;
    longer gaps are left fully as NaN (so any window touching them is dropped).

    Done per column, precisely: pandas' `interpolate(limit=...)` would partially
    fill a 5+ run, which we explicitly do NOT want.
    """
    out = grid.copy()
    for col in grid.columns:
        s = grid[col]
        isna = s.isna().to_numpy()
        if not isna.any():
            continue
        filled = s.interpolate(method="linear", limit_area="inside").to_numpy(copy=True)
        long_gap = isna & (_nan_run_lengths(isna) > C.MAX_INTERP_GAP)
        filled[long_gap] = np.nan          # undo fills inside long gaps
        out[col] = filled
    return out


def split_ids(n: int) -> np.ndarray:
    """Per-row split id: 0=train, 1=val, 2=test (contiguous, chronological)."""
    train_end = int(n * C.TRAIN_FRAC)
    val_end   = int(n * (C.TRAIN_FRAC + C.VAL_FRAC))
    ids = np.full(n, 2, dtype=np.int8)
    ids[:train_end] = 0
    ids[train_end:val_end] = 1
    return ids


# ──────────────────────────────────────────────────────────────────────────────
#  Windowing
# ──────────────────────────────────────────────────────────────────────────────

def make_windows(feat_scaled: np.ndarray,
                 target_raw: np.ndarray,
                 row_valid: np.ndarray,
                 split_id: np.ndarray,
                 W: int,
                 H: int):
    """Build all valid (window, target) pairs for one machine and one horizon.

    A window with input rows [i .. i+W-1] and target row (i+W-1+H) is kept iff
      * the target row exists,
      * every input row and the target row are fully observed (row_valid),
      * the window start and the target fall in the SAME split.
    Because splits are contiguous in time, equal endpoints imply the whole span
    stays inside one split — so no window crosses a boundary or a long gap.

    Returns (X, y_reg, y_clf, win_split) — all aligned, win_split in {0,1,2}.
    """
    T = feat_scaled.shape[0]
    n_win = T - W + 1
    if n_win <= 0:
        empty = np.empty((0, W, feat_scaled.shape[1]), np.float32)
        return empty, np.empty(0, np.float32), np.empty(0, np.int8), np.empty(0, np.int8)

    # all windows ending at rows [W-1 .. T-1]; window i covers rows [i .. i+W-1]
    win = np.lib.stride_tricks.sliding_window_view(feat_scaled, W, axis=0)
    win = win.transpose(0, 2, 1)                                   # (n_win, W, F)
    win_ok = np.lib.stride_tricks.sliding_window_view(row_valid, W).all(axis=1)

    starts = np.arange(n_win)
    targets = starts + W - 1 + H
    in_range = targets < T

    idx      = starts[in_range]
    tgt      = targets[in_range]
    keep     = win_ok[idx] & row_valid[tgt] & (split_id[idx] == split_id[tgt])

    X        = win[idx][keep].astype(np.float32)
    y_reg    = target_raw[tgt][keep].astype(np.float32)
    y_clf    = (y_reg > C.CLASSIFICATION_THRESHOLD).astype(np.int8)
    win_spl  = split_id[idx][keep].astype(np.int8)
    return X, y_reg, y_clf, win_spl


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    C.WINDOWS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" PREPROCESS — Alibaba forecasting dataset")
    print("=" * 70)
    print(f"  input            : {C.INPUT_CSV}")
    print(f"  resample grid    : {C.RESAMPLE_S}s   window W = {C.WINDOW_SIZE}   "
          f"horizons = {C.HORIZONS}")
    print(f"  features ({len(C.FEATURE_COLS)})    : {C.FEATURE_COLS}")
    print(f"  target           : {C.TARGET_METRIC}   clf threshold = "
          f"{C.CLASSIFICATION_THRESHOLD}%")

    if not C.INPUT_CSV.exists():
        sys.exit(f"\n  ERROR: {C.INPUT_CSV} not found. Run the v2 selection first.")

    # ── Load (only the columns we need) ───────────────────────────────────────
    usecols = ["machine_id", "time_stamp", *C.FEATURE_COLS, "machine_profile"]
    print("\n  loading csv ...", flush=True)
    df = pd.read_csv(C.INPUT_CSV, usecols=usecols)
    print(f"  loaded {len(df):,} rows, {df['machine_id'].nunique()} machines")

    machine_ids = sorted(df["machine_id"].unique())
    profile_of  = (df.groupby("machine_id")["machine_profile"].first()
                     .reindex(machine_ids).tolist())

    # ── Pass 1: clean + resample + interpolate; accumulate global train scaler ─
    print("\n  pass 1: clean / resample / interpolate ...", flush=True)
    grids        = {}                       # machine_id -> processed grid (DataFrame)
    feat_min     = np.full(len(C.FEATURE_COLS),  np.inf)
    feat_max     = np.full(len(C.FEATURE_COLS), -np.inf)

    for mid, g in df.groupby("machine_id"):
        g = g.sort_values("time_stamp", kind="mergesort")
        grid = interpolate_short_gaps(resample_to_grid(clean(g)))
        grids[mid] = grid

        # global scaler is fitted ONLY on this machine's training rows
        n = len(grid)
        train_end = int(n * C.TRAIN_FRAC)
        train_block = grid.iloc[:train_end].to_numpy()
        if np.isfinite(train_block).any():
            feat_min = np.fmin(feat_min, np.nanmin(train_block, axis=0))
            feat_max = np.fmax(feat_max, np.nanmax(train_block, axis=0))

    # guard: if a feature was all-NaN everywhere in training, fall back to [0,100]
    feat_min = np.where(np.isfinite(feat_min), feat_min,   0.0)
    feat_max = np.where(np.isfinite(feat_max), feat_max, 100.0)
    span = np.where((feat_max - feat_min) > 0, feat_max - feat_min, 1.0)
    print("  global min-max scaler (fitted on training rows only):")
    for name, lo, hi in zip(C.FEATURE_COLS, feat_min, feat_max):
        print(f"    {name:<20} [{lo:7.3f}, {hi:7.3f}]")

    tgt_col = C.FEATURE_COLS.index(C.TARGET_METRIC)

    # ── Pass 2: scale + window, per horizon ───────────────────────────────────
    print("\n  pass 2: scaling + windowing ...", flush=True)
    # accumulators: per horizon -> per split -> list of arrays
    buckets = {H: {s: {"X": [], "yr": [], "yc": [], "m": []} for s in (0, 1, 2)}
               for H in C.HORIZONS}

    for m_idx, mid in enumerate(machine_ids):
        grid = grids[mid]
        raw  = grid.to_numpy(dtype=np.float64)
        feat_scaled = (raw - feat_min) / span                 # (T, F) scaled
        target_raw  = raw[:, tgt_col].copy()                  # (T,) raw CPU %
        row_valid   = np.isfinite(raw).all(axis=1)            # (T,) fully observed
        spl         = split_ids(len(grid))

        for H in C.HORIZONS:
            X, yr, yc, ws = make_windows(feat_scaled, target_raw, row_valid,
                                         spl, C.WINDOW_SIZE, H)
            for s in (0, 1, 2):
                sel = ws == s
                if sel.any():
                    buckets[H][s]["X"].append(X[sel])
                    buckets[H][s]["yr"].append(yr[sel])
                    buckets[H][s]["yc"].append(yc[sel])
                    buckets[H][s]["m"].append(np.full(sel.sum(), m_idx, np.int16))

    # ── Save one npz per horizon ─────────────────────────────────────────────
    split_name = {0: "train", 1: "val", 2: "test"}
    summary = {
        "input_csv": str(C.INPUT_CSV),
        "n_machines": len(machine_ids),
        "resample_s": C.RESAMPLE_S,
        "window_size": C.WINDOW_SIZE,
        "horizons": C.HORIZONS,
        "features": C.FEATURE_COLS,
        "target": C.TARGET_METRIC,
        "clf_threshold": C.CLASSIFICATION_THRESHOLD,
        "scaler_min": feat_min.tolist(),
        "scaler_max": feat_max.tolist(),
        "per_horizon": {},
    }

    print("\n  saving windows ...")
    for H in C.HORIZONS:
        out = {}
        h_summary = {}
        for s in (0, 1, 2):
            b = buckets[H][s]
            if b["X"]:
                X  = np.concatenate(b["X"])
                yr = np.concatenate(b["yr"])
                yc = np.concatenate(b["yc"])
                m  = np.concatenate(b["m"])
            else:
                X  = np.empty((0, C.WINDOW_SIZE, len(C.FEATURE_COLS)), np.float32)
                yr = np.empty(0, np.float32)
                yc = np.empty(0, np.int8)
                m  = np.empty(0, np.int16)
            nm = split_name[s]
            out[f"X_{nm}"]     = X
            out[f"y_reg_{nm}"] = yr
            out[f"y_clf_{nm}"] = yc
            out[f"m_{nm}"]     = m
            pos = int(yc.sum())
            h_summary[nm] = {
                "n_windows": int(len(X)),
                "clf_positive": pos,
                "clf_positive_pct": round(100 * pos / len(X), 2) if len(X) else 0.0,
            }
        out["feature_names"] = np.array(C.FEATURE_COLS)
        out["profiles"]      = np.array(profile_of)
        out["scaler_min"]    = feat_min
        out["scaler_max"]    = feat_max
        np.savez_compressed(C.WINDOWS_DIR / f"h{H}.npz", **out)
        summary["per_horizon"][H] = h_summary
        tr, va, te = (h_summary["train"]["n_windows"],
                      h_summary["val"]["n_windows"],
                      h_summary["test"]["n_windows"])
        print(f"    h{H:>2}.npz   train={tr:>7,}  val={va:>7,}  test={te:>7,}   "
              f"test +class={h_summary['test']['clf_positive_pct']:>5.2f}%")

    # scaler + summary
    np.savez(C.WINDOWS_DIR / "scaler.npz",
             feature_names=np.array(C.FEATURE_COLS),
             scaler_min=feat_min, scaler_max=feat_max)
    with open(C.WINDOWS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  done in {time.time() - t0:.1f}s")
    print(f"  outputs -> {C.WINDOWS_DIR}")
    print("=" * 70)
    print(" Paste this summary back so I can sanity-check the windowing:")
    print("=" * 70)
    for H in C.HORIZONS:
        hs = summary["per_horizon"][H]
        print(f"  H={H:>2}min | train {hs['train']['n_windows']:>7,} "
              f"| val {hs['val']['n_windows']:>7,} "
              f"| test {hs['test']['n_windows']:>7,} "
              f"| test pos-class {hs['test']['clf_positive_pct']:>5.2f}%")


if __name__ == "__main__":
    main()
