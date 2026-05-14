"""
config.py — single source of truth for the whole ML pipeline.

Imported by preprocess.py, train_baseline.py, train_xgb.py, train_lstm.py
and aggregate.py. Edit values here; every script picks them up.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
# config.py lives in  3_Alibaba_v2018/ml_pipeline/  → project root is its parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "alibaba_processado"      # v2 outputs live here
ARTIFACTS    = PROJECT_ROOT / "artifacts"
WINDOWS_DIR  = ARTIFACTS / "windows"                    # preprocess.py output
RESULTS_DIR  = ARTIFACTS / "results"                    # train_*.py metric JSONs
MODELS_DIR   = ARTIFACTS / "models"                     # saved model checkpoints
FIGURES_DIR  = ARTIFACTS / "figures"                    # plots for the paper

# Which selection to analyse (the paper focuses on forecasting)
PRIMARY_DATASET = "forecasting"                         # general | forecasting | anomaly
INPUT_CSV     = DATA_DIR / f"dataset_{PRIMARY_DATASET}.csv"
SELECTION_CSV = DATA_DIR / f"selecao_{PRIMARY_DATASET}.csv"

# ── Resolution ─────────────────────────────────────────────────────────────
# The raw machine_usage table is sampled every ~10 s. We resample each
# machine's series onto a fixed 1-minute grid by averaging (smooths sensor
# noise, makes horizons clean step counts, cuts series to ~11.5k pts/machine).
RESAMPLE_S = 60                                         # 1-minute grid

# ── Features ───────────────────────────────────────────────────────────────
# mkpi and mem_gps are dropped: ~82 % missing in dataset_forecasting.csv.
FEATURE_COLS = [
    "cpu_util_percent",
    "mem_util_percent",
    "net_in",
    "net_out",
    "disk_io_percent",
]
TARGET_METRIC = "cpu_util_percent"                      # the variable we forecast

# ── Windowing / horizons ───────────────────────────────────────────────────
# On the 1-min grid, 1 step = 1 minute, so horizon (minutes) == horizon (steps).
WINDOW_SIZE = 12                                        # 12 min of context
HORIZONS    = [5, 15, 30, 60]                           # minutes / steps ahead

# Binary classification: does the target metric exceed this threshold at t+H?
CLASSIFICATION_THRESHOLD = 80.0                         # % CPU utilisation

# ── Chronological split (per machine, no leakage across machines) ──────────
TRAIN_FRAC = 0.60
VAL_FRAC   = 0.20
TEST_FRAC  = 0.20

# ── Cleaning ───────────────────────────────────────────────────────────────
# disk_io_percent sensor error codes (documented in the Alibaba schema)
DISK_INVALID_CODES = (-1, 101)
# Linearly interpolate runs of STRICTLY FEWER than 5 consecutive missing
# minutes; longer gaps are left as NaN and any window touching them is dropped.
MAX_INTERP_GAP = 4

# ── Reproducibility ────────────────────────────────────────────────────────
RANDOM_SEED = 42
