# IAA — Predictive Monitoring of Cloud Cluster Resources

Time-series forecasting pipeline on the **Alibaba Cluster Trace v2018**. Given a
machine's recent multivariate telemetry (CPU, memory, network, disk), predict
its CPU utilisation at horizons of **5, 15, 30 and 60 minutes** — as both a
regression (future value) and a classification (will CPU exceed 80 %?) task.

Three models are compared: a **Linear/Logistic baseline**, **XGBoost** (with
delta-target regression), and an **LSTM**.

## Pipeline

```text
machine_usage.csv  (raw, ~4000 machines, 10 s sampling)
        │
        ▼  selecionar_maquinas_v2.py   stratified selection of 100 machines
        │                              across 7 behavioural profiles
        ▼  preprocess.py               clean → resample to 1 min → sliding
        │                              windows (W=12) → 60/20/20 split
        ▼  train_baseline.py  train_xgb.py  train_lstm.py
        │                              3 models × 2 tasks × 4 horizons
        ▼  aggregate.py                comparison tables + figures
        │                              (overall + per-VM-profile)
        ▼  tune_threshold.py           post-hoc F1-optimal decision
                                       threshold per (model, horizon)
```

## Files

| File | Role |
|---|---|
| `selecionar_maquinas_v2.py` | Selects 100 machines, stratified by behavioural profile |
| `config.py` | Central config — paths, window size, horizons, splits |
| `common.py` | Shared feature extraction + metrics (identical scoring for all models) |
| `preprocess.py` | Cleaning, resampling, windowing, chronological split → `.npz` |
| `train_baseline.py` | Linear / Logistic Regression baseline |
| `train_xgb.py` | XGBoost, grid search, GPU-enabled, residual (delta) target |
| `train_lstm.py` | LSTM, validation-based hyperparameter search, GPU-enabled |
| `aggregate.py` | Comparison tables + figures + per-profile breakdown |
| `tune_threshold.py` | Post-hoc threshold sweep (F1-optimal cutoff per model/horizon) |
| `perfil_todas_maquinas_v2.csv` | Profile of every machine in the trace |
| `selecao_forecasting.csv` | The 100 selected machines + their profiles |

## Quick start

```bash
pip install numpy pandas scikit-learn xgboost matplotlib tabulate torch

python selecionar_maquinas_v2.py      # → dataset_forecasting.csv
python preprocess.py                  # → artifacts/W{W}/windows/*.npz
python train_baseline.py
python train_xgb.py
python train_lstm.py
python aggregate.py                   # → comparison CSVs + figures + summary.md
python tune_threshold.py              # → threshold_tuning.{json,md}
```

All outputs land under `artifacts/W{WINDOW_SIZE}/` (e.g. `artifacts/W12/`), so
different context lengths accumulate side-by-side instead of overwriting. To
run a W=60 ablation: set `WINDOW_SIZE = 60` in `config.py`, rerun
`preprocess.py`, then the three train scripts, then `aggregate.py`.

## Notes on the modelling choices

- **Resampling.** Raw samples are at ~10 s; we average onto a 1-min grid for
  cleaner horizons (1 step = 1 minute, so horizon in minutes = horizon in
  steps) and less sensor noise.
- **Window length.** Default W=12 (12 min of context). Increase for seasonal /
  bursty behaviours; trade-off is fewer usable windows.
- **Dropped features.** `mkpi` and `mem_gps` are ~82 % missing in the
  forecasting subset and are not used.
- **Chronological split.** Per machine, first 60 % train / next 20 % val /
  last 20 % test. No leakage across machines or across time within a machine.
- **XGBoost regression target.** Trees are piecewise-constant and cannot
  reproduce the near-identity mapping `y_{t+H} ≈ y_t` that dominates CPU
  autocorrelation. We train on the residual `y_{t+H} − y_t` and add the last
  observed value back at inference. A `persistence_mae`/`persistence_rmse`
  baseline is reported alongside each XGBoost row for sanity.
- **Threshold tuning.** `tune_threshold.py` sweeps the decision threshold on
  the test predictions and reports the F1-optimal cutoff. This is a
  diagnostic upper bound — for deployment, tune on the validation set.

## Objectives

- Predictive cloud monitoring and resource-usage forecasting
- Threshold-violation (spike) prediction
- Comparison of classical ML and deep-learning models across forecast horizons,
  including per-VM-profile behaviour and decision-threshold sensitivity
