# IAA — Predictive Monitoring of Cloud Cluster Resources

Time-series forecasting pipeline on the **Alibaba Cluster Trace v2018**. Given a
machine's recent multivariate telemetry (CPU, memory, network, disk), predict
its CPU utilisation at horizons of **5, 15, 30 and 60 minutes** — as both a
regression (future value) and a classification (will CPU exceed 80 %?) task.

Three models are compared: a **Linear/Logistic baseline**, **XGBoost**, and an
**LSTM**.

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
```

## Files

| File | Role |
|---|---|
| `selecionar_maquinas_v2.py` | Selects 100 machines, stratified by behavioural profile |
| `config.py` | Central config — paths, window size, horizons, splits |
| `common.py` | Shared feature extraction + metrics (identical scoring for all models) |
| `preprocess.py` | Cleaning, resampling, windowing, chronological split → `.npz` |
| `train_baseline.py` | Linear / Logistic Regression baseline |
| `train_xgb.py` | XGBoost, grid search, GPU-enabled |
| `train_lstm.py` | LSTM, validation-based hyperparameter search, GPU-enabled |
| `aggregate.py` | Builds comparison tables and paper figures |
| `perfil_todas_maquinas_v2.csv` | Profile of every machine in the trace |
| `selecao_forecasting.csv` | The 100 selected machines + their profiles |

## Quick start

```bash
pip install numpy pandas scikit-learn xgboost matplotlib tabulate torch

python selecionar_maquinas_v2.py      # → dataset_forecasting.csv
python preprocess.py                  # → artifacts/windows/*.npz
python train_baseline.py
python train_xgb.py
python train_lstm.py
python aggregate.py                   # → artifacts/results/ , artifacts/figures/
```

## Objectives

- Predictive cloud monitoring and resource-usage forecasting
- Threshold-violation (spike) prediction
- Comparison of classical ML and deep-learning models across forecast horizons
