# IAA Predictive Monitoring Cloud

Machine learning pipeline for predictive monitoring using the Alibaba Cluster Trace dataset.

## Files

### `aggregate.py`
Aggregates and summarizes the results produced by the ML models.

### `common.py`
Shared utility and helper functions used across the pipeline.

### `config.py`
Central configuration file containing paths and experiment parameters.

### `perfil_todas_maquinas_v2.csv`
Machine profile dataset used during machine selection and analysis.

### `preprocess.py`
Preprocesses and prepares the dataset for training.

### `selecao_forecasting.csv`
List of machines selected for forecasting experiments.

### `selecionar_maquinas_v2.py`
Selects suitable machines from the dataset for the experiments.

### `train_baseline.py`
Trains baseline Linear and Logistic Regression models.

### `train_lstm.py`
Trains LSTM models for forecasting and classification tasks.

### `train_xgb.py`
Trains XGBoost models for forecasting and classification tasks.

---

## Pipeline

```text
Raw Data
   ↓
selecionar_maquinas_v2.py
   ↓
aggregate.py
   ↓
preprocess.py
   ↓
train_baseline.py / train_xgb.py / train_lstm.py
```

## Objectives

- Predictive cloud monitoring
- Resource usage forecasting
- Event/anomaly prediction
- Comparison of ML and DL models
