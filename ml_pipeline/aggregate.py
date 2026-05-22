"""
aggregate.py — collect results, build comparison tables and paper figures.
================================================================================
Reads whatever model results exist (baseline / xgboost / lstm — missing ones are
skipped gracefully) and produces:

  Tables  (artifacts/results/)
    comparison_regression.csv      model x horizon : MAE, RMSE, R2
    comparison_classification.csv  model x horizon : F1, F1-macro, P, R, AUC
    summary_for_paper.md           tidy digest to drop into the write-up

  Figures (artifacts/figures/, 300 dpi)
    fig_profile_distribution.png   the 100 selected machines by behavioural profile
    fig_mae_vs_horizon.png         regression error growth with horizon
    fig_rmse_vs_horizon.png
    fig_f1_vs_horizon.png          classification F1 vs horizon
    fig_xgb_importance.png         top gain-importance features (XGBoost)
    fig_example_forecast.png       predicted vs actual CPU trace, one bursty machine
    fig_confusion_matrix.png       best classifier (model, horizon) by F1

Run:
    python ml_pipeline/aggregate.py     (needs: pip install matplotlib pandas)
================================================================================
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from common import regression_metrics, classification_metrics

MODEL_FILES = {"baseline": "baseline", "xgboost": "xgb", "lstm": "lstm"}
MIN_PROFILE_WINDOWS = 50            # skip a profile cell with too few test windows
PRETTY      = {"baseline": "Baseline", "xgboost": "XGBoost", "lstm": "LSTM"}
COLORS      = {"baseline": "#7f7f7f", "xgboost": "#1f77b4", "lstm": "#d62728"}
DPI = 300


# ──────────────────────────────────────────────────────────────────────────────
#  Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_results() -> dict:
    res = {}
    for name, stem in MODEL_FILES.items():
        p = C.RESULTS_DIR / f"{stem}.json"
        if p.exists():
            with open(p) as f:
                res[name] = json.load(f)
            print(f"  loaded {p.name}")
        else:
            print(f"  (skipping {name} — {p.name} not found)")
    return res


def load_preds() -> dict:
    preds = {}
    for name, stem in MODEL_FILES.items():
        p = C.RESULTS_DIR / f"{stem}_pred.npz"
        if p.exists():
            preds[name] = dict(np.load(p, allow_pickle=True))
    return preds


# ──────────────────────────────────────────────────────────────────────────────
#  Tables
# ──────────────────────────────────────────────────────────────────────────────

def build_tables(res: dict):
    reg_rows, clf_rows = [], []
    for name, r in res.items():
        for H in C.HORIZONS:
            h = r["horizons"].get(str(H))
            if not h:
                continue
            rm, cm = h["regression"], h["classification"]
            reg_rows.append({"model": PRETTY[name], "horizon_min": H,
                             "MAE": rm["mae"], "RMSE": rm["rmse"], "R2": rm["r2"]})
            clf_rows.append({"model": PRETTY[name], "horizon_min": H,
                             "F1": cm["f1"], "F1_macro": cm["f1_macro"],
                             "precision": cm["precision"], "recall": cm["recall"],
                             "AUC_ROC": cm["auc_roc"]})
    return pd.DataFrame(reg_rows), pd.DataFrame(clf_rows)


def print_table(df, title, fmt):
    print(f"\n  {title}")
    print("  " + "-" * (len(title)))
    if df.empty:
        print("  (no data)")
        return
    with pd.option_context("display.float_format", fmt):
        print(df.to_string(index=False).replace("\n", "\n  "))


def build_per_profile_table(preds: dict) -> pd.DataFrame:
    """Per-(model, horizon, profile) metrics, recomputed on the subset of test
    windows belonging to machines of each behavioural profile.

    Answers: do bursty / erratic VMs forecast worse than stable ones? — which is
    a core discussion point for the paper.
    """
    if not preds:
        return pd.DataFrame()
    # machine-index -> profile string; identical across horizons, load once
    w = np.load(C.WINDOWS_DIR / f"h{C.HORIZONS[0]}.npz", allow_pickle=True)
    profiles = w["profiles"]

    rows = []
    for name, p in preds.items():
        for H in C.HORIZONS:
            if f"m_test_h{H}" not in p:
                continue
            prof = profiles[p[f"m_test_h{H}"]]
            yreg_t, yreg_p = p[f"y_reg_h{H}"], p[f"reg_h{H}"]
            yclf_t, proba  = p[f"y_clf_h{H}"], p[f"clf_proba_h{H}"]
            for profile in sorted(set(prof.tolist())):
                mask = prof == profile
                n = int(mask.sum())
                if n < MIN_PROFILE_WINDOWS:
                    continue
                rm = regression_metrics(yreg_t[mask], yreg_p[mask])
                cm = classification_metrics(yclf_t[mask],
                                            (proba[mask] >= 0.5).astype(int),
                                            proba[mask])
                rows.append({"model": PRETTY[name], "horizon_min": H,
                             "profile": profile, "n_windows": n,
                             "MAE": rm["mae"], "RMSE": rm["rmse"], "R2": rm["r2"],
                             "F1": cm["f1"], "precision": cm["precision"],
                             "recall": cm["recall"], "AUC_ROC": cm["auc_roc"]})
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
#  Figures
# ──────────────────────────────────────────────────────────────────────────────

def _save(fig, name):
    path = C.FIGURES_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {path.name}")


def fig_profile_distribution():
    if not C.SELECTION_CSV.exists():
        print(f"  (skipping profile figure — {C.SELECTION_CSV.name} missing)")
        return
    sel = pd.read_csv(C.SELECTION_CSV)
    vc = sel["profile"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(vc.index, vc.values, color="#1f77b4")
    for i, v in enumerate(vc.values):
        ax.text(i, v + 0.5, str(v), ha="center", fontsize=8)
    ax.set_ylabel("machines")
    ax.set_title(f"Behavioural profiles of the {len(sel)} selected machines "
                 f"(forecasting set)")
    plt.xticks(rotation=30, ha="right")
    _save(fig, "fig_profile_distribution.png")


def fig_metric_vs_horizon(reg_df, clf_df):
    specs = [
        ("MAE",  reg_df, "Test MAE (CPU percentage points)", "fig_mae_vs_horizon.png", False),
        ("RMSE", reg_df, "Test RMSE (CPU percentage points)", "fig_rmse_vs_horizon.png", False),
        ("F1",   clf_df, "Test F1 (positive class)",          "fig_f1_vs_horizon.png", True),
    ]
    for metric, df, ylabel, fname, higher_better in specs:
        if df.empty:
            continue
        fig, ax = plt.subplots(figsize=(5, 3.2))
        for name in df["model"].unique():
            sub = df[df["model"] == name].sort_values("horizon_min")
            key = [k for k, v in PRETTY.items() if v == name][0]
            ax.plot(sub["horizon_min"], sub[metric], "o-", label=name,
                    color=COLORS.get(key), lw=1.8, ms=5)
        ax.set_xlabel("forecast horizon (minutes)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(C.HORIZONS)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
        arrow = "higher is better" if higher_better else "lower is better"
        ax.set_title(f"{metric} vs horizon  ({arrow})", fontsize=10)
        _save(fig, fname)


def fig_xgb_importance(res):
    if "xgboost" not in res:
        print("  (skipping XGB importance figure — no xgboost results)")
        return
    H0 = C.HORIZONS[0]
    imp = res["xgboost"]["horizons"][str(H0)]["regression"]\
            .get("feature_importance_gain", {})
    if not imp:
        return
    top = sorted(imp.items(), key=lambda kv: kv[1])[-15:]
    names = [k for k, _ in top]
    vals  = [v for _, v in top]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.barh(names, vals, color="#1f77b4")
    ax.set_xlabel("gain importance")
    ax.set_title(f"XGBoost top-15 features — regression, H={H0} min")
    _save(fig, "fig_xgb_importance.png")


def fig_example_forecast(preds):
    if not preds:
        return
    model = "xgboost" if "xgboost" in preds else next(iter(preds))
    H = 15 if 15 in C.HORIZONS else C.HORIZONS[0]
    p = preds[model]
    if f"reg_h{H}" not in p:
        return
    yhat, ytrue, m = p[f"reg_h{H}"], p[f"y_reg_h{H}"], p[f"m_test_h{H}"]

    w = np.load(C.WINDOWS_DIR / f"h{H}.npz", allow_pickle=True)
    profiles = w["profiles"]
    # prefer a bursty machine with a decent test trace
    chosen, best_n = None, 0
    for idx in range(len(profiles)):
        n = int((m == idx).sum())
        if profiles[idx] == "bursty" and n > 200:
            chosen = idx
            break
        if n > best_n:
            chosen, best_n = idx, n
    sel = m == chosen
    yt, yp = ytrue[sel], yhat[sel]
    n = min(400, len(yt))

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(range(n), yt[:n], color="black", lw=1.0, label="actual")
    ax.plot(range(n), yp[:n], color="#d62728", lw=1.0, alpha=0.85,
            label=f"{PRETTY[model]} predicted")
    ax.set_xlabel("test window index (chronological)")
    ax.set_ylabel("CPU utilisation (%)")
    ax.set_title(f"Example forecast — machine #{chosen} "
                 f"({profiles[chosen]}), H={H} min")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    _save(fig, "fig_example_forecast.png")


def fig_confusion_matrix(res):
    best = None  # (f1, model, H, cm)
    for name, r in res.items():
        for H in C.HORIZONS:
            h = r["horizons"].get(str(H))
            if not h:
                continue
            cm = h["classification"]
            if best is None or cm["f1"] > best[0]:
                best = (cm["f1"], name, H, np.array(cm["confusion_matrix"]))
    if best is None:
        return
    f1, name, H, cm = best
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred 0", "pred 1"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["true 0", "true 1"])
    ax.set_title(f"Best classifier: {PRETTY[name]}, H={H} min\nF1={f1:.3f}")
    _save(fig, "fig_confusion_matrix.png")


def fig_metric_by_profile(prof_df):
    """Grouped bars: per-VM-profile MAE and F1, one bar per model, at one horizon."""
    if prof_df.empty:
        return
    H = 15 if 15 in C.HORIZONS else C.HORIZONS[0]
    for metric, fname, title in [
        ("MAE", "fig_mae_by_profile.png", "Test MAE by VM profile"),
        ("F1",  "fig_f1_by_profile.png",  "Test F1 by VM profile"),
    ]:
        sub = prof_df[prof_df["horizon_min"] == H]
        if sub.empty:
            continue
        profs  = sorted(sub["profile"].unique())
        models = list(sub["model"].unique())
        x = np.arange(len(profs))
        width = 0.8 / max(len(models), 1)
        fig, ax = plt.subplots(figsize=(7, 3.4))
        for i, mdl in enumerate(models):
            vals = [sub[(sub.model == mdl) & (sub.profile == pr)][metric].mean()
                    for pr in profs]
            key = [k for k, v in PRETTY.items() if v == mdl][0]
            ax.bar(x + i * width, vals, width, label=mdl, color=COLORS.get(key))
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(profs, rotation=30, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(f"{title}  (H={H} min)")
        ax.legend(frameon=False)
        ax.grid(alpha=0.3, axis="y")
        _save(fig, fname)


# ──────────────────────────────────────────────────────────────────────────────
#  Markdown digest
# ──────────────────────────────────────────────────────────────────────────────

def _as_md(df):
    """Markdown table if `tabulate` is installed, else a plain-text fallback."""
    if df.empty:
        return "_no data_"
    try:
        return df.to_markdown(index=False, floatfmt=".3f")
    except ImportError:
        return "```\n" + df.round(3).to_string(index=False) + "\n```"


def write_markdown(reg_df, clf_df, prof_df, res):
    lines = ["# Results digest\n"]
    lines.append("## Regression (test set)\n")
    lines.append(_as_md(reg_df))
    lines.append("\n\n## Classification (test set, tau = "
                 f"{C.CLASSIFICATION_THRESHOLD:.0f}% CPU)\n")
    lines.append(_as_md(clf_df))
    if not prof_df.empty:
        H0 = 15 if 15 in C.HORIZONS else C.HORIZONS[0]
        lines.append(f"\n\n## Per-profile breakdown (test set, H={H0} min)\n")
        lines.append(_as_md(prof_df[prof_df["horizon_min"] == H0]
                            .drop(columns=["horizon_min"])))
    lines.append("\n\n## Run metadata\n")
    for name, r in res.items():
        lines.append(f"- **{PRETTY[name]}**: runtime "
                     f"{r.get('runtime_s', '?')}s")
        if name == "lstm":
            lines.append(f"  (device: {r.get('device', '?')})")
    path = C.RESULTS_DIR / "summary_for_paper.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  digest -> {path.name}")


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print(" AGGREGATE — tables + figures")
    print("=" * 70)

    res = load_results()
    if not res:
        sys.exit("  no result files found — run the train_*.py scripts first.")
    preds = load_preds()

    reg_df, clf_df = build_tables(res)
    reg_df.to_csv(C.RESULTS_DIR / "comparison_regression.csv", index=False)
    clf_df.to_csv(C.RESULTS_DIR / "comparison_classification.csv", index=False)

    prof_df = build_per_profile_table(preds)
    prof_df.to_csv(C.RESULTS_DIR / "comparison_by_profile.csv", index=False)

    print_table(reg_df, "REGRESSION (test set)", "{:.3f}".format)
    print_table(clf_df, "CLASSIFICATION (test set)", "{:.3f}".format)
    if not prof_df.empty:
        H0 = 15 if 15 in C.HORIZONS else C.HORIZONS[0]
        print_table(prof_df[prof_df["horizon_min"] == H0]
                    .drop(columns=["horizon_min"]),
                    f"PER-PROFILE (test set, H={H0} min)", "{:.3f}".format)

    print()
    fig_profile_distribution()
    fig_metric_vs_horizon(reg_df, clf_df)
    fig_metric_by_profile(prof_df)
    fig_xgb_importance(res)
    fig_example_forecast(preds)
    fig_confusion_matrix(res)
    write_markdown(reg_df, clf_df, prof_df, res)

    print("\n  all outputs in:")
    print(f"    {C.RESULTS_DIR}")
    print(f"    {C.FIGURES_DIR}")
    print("=" * 70)
    print(" Paste comparison_regression.csv + comparison_classification.csv")
    print(" back to me and I'll write the IEEE paper around them.")
    print("=" * 70)


if __name__ == "__main__":
    main()
