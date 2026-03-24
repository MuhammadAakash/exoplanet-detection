"""
compare_models.py
=================
Stage 6 — Final Model Comparison for Exoplanet Candidate Vetting.

Loads saved metrics for all five models across Stages 3–5, generates a
comprehensive set of dissertation-ready comparison figures, and saves a
clean summary table.

Models compared
---------------
    Stage 3 — Random Forest
    Stage 3 — Support Vector Machine
    Stage 3 — Logistic Regression
    Stage 4 — Baseline CNN  (single-branch, kernel=3)
    Stage 5 — Genesis CNN   (dual-branch, kernel=3 + kernel=7)

Figures produced
----------------
    comp_01_overall_metrics.png     — Accuracy / F1 / ROC-AUC / κ bar chart
    comp_02_perclass_f1.png         — Per-class F1 grouped bar chart
    comp_03_heatmap.png             — Metrics heatmap across all models
    comp_04_radar.png               — Radar chart (all metrics per model)
    comp_05_cnn_deep_dive.png       — Baseline vs Genesis CNN head-to-head

Usage
-----
    python -m src.evaluation.compare_models

    # Or called directly:
    from src.evaluation.compare_models import run_comparison
    run_comparison()
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from math import pi

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import METRICS_DIR, FIGURES_DIR, CLASS_NAMES
from src.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Consistent visual identity
# ---------------------------------------------------------------------------
# One colour per model — used throughout every figure
_MODEL_COLORS = {
    "Random Forest"      : "#2ecc71",   # green
    "Logistic Regression": "#e67e22",   # orange
    "SVM"                : "#e74c3c",   # red
    "Baseline CNN"       : "#3498db",   # blue
    "Genesis CNN"        : "#9b59b6",   # purple
}

# Display order — classical ML first, then deep learning
_MODEL_ORDER = [
    "Random Forest",
    "Logistic Regression",
    "SVM",
    "Baseline CNN",
    "Genesis CNN",
]

# Metrics for the overall bar chart
_OVERVIEW_METRICS = {
    "accuracy"      : "Accuracy",
    "f1_macro"      : "F1 Macro",
    "roc_auc_macro" : "ROC-AUC",
    "cohen_kappa"   : "Cohen's κ",
}


# =============================================================================
# 1. DATA LOADING
# =============================================================================

def _load_all_metrics() -> pd.DataFrame:
    """
    Load and consolidate metrics for all five models.

    Reads from results/metrics/all_models_metrics.csv (written by
    evaluate_model() after each stage) and validates that all five
    expected models are present.

    Returns
    -------
    pd.DataFrame  with one row per model, ordered by _MODEL_ORDER.
    """
    master_csv = METRICS_DIR / "all_models_metrics.csv"
    if not master_csv.exists():
        raise FileNotFoundError(
            f"Master metrics CSV not found: {master_csv}\n"
            "Run Stages 3–5 before running Stage 6."
        )

    df = pd.read_csv(master_csv)
    log.info(f"Loaded metrics for {len(df)} models from {master_csv.name}")

    # Keep only the models we expect and reorder them
    df = df[df["model"].isin(_MODEL_ORDER)].copy()
    df["model"] = pd.Categorical(df["model"], categories=_MODEL_ORDER, ordered=True)
    df = df.sort_values("model").reset_index(drop=True)

    missing = set(_MODEL_ORDER) - set(df["model"])
    if missing:
        log.warning(f"Missing models in metrics CSV: {missing}")

    return df


# =============================================================================
# 2. FIGURES
# =============================================================================

def _plot_overall_metrics(df: pd.DataFrame) -> None:
    """
    Four-panel bar chart: Accuracy, F1 Macro, ROC-AUC, Cohen's κ.

    Each panel shows one metric for all five models, sorted by the
    display order.  Models are colour-coded consistently.
    """
    fig, axes = plt.subplots(1, 4, figsize=(17, 5))
    fig.suptitle(
        "Stage 6 — All-Model Comparison: Overall Metrics",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for ax, (col, label) in zip(axes, _OVERVIEW_METRICS.items()):
        if col not in df.columns:
            ax.set_visible(False)
            continue

        values = df[col].values
        models = df["model"].values
        colors = [_MODEL_COLORS.get(m, "#aaaaaa") for m in models]

        bars = ax.bar(range(len(models)), values, color=colors,
                      edgecolor="white", linewidth=0.5, width=0.6)

        # Value labels on each bar
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v:.3f}",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold",
            )

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_ylim(0, min(1.12, max(values) * 1.15))
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = FIGURES_DIR / "comp_01_overall_metrics.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved → {path.name}")


def _plot_perclass_f1(df: pd.DataFrame) -> None:
    """
    Grouped bar chart: per-class F1 scores for all five models.

    Three groups (CONFIRMED / FALSE POSITIVE / CANDIDATE), five bars
    per group (one per model).  This directly shows which class each
    model struggles with.
    """
    f1_cols = {
        "f1_confirmed"      : "CONFIRMED",
        "f1_false_positive" : "FALSE POSITIVE",
        "f1_candidate"      : "CANDIDATE",
    }

    models  = df["model"].values
    n_models = len(models)
    n_classes = len(f1_cols)
    x = np.arange(n_classes)
    bar_w = 0.15
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * bar_w

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, model in enumerate(models):
        row = df[df["model"] == model].iloc[0]
        f1_vals = [row.get(col, 0) for col in f1_cols]
        color = _MODEL_COLORS.get(model, "#aaaaaa")
        bars = ax.bar(x + offsets[i], f1_vals, bar_w, label=model,
                      color=color, alpha=0.88, edgecolor="white", linewidth=0.4)
        for bar, v in zip(bars, f1_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{v:.2f}",
                ha="center", va="bottom", fontsize=6.5, rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(list(f1_cols.values()), fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_title(
        "Stage 6 — Per-Class F1 Score: All Models",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=8, loc="upper left", framealpha=0.8)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    # Shade CANDIDATE column to highlight it as the hard class
    ax.axvspan(1.5, 2.5, alpha=0.06, color="#e74c3c", zorder=0)
    ax.text(2, 1.09, "hardest class", ha="center", fontsize=8,
            color="#e74c3c", style="italic")

    plt.tight_layout()
    path = FIGURES_DIR / "comp_02_perclass_f1.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved → {path.name}")


def _plot_heatmap(df: pd.DataFrame) -> None:
    """
    Heatmap of all key metrics across all models.

    Rows = models, columns = metrics.  Cell values are shown and
    cells are coloured by value (higher = darker green).
    """
    metric_cols = list(_OVERVIEW_METRICS.keys()) + [
        "f1_confirmed", "f1_false_positive", "f1_candidate",
    ]
    metric_labels = list(_OVERVIEW_METRICS.values()) + [
        "F1 Confirmed", "F1 False Pos.", "F1 Candidate",
    ]

    # Build heatmap matrix
    present_cols = [c for c in metric_cols if c in df.columns]
    present_labels = [metric_labels[metric_cols.index(c)] for c in present_cols]

    heat_df = df.set_index("model")[present_cols].rename(
        columns=dict(zip(present_cols, present_labels))
    )
    heat_df = heat_df.loc[[m for m in _MODEL_ORDER if m in heat_df.index]]

    fig, ax = plt.subplots(figsize=(11, 4))
    sns.heatmap(
        heat_df.astype(float),
        annot=True,
        fmt=".3f",
        cmap="YlGn",
        linewidths=0.5,
        linecolor="#cccccc",
        ax=ax,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 9},
        vmin=0.3,
        vmax=1.0,
    )
    ax.set_title(
        "Stage 6 — Metrics Heatmap: All Models × All Metrics",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9, rotation=0)
    plt.tight_layout()

    path = FIGURES_DIR / "comp_03_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved → {path.name}")


def _plot_radar(df: pd.DataFrame) -> None:
    """
    Radar (spider) chart — one polygon per model across five metrics.

    Normalises all metrics to [0, 1] for a fair visual comparison.
    """
    radar_cols  = ["accuracy", "f1_macro", "roc_auc_macro",
                   "cohen_kappa", "f1_candidate"]
    radar_labels = ["Accuracy", "F1 Macro", "ROC-AUC", "Cohen's κ", "F1 Candidate"]
    n = len(radar_cols)

    # Angles for each axis (evenly spaced around the circle)
    angles = [i * 2 * pi / n for i in range(n)]
    angles += angles[:1]   # close the polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#f8f9fa")

    for _, row in df.iterrows():
        model = row["model"]
        vals = [float(row.get(c, 0)) for c in radar_cols]
        vals += vals[:1]
        color = _MODEL_COLORS.get(model, "#aaaaaa")
        ax.plot(angles, vals, color=color, lw=2, label=model)
        ax.fill(angles, vals, color=color, alpha=0.08)

    # Axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7,
                        color="grey")
    ax.grid(color="grey", alpha=0.4)

    ax.set_title(
        "Stage 6 — Radar Chart: Multi-Metric Comparison",
        fontsize=11, fontweight="bold", pad=20,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)

    plt.tight_layout()
    path = FIGURES_DIR / "comp_04_radar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved → {path.name}")


def _plot_cnn_deep_dive(df: pd.DataFrame) -> None:
    """
    Head-to-head comparison between Baseline CNN and Genesis CNN.

    Two panels:
      Left  — four overall metrics (Accuracy, F1, AUC, κ)
      Right — per-class F1

    Designed to support the dissertation discussion of whether the
    dual-branch architecture improved on the single-branch baseline.
    """
    cnn_models = ["Baseline CNN", "Genesis CNN"]
    cnn_df = df[df["model"].isin(cnn_models)].copy()
    if cnn_df.empty:
        log.warning("CNN metrics not found — skipping CNN deep-dive plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Stage 6 — CNN Deep Dive: Baseline vs Genesis",
        fontsize=13, fontweight="bold",
    )

    # --- Left: overall metrics ---
    overview_cols  = list(_OVERVIEW_METRICS.keys())
    overview_labels = list(_OVERVIEW_METRICS.values())
    x = np.arange(len(overview_cols))
    bar_w = 0.35

    for i, model in enumerate(cnn_models):
        row = cnn_df[cnn_df["model"] == model]
        if row.empty:
            continue
        row = row.iloc[0]
        vals = [float(row.get(c, 0)) for c in overview_cols]
        color = _MODEL_COLORS.get(model, "#aaaaaa")
        offset = (i - 0.5) * bar_w
        bars = ax1.bar(x + offset, vals, bar_w, label=model,
                       color=color, alpha=0.9, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.005,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(overview_labels, fontsize=10)
    ax1.set_ylabel("Score"); ax1.set_ylim(0, 1.12)
    ax1.set_title("Overall Metrics", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9); ax1.grid(axis="y", alpha=0.3)
    ax1.spines[["top", "right"]].set_visible(False)

    # --- Right: per-class F1 ---
    f1_cols   = ["f1_confirmed", "f1_false_positive", "f1_candidate"]
    f1_labels = ["CONFIRMED", "FALSE POS.", "CANDIDATE"]
    x2 = np.arange(len(f1_cols))

    for i, model in enumerate(cnn_models):
        row = cnn_df[cnn_df["model"] == model]
        if row.empty:
            continue
        row = row.iloc[0]
        vals = [float(row.get(c, 0)) for c in f1_cols]
        color = _MODEL_COLORS.get(model, "#aaaaaa")
        offset = (i - 0.5) * bar_w
        bars = ax2.bar(x2 + offset, vals, bar_w, label=model,
                       color=color, alpha=0.9, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.005,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(f1_labels, fontsize=10)
    ax2.set_ylabel("F1 Score"); ax2.set_ylim(0, 1.12)
    ax2.set_title("Per-Class F1", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(axis="y", alpha=0.3)
    ax2.spines[["top", "right"]].set_visible(False)

    # Highlight the CANDIDATE column
    ax2.axvspan(1.5, 2.5, alpha=0.07, color="#e74c3c", zorder=0)

    plt.tight_layout()
    path = FIGURES_DIR / "comp_05_cnn_deep_dive.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved → {path.name}")


# =============================================================================
# 3. SUMMARY TABLE
# =============================================================================

def _build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a clean, dissertation-ready summary table.

    Columns: Model, Accuracy, F1 Macro, ROC-AUC, Cohen's κ,
             F1 Confirmed, F1 False Pos., F1 Candidate
    Rows are sorted by Accuracy descending, with rank column added.
    """
    col_map = {
        "model"             : "Model",
        "accuracy"          : "Accuracy",
        "f1_macro"          : "F1 Macro",
        "roc_auc_macro"     : "ROC-AUC",
        "cohen_kappa"       : "Cohen's κ",
        "f1_confirmed"      : "F1 Confirmed",
        "f1_false_positive" : "F1 False Pos.",
        "f1_candidate"      : "F1 Candidate",
    }

    present = {k: v for k, v in col_map.items() if k in df.columns}
    summary = df[list(present.keys())].rename(columns=present).copy()
    summary = summary.sort_values("Accuracy", ascending=False).reset_index(drop=True)
    summary.insert(0, "Rank", range(1, len(summary) + 1))

    # Round all numeric columns to 4dp
    for col in summary.columns:
        if col not in ("Rank", "Model"):
            summary[col] = summary[col].round(4)

    return summary


# =============================================================================
# 4. MAIN ENTRY POINT
# =============================================================================

def run_comparison() -> pd.DataFrame:
    """
    Full Stage 6 pipeline: load → compare → visualise → save.

    Steps
    -----
    1.  Load all model metrics from the master CSV
    2.  Generate five comparison figures
    3.  Build and save the dissertation summary table
    4.  Print a formatted leaderboard to the log

    Returns
    -------
    pd.DataFrame
        The dissertation summary table (one row per model, sorted by
        Accuracy descending).
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 — Load
    # ------------------------------------------------------------------
    log.info("=" * 55)
    log.info("STAGE 6 — FINAL MODEL COMPARISON")
    log.info("=" * 55)
    df = _load_all_metrics()

    # ------------------------------------------------------------------
    # Step 2 — Figures
    # ------------------------------------------------------------------
    log.info("\nGenerating comparison figures …")
    _plot_overall_metrics(df)
    _plot_perclass_f1(df)
    _plot_heatmap(df)
    _plot_radar(df)
    _plot_cnn_deep_dive(df)

    # ------------------------------------------------------------------
    # Step 3 — Summary table
    # ------------------------------------------------------------------
    summary = _build_summary_table(df)
    summary_path = METRICS_DIR / "stage6_final_comparison.csv"
    summary.to_csv(summary_path, index=False)
    log.info(f"\nSummary table saved → {summary_path.name}")

    # ------------------------------------------------------------------
    # Step 4 — Print leaderboard
    # ------------------------------------------------------------------
    log.info("\n" + "=" * 75)
    log.info("FINAL LEADERBOARD")
    log.info("=" * 75)
    log.info(
        f"{'Rank':<5} {'Model':<22} {'Accuracy':>9} {'F1 Macro':>9} "
        f"{'ROC-AUC':>9} {'Cohen κ':>9} {'F1 Cand.':>9}"
    )
    log.info("-" * 75)
    for _, row in summary.iterrows():
        log.info(
            f"{int(row['Rank']):<5} {row['Model']:<22} "
            f"{row['Accuracy']:>9.4f} {row['F1 Macro']:>9.4f} "
            f"{row.get('ROC-AUC', 0):>9.4f} {row.get('Cohen\'s κ', 0):>9.4f} "
            f"{row.get('F1 Candidate', 0):>9.4f}"
        )
    log.info("=" * 75)

    # Key findings for dissertation
    best_model      = summary.iloc[0]["Model"]
    best_acc        = summary.iloc[0]["Accuracy"]
    genesis_row     = summary[summary["Model"] == "Genesis CNN"]
    baseline_row    = summary[summary["Model"] == "Baseline CNN"]

    if not genesis_row.empty and not baseline_row.empty:
        genesis_acc  = genesis_row.iloc[0]["Accuracy"]
        baseline_acc = baseline_row.iloc[0]["Accuracy"]
        delta_gb     = genesis_acc - baseline_acc
        rf_row       = summary[summary["Model"] == "Random Forest"]
        if not rf_row.empty:
            rf_acc   = rf_row.iloc[0]["Accuracy"]
            delta_gr = genesis_acc - rf_acc
            log.info(
                f"\nKey finding: Genesis CNN ({genesis_acc:.4f}) "
                f"{'beats' if delta_gb > 0 else 'trails'} Baseline CNN "
                f"({baseline_acc:.4f}) by {abs(delta_gb)*100:.2f} pp"
            )
            log.info(
                f"Key finding: Genesis CNN ({genesis_acc:.4f}) "
                f"{'beats' if delta_gr > 0 else 'trails'} Random Forest "
                f"({rf_acc:.4f}) by {abs(delta_gr)*100:.2f} pp"
            )

    log.info(f"\nBest overall model: {best_model}  (Accuracy = {best_acc:.4f})")
    log.info("\nAll Stage 6 outputs saved to results/figures/ and results/metrics/")

    return summary


# =============================================================================
# Script entry point
# =============================================================================
if __name__ == "__main__":
    run_comparison()
