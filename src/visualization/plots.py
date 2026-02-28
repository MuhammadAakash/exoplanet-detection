"""
plots.py
========
Centralised plotting functions for the Exoplanet Candidate Vetting project.

All figures are saved to ``results/figures/`` with a consistent style so
they can be dropped directly into the dissertation document.

Functions
---------
plot_class_distribution      – Bar chart of label frequencies
plot_correlation_heatmap     – Feature correlation heatmap
plot_missing_values          – Horizontal bar chart of missing-value rates
plot_feature_distributions   – Grid of histograms per feature group
plot_confusion_matrix        – Annotated confusion matrix heat-map
plot_roc_curves              – One-vs-Rest ROC curves for all classes
plot_precision_recall_curves – PR curves for all classes
plot_training_history        – Loss and accuracy over epochs (CNN)
plot_feature_importance      – Horizontal bar chart (Random Forest / XGBoost)
plot_model_comparison        – Grouped bar chart comparing all models

Usage
-----
    from src.visualization.plots import plot_class_distribution
    plot_class_distribution(df, save=True)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)
from sklearn.preprocessing import label_binarize

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import (
    FIGURES_DIR, TARGET_COL, CLASS_NAMES,
    PLOT_STYLE, PLOT_DPI, PLOT_FIGSIZE, COLOR_PALETTE,
    MULTICLASS_LABEL_MAP,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Apply global style settings
# ---------------------------------------------------------------------------
try:
    plt.style.use(PLOT_STYLE)
except OSError:
    plt.style.use("seaborn-v0_8-whitegrid")

PALETTE = COLOR_PALETTE   # [green, red, blue]
CLASS_COLORS = {
    "CONFIRMED"    : PALETTE[0],
    "FALSE POSITIVE": PALETTE[1],
    "CANDIDATE"    : PALETTE[2],
}


def _save_or_show(fig: plt.Figure, filename: str, save: bool) -> None:
    """Helper: save figure to FIGURES_DIR or display it."""
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        out = FIGURES_DIR / filename
        fig.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
        log.info(f"Figure saved → {out}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Class distribution
# ---------------------------------------------------------------------------

def plot_class_distribution(
    df: pd.DataFrame,
    col: str = TARGET_COL,
    save: bool = True,
    filename: str = "01_class_distribution.png",
) -> None:
    """
    Bar chart showing the count and percentage of each KOI disposition class.

    Parameters
    ----------
    df       : pd.DataFrame  – DataFrame containing the target column.
    col      : str           – Name of the target column.
    save     : bool          – If True, save to figures dir; else display.
    filename : str           – Output filename.
    """
    counts = df[col].value_counts()
    pcts   = counts / counts.sum() * 100
    colors = [CLASS_COLORS.get(c, "#95a5a6") for c in counts.index]

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=0.8)

    # Annotate bars with count and %
    for bar, pct in zip(bars, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            f"{int(bar.get_height()):,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_title("Kepler KOI — Class Distribution", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Disposition", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_ylim(0, counts.max() * 1.2)
    fig.tight_layout()
    _save_or_show(fig, filename, save)


# ---------------------------------------------------------------------------
# 2. Missing values
# ---------------------------------------------------------------------------

def plot_missing_values(
    df: pd.DataFrame,
    top_n: int = 30,
    save: bool = True,
    filename: str = "02_missing_values.png",
) -> None:
    """
    Horizontal bar chart of missing-value percentage for top_n columns.

    Parameters
    ----------
    df    : pd.DataFrame
    top_n : int  – Show only the top N columns by missing-value rate.
    save  : bool
    """
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0].head(top_n)

    if missing_pct.empty:
        log.info("No missing values found — skipping missing-values plot.")
        return

    fig, ax = plt.subplots(figsize=(10, max(6, len(missing_pct) * 0.35)))
    bars = ax.barh(missing_pct.index[::-1], missing_pct.values[::-1],
                   color="#e74c3c", alpha=0.8, edgecolor="white")
    ax.set_xlabel("Missing (%)", fontsize=12)
    ax.set_title(f"Top {top_n} Columns by Missing Value Rate", fontsize=13, fontweight="bold")
    ax.axvline(x=50, color="black", linestyle="--", linewidth=0.8, alpha=0.5, label="50% threshold")
    ax.legend()
    fig.tight_layout()
    _save_or_show(fig, filename, save)


# ---------------------------------------------------------------------------
# 3. Correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    df: pd.DataFrame,
    features: List[str],
    save: bool = True,
    filename: str = "03_correlation_heatmap.png",
) -> None:
    """
    Pearson correlation heatmap for a given list of numeric features.

    Parameters
    ----------
    df       : pd.DataFrame
    features : list[str]  – Numeric feature columns to include.
    save     : bool
    """
    data = df[features].select_dtypes(include=[np.number])
    corr = data.corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))  # Show lower triangle only
    sns.heatmap(
        corr, mask=mask, annot=False, fmt=".2f",
        cmap="coolwarm", center=0, linewidths=0.3,
        ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Matrix (Pearson)", fontsize=13, fontweight="bold", pad=15)
    fig.tight_layout()
    _save_or_show(fig, filename, save)


# ---------------------------------------------------------------------------
# 4. Feature distributions by class
# ---------------------------------------------------------------------------

def plot_feature_distributions(
    df: pd.DataFrame,
    features: List[str],
    target_col: str = TARGET_COL,
    max_features: int = 12,
    save: bool = True,
    filename: str = "04_feature_distributions.png",
) -> None:
    """
    Grid of KDE plots for selected features, coloured by class.

    Parameters
    ----------
    df          : pd.DataFrame
    features    : list[str]  – Features to plot.
    target_col  : str
    max_features: int  – Cap at this many subplots to keep figure readable.
    save        : bool
    """
    features = features[:max_features]
    n_cols = 3
    n_rows = int(np.ceil(len(features) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3.5))
    axes = axes.flatten()

    classes = df[target_col].unique()
    for i, feat in enumerate(features):
        ax = axes[i]
        for cls in classes:
            subset = df[df[target_col] == cls][feat].dropna()
            color  = CLASS_COLORS.get(cls, "#95a5a6")
            subset.plot.kde(ax=ax, label=cls, color=color, linewidth=1.8)
        ax.set_title(feat, fontsize=9, fontweight="bold")
        ax.set_xlabel("")
        ax.legend(fontsize=7)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions by KOI Disposition", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save_or_show(fig, filename, save)


# ---------------------------------------------------------------------------
# 5. Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = CLASS_NAMES,
    title: str = "Confusion Matrix",
    save: bool = True,
    filename: str = "confusion_matrix.png",
) -> None:
    """
    Annotated, normalised confusion matrix heat-map.

    Both raw counts and row-normalised percentages are shown in each cell.

    Parameters
    ----------
    y_true  : np.ndarray  – Ground-truth integer labels.
    y_pred  : np.ndarray  – Predicted integer labels.
    labels  : list[str]   – Human-readable class names (in label-integer order).
    title   : str
    save    : bool
    """
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_norm, annot=False, fmt=".2f",
        cmap="Blues", linewidths=0.5,
        xticklabels=labels, yticklabels=labels,
        ax=ax, cbar_kws={"label": "Proportion"},
        vmin=0, vmax=1,
    )

    # Annotate cells with count (top) and % (bottom)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j + 0.5, i + 0.38,
                f"{cm[i, j]}",
                ha="center", va="center",
                fontsize=11, fontweight="bold",
                color="white" if cm_norm[i, j] > 0.5 else "black",
            )
            ax.text(
                j + 0.5, i + 0.65,
                f"({cm_norm[i, j]:.1%})",
                ha="center", va="center",
                fontsize=8,
                color="white" if cm_norm[i, j] > 0.5 else "black",
            )

    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    _save_or_show(fig, filename, save)


# ---------------------------------------------------------------------------
# 6. ROC curves (One-vs-Rest)
# ---------------------------------------------------------------------------

def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: List[str] = CLASS_NAMES,
    title: str = "ROC Curves (One-vs-Rest)",
    save: bool = True,
    filename: str = "roc_curves.png",
) -> None:
    """
    One-vs-Rest ROC curves for each class with AUC scores in the legend.

    Parameters
    ----------
    y_true : np.ndarray  – Integer ground-truth labels (shape N,).
    y_prob : np.ndarray  – Predicted probabilities (shape N × C).
    labels : list[str]   – Class names.
    title  : str
    save   : bool
    """
    n_classes = len(labels)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    colors = [PALETTE[0], PALETTE[1], PALETTE[2], "#9b59b6", "#f39c12"]

    for i, (label, color) in enumerate(zip(labels, colors)):
        if y_bin.shape[1] == 1:
            # Binary case
            fpr, tpr, _ = roc_curve(y_bin[:, 0], y_prob[:, i])
        else:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{label}  (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random (AUC = 0.500)")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    _save_or_show(fig, filename, save)


# ---------------------------------------------------------------------------
# 7. Precision-Recall curves
# ---------------------------------------------------------------------------

def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: List[str] = CLASS_NAMES,
    title: str = "Precision-Recall Curves",
    save: bool = True,
    filename: str = "pr_curves.png",
) -> None:
    """
    Precision-Recall curves for each class.  Particularly informative for
    imbalanced datasets like this one.

    Parameters
    ----------
    y_true : np.ndarray
    y_prob : np.ndarray  – Shape (N, C)
    labels : list[str]
    title  : str
    save   : bool
    """
    n_classes = len(labels)
    y_bin  = label_binarize(y_true, classes=list(range(n_classes)))
    colors = [PALETTE[0], PALETTE[1], PALETTE[2]]

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    for i, (label, color) in enumerate(zip(labels, colors)):
        col = y_bin[:, i] if y_bin.ndim > 1 else y_bin[:, 0]
        precision, recall, _ = precision_recall_curve(col, y_prob[:, i])
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, color=color, lw=2,
                label=f"{label}  (AP = {pr_auc:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    _save_or_show(fig, filename, save)


# ---------------------------------------------------------------------------
# 8. Training history (CNN)
# ---------------------------------------------------------------------------

def plot_training_history(
    history: Dict[str, List[float]],
    model_name: str = "Genesis CNN",
    save: bool = True,
    filename: str = "training_history.png",
) -> None:
    """
    Two-panel plot of training/validation loss and accuracy across epochs.

    Parameters
    ----------
    history    : dict  – Keys: 'loss', 'val_loss', 'accuracy', 'val_accuracy'
    model_name : str
    save       : bool
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Loss panel ---
    ax1.plot(history.get("loss", []),     label="Train Loss", color=PALETTE[2], lw=2)
    ax1.plot(history.get("val_loss", []), label="Val Loss",   color=PALETTE[1], lw=2, linestyle="--")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.set_title(f"{model_name} — Loss", fontsize=12, fontweight="bold")
    ax1.legend()

    # --- Accuracy panel ---
    ax2.plot(history.get("accuracy", []),     label="Train Acc", color=PALETTE[0], lw=2)
    ax2.plot(history.get("val_accuracy", []), label="Val Acc",   color=PALETTE[1], lw=2, linestyle="--")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.set_title(f"{model_name} — Accuracy", fontsize=12, fontweight="bold")
    ax2.set_ylim([0, 1])
    ax2.legend()

    fig.suptitle(f"{model_name} Training History", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_or_show(fig, filename, save)


# ---------------------------------------------------------------------------
# 9. Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(
    importances: np.ndarray,
    feature_names: List[str],
    model_name: str = "Random Forest",
    top_n: int = 20,
    save: bool = True,
    filename: str = "feature_importance.png",
) -> None:
    """
    Horizontal bar chart of the top-N most important features.

    Parameters
    ----------
    importances   : np.ndarray  – Feature importance scores.
    feature_names : list[str]
    model_name    : str
    top_n         : int         – Show only the top N features.
    save          : bool
    """
    idx = np.argsort(importances)[-top_n:]
    top_names  = [feature_names[i] for i in idx]
    top_scores = importances[idx]

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.4)))
    bars = ax.barh(top_names, top_scores, color=PALETTE[2], alpha=0.85, edgecolor="white")
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title(f"Top-{top_n} Feature Importances — {model_name}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save_or_show(fig, filename, save)


# ---------------------------------------------------------------------------
# 10. Model comparison bar chart
# ---------------------------------------------------------------------------

def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ["accuracy", "f1_macro", "roc_auc_macro"],
    save: bool = True,
    filename: str = "model_comparison.png",
) -> None:
    """
    Grouped bar chart comparing all models across key metrics.

    Parameters
    ----------
    results : dict
        Nested dict: {model_name: {metric_name: value, ...}, ...}
    metrics : list[str]
        Which metrics to show.  Must be keys in each model's inner dict.
    save    : bool

    Example
    -------
    results = {
        "Random Forest" : {"accuracy": 0.88, "f1_macro": 0.85, "roc_auc_macro": 0.91},
        "Genesis CNN"   : {"accuracy": 0.91, "f1_macro": 0.89, "roc_auc_macro": 0.95},
    }
    """
    model_names = list(results.keys())
    n_models    = len(model_names)
    n_metrics   = len(metrics)
    x = np.arange(n_models)
    width = 0.8 / n_metrics
    bar_colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12"]

    fig, ax = plt.subplots(figsize=(max(10, n_models * 2), 6))

    for i, metric in enumerate(metrics):
        values = [results[m].get(metric, 0) for m in model_names]
        offset = (i - n_metrics / 2 + 0.5) * width
        rects  = ax.bar(
            x + offset, values, width * 0.9,
            label=metric.replace("_", " ").title(),
            color=bar_colors[i % len(bar_colors)],
            alpha=0.85, edgecolor="white",
        )
        # Value labels on bars
        for rect, val in zip(rects, values):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=10, rotation=15, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    fig.tight_layout()
    _save_or_show(fig, filename, save)
