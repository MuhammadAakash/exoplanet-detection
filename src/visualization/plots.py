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