"""
eda.py
======
Exploratory Data Analysis module for the Exoplanet Candidate Vetting project.

This module is Stage 2 of the pipeline.  It expects the preprocessing stage
(Stage 1) to have already run so that ``data/processed/koi_processed.csv``
exists alongside the raw file at ``data/raw/koi_data.csv``.

EDA Sections
------------
Section 1 — Dataset Overview
    • Raw shape, column inventory, data types
    • Class distribution & imbalance statistics
    • Dataset completeness summary

Section 2 — Missing Value Analysis
    • Missing rates per column (raw data)
    • Missing rates per feature group (post-drop)

Section 3 — Feature Distributions by Class
    • KDE plots for all transit & stellar features
    • Box-plots for signal quality features
    • Statistical summary table (mean ± std per class)

Section 4 — Correlation Structure
    • Pearson correlation heatmap across the 37 model features
    • High-correlation pair identification (|r| > 0.85)

Section 5 — Key Astrophysical Relationships
    • Period vs Transit Depth scatter (coloured by class)
    • Planet Radius vs Stellar Radius scatter
    • SNR vs Transit Duration scatter
    • False Positive Flag analysis (flag co-occurrence)

Section 6 — Outlier & Range Analysis
    • Per-feature range table
    • Boxplots of key features before/after scaling

All figures are saved to ``results/figures/eda_*.png``.
A summary statistics CSV is saved to ``results/metrics/eda_summary.csv``.

Usage
-----
    python -m src.data.eda          # Run as module
    from src.data.eda import run_eda  # Import for pipeline integration

Author : MSc Data Science Dissertation
Dataset: NASA Kepler Objects of Interest (KOI) — Q1-Q17 DR25
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns

# Allow running as a standalone script from any directory
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import (
    RAW_DATA_FILE, PROCESSED_FILE,
    FIGURES_DIR, METRICS_DIR,
    TARGET_COL, CLASS_NAMES, MULTICLASS_LABEL_MAP,
    TRANSIT_FEATURES, STELLAR_FEATURES,
    PHOTOMETRIC_FLAGS, SIGNAL_FEATURES, MAGNITUDE_FEATURES,
    ALL_MODEL_FEATURES,
    PLOT_STYLE, PLOT_DPI, COLOR_PALETTE,
    DROP_COLS,
)
from src.utils.logger import get_logger
from src.utils.seed import set_all_seeds

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
try:
    plt.style.use(PLOT_STYLE)
except OSError:
    plt.style.use("seaborn-v0_8-whitegrid")

# Consistent colour mapping across all EDA plots
CLASS_COLORS = {
    "CONFIRMED"     : COLOR_PALETTE[0],   # green
    "FALSE POSITIVE": COLOR_PALETTE[1],   # red
    "CANDIDATE"     : COLOR_PALETTE[2],   # blue
}
FEATURE_GROUP_COLORS = {
    "Transit"     : "#3498db",
    "Stellar"     : "#2ecc71",
    "Signal"      : "#e67e22",
    "FP Flags"    : "#e74c3c",
    "Magnitudes"  : "#9b59b6",
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _savefig(fig: plt.Figure, filename: str) -> None:
    """Save a figure to FIGURES_DIR with consistent DPI and tight layout."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / filename
    fig.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved → {out.name}")


def _load_raw() -> pd.DataFrame:
    """Load the raw KOI CSV (skip NASA '#' comment lines)."""
    log.info(f"Loading raw data from {RAW_DATA_FILE.name} …")
    df = pd.read_csv(RAW_DATA_FILE, comment="#", low_memory=False)
    log.info(f"  Raw shape: {df.shape}")
    return df


def _load_processed() -> pd.DataFrame:
    """Load the already-cleaned processed dataframe."""
    log.info(f"Loading processed data from {PROCESSED_FILE.name} …")
    df = pd.read_csv(PROCESSED_FILE, low_memory=False)
    log.info(f"  Processed shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# SECTION 1 — Dataset Overview
# ---------------------------------------------------------------------------

def section1_overview(raw_df: pd.DataFrame, proc_df: pd.DataFrame) -> Dict:
    """
    Print and return a structured overview of the dataset.

    Covers:
    - Total samples and columns in raw vs processed data
    - Class counts, percentages, and imbalance ratio
    - Data types breakdown

    Parameters
    ----------
    raw_df  : pd.DataFrame  Raw KOI CSV (all 141 cols)
    proc_df : pd.DataFrame  Processed dataframe (post-drop, labelled)

    Returns
    -------
    dict  Summary statistics for downstream saving.
    """
    log.info("\n" + "=" * 55)
    log.info("SECTION 1 — Dataset Overview")
    log.info("=" * 55)

    # ── Class distribution ────────────────────────────────────
    class_counts = proc_df[TARGET_COL].value_counts()
    total        = len(proc_df)
    imbalance    = class_counts.max() / class_counts.min()

    log.info(f"\nRaw dataset   : {raw_df.shape[0]:,} rows × {raw_df.shape[1]} columns")
    log.info(f"Processed     : {proc_df.shape[0]:,} rows × {proc_df.shape[1]} columns")
    log.info(f"Columns dropped: {raw_df.shape[1] - proc_df.shape[1]}")
    log.info(f"\nClass distribution:")
    for cls in CLASS_NAMES:
        n   = class_counts.get(cls, 0)
        pct = n / total * 100
        log.info(f"  {cls:20s} : {n:4d}  ({pct:.1f}%)")
    log.info(f"\nImbalance ratio (max/min): {imbalance:.2f}×")

    # ── Figure: enhanced class distribution ──────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Panel A: Bar chart ---
    ax = axes[0]
    colors = [CLASS_COLORS.get(c, "#95a5a6") for c in class_counts.index]
    bars   = ax.bar(class_counts.index, class_counts.values,
                    color=colors, edgecolor="white", linewidth=1.2, width=0.6)

    for bar, (cls, n) in zip(bars, class_counts.items()):
        pct = n / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            f"{n:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_title("KOI Class Distribution", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Disposition",    fontsize=11)
    ax.set_ylabel("Number of KOIs", fontsize=11)
    ax.set_ylim(0, class_counts.max() * 1.28)

    # --- Panel B: Pie chart ---
    ax2 = axes[1]
    wedge_props = {"edgecolor": "white", "linewidth": 2}
    wedges, texts, autotexts = ax2.pie(
        class_counts.values,
        labels    = class_counts.index,
        colors    = colors,
        autopct   = "%1.1f%%",
        startangle= 140,
        wedgeprops= wedge_props,
        textprops  = {"fontsize": 11},
    )
    for at in autotexts:
        at.set_fontweight("bold")
    ax2.set_title("Class Proportion", fontsize=13, fontweight="bold", pad=12)

    fig.suptitle(
        f"NASA Kepler KOI Dataset — {total:,} Objects of Interest",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    _savefig(fig, "eda_01_class_distribution.png")

    summary = {
        "total_samples"     : total,
        "raw_columns"       : raw_df.shape[1],
        "processed_columns" : proc_df.shape[1],
        "class_counts"      : class_counts.to_dict(),
        "imbalance_ratio"   : round(imbalance, 3),
    }
    return summary


# ---------------------------------------------------------------------------
# SECTION 2 — Missing Value Analysis
# ---------------------------------------------------------------------------

def section2_missing_values(raw_df: pd.DataFrame, proc_df: pd.DataFrame) -> None:
    """
    Visualise missing value rates in the raw and processed datasets.

    Two figures:
    1. Raw data — top-30 columns by missing rate (highlights why DROP_COLS
       were removed — many are >80% empty).
    2. Processed feature set — missing rates for the 37 model features,
       grouped by feature category.

    Parameters
    ----------
    raw_df  : pd.DataFrame
    proc_df : pd.DataFrame
    """
    log.info("\n" + "=" * 55)
    log.info("SECTION 2 — Missing Value Analysis")
    log.info("=" * 55)

    # ── Figure A: Raw data missing values ────────────────────
    raw_missing = (raw_df.isnull().sum() / len(raw_df) * 100).sort_values(ascending=False)
    raw_missing = raw_missing[raw_missing > 0].head(35)

    fig, ax = plt.subplots(figsize=(11, max(7, len(raw_missing) * 0.32)))
    colors = ["#e74c3c" if v > 50 else "#f39c12" if v > 20 else "#3498db"
              for v in raw_missing.values]
    ax.barh(raw_missing.index[::-1], raw_missing.values[::-1],
            color=colors[::-1], alpha=0.85, edgecolor="white")
    ax.set_xlabel("Missing Value Rate (%)", fontsize=11)
    ax.set_title(
        "Raw Data — Top 35 Columns by Missing Value Rate\n"
        "(Red >50%, Orange >20%)",
        fontsize=12, fontweight="bold",
    )
    ax.axvline(x=50, color="black", linestyle="--", linewidth=1.0,
               alpha=0.6, label="50% threshold (drop boundary)")
    ax.axvline(x=20, color="gray",  linestyle=":",  linewidth=1.0,
               alpha=0.5, label="20% threshold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _savefig(fig, "eda_02a_missing_raw.png")

    n_heavy = (raw_missing > 50).sum()
    log.info(f"  {n_heavy} columns have >50% missing values in raw data → dropped")

    # ── Figure B: Processed feature set missing rates ─────────
    feat_missing = (proc_df[ALL_MODEL_FEATURES].isnull().sum() / len(proc_df) * 100)

    # Group features by category for colour-coding
    feature_groups = {}
    for f in ALL_MODEL_FEATURES:
        if f in TRANSIT_FEATURES:
            feature_groups[f] = "Transit"
        elif f in STELLAR_FEATURES:
            feature_groups[f] = "Stellar"
        elif f in SIGNAL_FEATURES:
            feature_groups[f] = "Signal"
        elif f in PHOTOMETRIC_FLAGS:
            feature_groups[f] = "FP Flags"
        elif f in MAGNITUDE_FEATURES:
            feature_groups[f] = "Magnitudes"
        else:
            feature_groups[f] = "Other"

    feat_missing_df = pd.DataFrame({
        "feature"     : feat_missing.index,
        "missing_pct" : feat_missing.values,
        "group"       : [feature_groups.get(f, "Other") for f in feat_missing.index],
    }).sort_values("missing_pct", ascending=False)

    feat_missing_with_values = feat_missing_df[feat_missing_df["missing_pct"] > 0]

    fig, ax = plt.subplots(figsize=(10, max(5, len(feat_missing_with_values) * 0.45)))
    if len(feat_missing_with_values) > 0:
        bar_colors = [FEATURE_GROUP_COLORS.get(g, "#95a5a6")
                      for g in feat_missing_with_values["group"]]
        ax.barh(
            feat_missing_with_values["feature"][::-1],
            feat_missing_with_values["missing_pct"][::-1],
            color=bar_colors[::-1], alpha=0.85, edgecolor="white",
        )
        ax.set_xlabel("Missing Value Rate (%)", fontsize=11)

        # Legend for groups
        legend_patches = [
            mpatches.Patch(color=c, label=g)
            for g, c in FEATURE_GROUP_COLORS.items()
            if g in feat_missing_with_values["group"].values
        ]
        ax.legend(handles=legend_patches, fontsize=9, loc="lower right")
    else:
        ax.text(0.5, 0.5, "No missing values in model features\n(after imputation)",
                ha="center", va="center", transform=ax.transAxes, fontsize=13)

    ax.set_title(
        "Model Features — Missing Value Rate by Category\n"
        "(After preprocessing; imputation applied to all)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    _savefig(fig, "eda_02b_missing_features.png")

    total_missing = (feat_missing > 0).sum()
    log.info(f"  {total_missing} of 37 model features have some missing values → median imputed")


# ---------------------------------------------------------------------------
# SECTION 3 — Feature Distributions by Class
# ---------------------------------------------------------------------------

def section3_feature_distributions(proc_df: pd.DataFrame) -> pd.DataFrame:
    """
    KDE and box plots for all transit and stellar features, coloured by class.

    Also computes and returns a statistical summary table (mean ± std per class)
    which is useful for the dissertation results chapter.

    Parameters
    ----------
    proc_df : pd.DataFrame  Processed KOI dataframe with TARGET_COL present.

    Returns
    -------
    pd.DataFrame  Statistical summary for saving to CSV.
    """
    log.info("\n" + "=" * 55)
    log.info("SECTION 3 — Feature Distributions by Class")
    log.info("=" * 55)

    classes = CLASS_NAMES
    available = [f for f in ALL_MODEL_FEATURES if f in proc_df.columns]

    # ── Figure A: Transit feature KDE grid ───────────────────
    transit_avail = [f for f in TRANSIT_FEATURES if f in proc_df.columns]
    n_cols = 3
    n_rows = int(np.ceil(len(transit_avail) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3.5))
    axes = axes.flatten()

    for i, feat in enumerate(transit_avail):
        ax = axes[i]
        for cls in classes:
            subset = proc_df[proc_df[TARGET_COL] == cls][feat].dropna()
            if len(subset) < 5:
                continue
            # Clip to 99th percentile to avoid extreme outliers distorting KDE
            clip_upper = subset.quantile(0.99)
            subset_clipped = subset.clip(upper=clip_upper)
            subset_clipped.plot.kde(
                ax=ax, label=cls,
                color=CLASS_COLORS.get(cls, "#95a5a6"),
                linewidth=2.0, alpha=0.85,
            )
        ax.set_title(feat, fontsize=9, fontweight="bold", pad=4)
        ax.set_xlabel("", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_axisbelow(True)

    for j in range(len(transit_avail), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Transit Parameter Distributions by KOI Disposition (KDE)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    _savefig(fig, "eda_03a_transit_distributions.png")

    # ── Figure B: Stellar feature box plots ──────────────────
    stellar_avail = [f for f in STELLAR_FEATURES if f in proc_df.columns]

    n_cols_s = 3
    n_rows_s = int(np.ceil(len(stellar_avail) / n_cols_s))
    fig, axes = plt.subplots(n_rows_s, n_cols_s, figsize=(15, n_rows_s * 3.5))
    axes = axes.flatten()

    for i, feat in enumerate(stellar_avail):
        ax = axes[i]
        data_per_class = [
            proc_df[proc_df[TARGET_COL] == cls][feat].dropna().values
            for cls in classes
        ]
        bp = ax.boxplot(
            data_per_class,
            patch_artist=True,
            notch=False,
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            flierprops=dict(marker="o", markersize=2, alpha=0.3),
        )
        for patch, cls in zip(bp["boxes"], classes):
            patch.set_facecolor(CLASS_COLORS.get(cls, "#95a5a6"))
            patch.set_alpha(0.7)

        ax.set_xticks(range(1, len(classes) + 1))
        ax.set_xticklabels(
            [c.replace(" ", "\n") for c in classes],
            fontsize=7,
        )
        ax.set_title(feat, fontsize=9, fontweight="bold", pad=4)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_axisbelow(True)

    for j in range(len(stellar_avail), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Stellar Parameter Distributions by KOI Disposition (Box Plots)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    _savefig(fig, "eda_03b_stellar_distributions.png")

    # ── Figure C: Signal quality KDE ─────────────────────────
    signal_avail = [f for f in SIGNAL_FEATURES if f in proc_df.columns]
    n_cols_q = min(3, len(signal_avail))
    n_rows_q = int(np.ceil(len(signal_avail) / n_cols_q))

    fig, axes = plt.subplots(n_rows_q, n_cols_q, figsize=(15, n_rows_q * 3.5))
    axes = np.array(axes).flatten()

    for i, feat in enumerate(signal_avail):
        ax = axes[i]
        for cls in classes:
            subset = proc_df[proc_df[TARGET_COL] == cls][feat].dropna()
            clip_upper = subset.quantile(0.99)
            subset.clip(upper=clip_upper).plot.kde(
                ax=ax, label=cls,
                color=CLASS_COLORS.get(cls, "#95a5a6"),
                linewidth=2.0,
            )
        ax.set_title(feat, fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
        ax.set_axisbelow(True)

    for j in range(len(signal_avail), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Signal Quality Feature Distributions by Class",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    _savefig(fig, "eda_03c_signal_distributions.png")

    # ── Statistical summary table ─────────────────────────────
    summary_rows = []
    for feat in available:
        row = {"feature": feat}
        for cls in classes:
            vals = proc_df[proc_df[TARGET_COL] == cls][feat].dropna()
            cls_key = cls.replace(" ", "_").lower()
            row[f"{cls_key}_mean"] = round(vals.mean(), 4) if len(vals) > 0 else np.nan
            row[f"{cls_key}_std"]  = round(vals.std(),  4) if len(vals) > 0 else np.nan
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(METRICS_DIR / "eda_feature_stats_by_class.csv", index=False)
    log.info(f"  Statistical summary saved → eda_feature_stats_by_class.csv")

    return summary_df


# ---------------------------------------------------------------------------
# SECTION 4 — Correlation Structure
# ---------------------------------------------------------------------------

def section4_correlations(proc_df: pd.DataFrame) -> None:
    """
    Pearson correlation heatmap of the 37 model features.

    Also prints the top 10 highly-correlated feature pairs (|r| > 0.85),
    which is relevant for the dissertation's feature analysis discussion.

    Parameters
    ----------
    proc_df : pd.DataFrame
    """
    log.info("\n" + "=" * 55)
    log.info("SECTION 4 — Correlation Structure")
    log.info("=" * 55)

    available = [f for f in ALL_MODEL_FEATURES if f in proc_df.columns]
    corr = proc_df[available].corr()

    # ── Figure: Full correlation heatmap ─────────────────────
    fig, ax = plt.subplots(figsize=(16, 13))
    mask = np.triu(np.ones_like(corr, dtype=bool))  # Lower triangle only

    sns.heatmap(
        corr,
        mask      = mask,
        annot     = False,
        cmap      = "coolwarm",
        center    = 0,
        linewidths= 0.3,
        ax        = ax,
        cbar_kws  = {"shrink": 0.75, "label": "Pearson r"},
        vmin      = -1, vmax=1,
    )
    ax.set_title(
        "Feature Correlation Matrix (Pearson r)\n"
        "37 Astrophysical Features — NASA Kepler KOI",
        fontsize=13, fontweight="bold", pad=15,
    )
    ax.tick_params(axis="x", labelsize=7, rotation=45)
    ax.tick_params(axis="y", labelsize=7, rotation=0)
    fig.tight_layout()
    _savefig(fig, "eda_04a_correlation_heatmap.png")

    # ── Figure: Top correlated pairs ─────────────────────────
    # Extract upper triangle pairs
    corr_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            r = corr.iloc[i, j]
            corr_pairs.append({
                "feature_1": corr.columns[i],
                "feature_2": corr.columns[j],
                "pearson_r": round(r, 4),
                "abs_r"    : abs(r),
            })

    corr_pairs_df = (
        pd.DataFrame(corr_pairs)
        .sort_values("abs_r", ascending=False)
    )

    high_corr = corr_pairs_df[corr_pairs_df["abs_r"] > 0.75].head(15)

    fig, ax = plt.subplots(figsize=(10, 5))
    y_labels = [f"{r['feature_1']}  ↔  {r['feature_2']}" for _, r in high_corr.iterrows()]
    bar_colors = ["#e74c3c" if r > 0 else "#3498db" for r in high_corr["pearson_r"]]

    ax.barh(
        range(len(high_corr)), high_corr["abs_r"].values,
        color=bar_colors, alpha=0.8, edgecolor="white",
    )
    ax.set_yticks(range(len(high_corr)))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("|Pearson r|", fontsize=11)
    ax.set_title(
        "Top Feature Pairs by Absolute Correlation (|r| > 0.75)",
        fontsize=12, fontweight="bold",
    )
    ax.axvline(x=0.85, color="black", linestyle="--", linewidth=1,
               alpha=0.6, label="|r| = 0.85 (high correlation threshold)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _savefig(fig, "eda_04b_high_correlation_pairs.png")

    # Log notable correlations
    log.info(f"\n  High-correlation pairs (|r| > 0.85):")
    for _, row in corr_pairs_df[corr_pairs_df["abs_r"] > 0.85].iterrows():
        log.info(f"    {row['feature_1']:25s} ↔ {row['feature_2']:25s}  r = {row['pearson_r']:.3f}")


# ---------------------------------------------------------------------------
# SECTION 5 — Key Astrophysical Relationships
# ---------------------------------------------------------------------------

def section5_astrophysical_relationships(proc_df: pd.DataFrame) -> None:
    """
    Scatter plots of physically meaningful feature pairs, coloured by class.

    These plots directly answer the research question by showing how well
    the three KOI classes separate in 2D feature subspaces.

    Plots generated:
    A. Orbital Period vs Transit Depth           (key planetary vs EB diagnostic)
    B. Planet Radius vs Stellar Radius           (size-based diagnostic)
    C. Model SNR vs Max Multiple Event Statistic (signal quality)
    D. False Positive Flag co-occurrence heatmap

    Parameters
    ----------
    proc_df : pd.DataFrame
    """
    log.info("\n" + "=" * 55)
    log.info("SECTION 5 — Astrophysical Relationships")
    log.info("=" * 55)

    # ── Figure A: Period vs Depth ─────────────────────────────
    if "koi_period" in proc_df.columns and "koi_depth" in proc_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))

        for cls in CLASS_NAMES:
            sub = proc_df[proc_df[TARGET_COL] == cls]
            x   = sub["koi_period"].clip(0, sub["koi_period"].quantile(0.98))
            y   = sub["koi_depth"].clip(0,  sub["koi_depth"].quantile(0.98))
            ax.scatter(x, y,
                       c=CLASS_COLORS.get(cls), label=cls,
                       alpha=0.35, s=12, edgecolors="none")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Orbital Period (days) — log scale", fontsize=11)
        ax.set_ylabel("Transit Depth (ppm) — log scale",  fontsize=11)
        ax.set_title(
            "Orbital Period vs Transit Depth\n"
            "(Key diagnostic: FPs often have deep transits at short periods)",
            fontsize=12, fontweight="bold",
        )
        ax.legend(markerscale=3, fontsize=10)
        ax.set_axisbelow(True)
        fig.tight_layout()
        _savefig(fig, "eda_05a_period_vs_depth.png")

    # ── Figure B: Planet Radius vs Stellar Radius ────────────
    if "koi_prad" in proc_df.columns and "koi_srad" in proc_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))

        for cls in CLASS_NAMES:
            sub = proc_df[proc_df[TARGET_COL] == cls]
            x   = sub["koi_prad"].clip(0, sub["koi_prad"].quantile(0.97))
            y   = sub["koi_srad"].clip(0, sub["koi_srad"].quantile(0.97))
            ax.scatter(x, y,
                       c=CLASS_COLORS.get(cls), label=cls,
                       alpha=0.4, s=12, edgecolors="none")

        ax.set_xlabel("Planet Radius (Earth Radii)", fontsize=11)
        ax.set_ylabel("Stellar Radius (Solar Radii)", fontsize=11)
        ax.set_title(
            "Planet Radius vs Host Star Radius\n"
            "(Large 'planet' radii > 15 R⊕ are likely eclipsing binaries)",
            fontsize=12, fontweight="bold",
        )
        # Annotate the Jupiter-size boundary (~11 Earth radii)
        ax.axvline(x=11.2, color="gray", linestyle="--", linewidth=1.0, alpha=0.6,
                   label="Jupiter radius (11.2 R⊕)")
        ax.legend(markerscale=3, fontsize=10)
        ax.set_axisbelow(True)
        fig.tight_layout()
        _savefig(fig, "eda_05b_planet_vs_stellar_radius.png")

    # ── Figure C: SNR vs Transit Duration ────────────────────
    if "koi_model_snr" in proc_df.columns and "koi_duration" in proc_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))

        for cls in CLASS_NAMES:
            sub = proc_df[proc_df[TARGET_COL] == cls]
            x   = sub["koi_duration"].clip(0,  sub["koi_duration"].quantile(0.97))
            y   = sub["koi_model_snr"].clip(0, sub["koi_model_snr"].quantile(0.97))
            ax.scatter(x, y,
                       c=CLASS_COLORS.get(cls), label=cls,
                       alpha=0.4, s=12, edgecolors="none")

        ax.set_xlabel("Transit Duration (hours)", fontsize=11)
        ax.set_ylabel("Model SNR",                fontsize=11)
        ax.set_title(
            "Transit Duration vs Model Signal-to-Noise Ratio\n"
            "(High-SNR, long-duration signals are more likely genuine planets)",
            fontsize=12, fontweight="bold",
        )
        ax.legend(markerscale=3, fontsize=10)
        ax.set_axisbelow(True)
        fig.tight_layout()
        _savefig(fig, "eda_05c_duration_vs_snr.png")

    # ── Figure D: FP Flag co-occurrence heatmap ───────────────
    flag_cols = [f for f in PHOTOMETRIC_FLAGS if f in proc_df.columns]
    if len(flag_cols) >= 2:
        fig, axes = plt.subplots(1, len(CLASS_NAMES), figsize=(15, 5))

        for ax, cls in zip(axes, CLASS_NAMES):
            sub        = proc_df[proc_df[TARGET_COL] == cls][flag_cols]
            co_occur   = sub.T.dot(sub)   # flag co-occurrence count matrix
            # Normalise by class size
            co_occur_n = co_occur / len(sub)

            sns.heatmap(
                co_occur_n,
                annot      = True,
                fmt        = ".2f",
                cmap       = "Reds",
                linewidths = 0.5,
                ax         = ax,
                cbar       = False,
                vmin=0, vmax=1,
                annot_kws  = {"size": 8},
            )
            ax.set_title(
                f"{cls}\n(n={len(sub):,})",
                fontsize=10, fontweight="bold",
            )
            short_names = [c.replace("koi_fpflag_", "") for c in flag_cols]
            ax.set_xticklabels(short_names, rotation=45, fontsize=8)
            ax.set_yticklabels(short_names, rotation=0,  fontsize=8)

        fig.suptitle(
            "False Positive Flag Co-occurrence by Class\n"
            "(Proportion of KOIs in each class with each flag combination set)",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout()
        _savefig(fig, "eda_05d_fp_flag_cooccurrence.png")

        # Log flag rates
        log.info("\n  False Positive Flag Rates by Class:")
        for cls in CLASS_NAMES:
            sub = proc_df[proc_df[TARGET_COL] == cls]
            rates = sub[flag_cols].mean() * 100
            log.info(f"\n    {cls}:")
            for flag, rate in rates.items():
                log.info(f"      {flag:25s}: {rate:.1f}%")


# ---------------------------------------------------------------------------
# SECTION 6 — Outlier & Range Analysis
# ---------------------------------------------------------------------------

def section6_outliers(proc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-feature range statistics and outlier identification.

    Generates a summary table and a figure showing the number of outliers
    (values beyond 3 standard deviations) per feature.  This is relevant
    for justifying the use of StandardScaler and median imputation.

    Parameters
    ----------
    proc_df : pd.DataFrame

    Returns
    -------
    pd.DataFrame  Range statistics per feature.
    """
    log.info("\n" + "=" * 55)
    log.info("SECTION 6 — Outlier & Range Analysis")
    log.info("=" * 55)

    available = [f for f in ALL_MODEL_FEATURES if f in proc_df.columns]
    feat_data = proc_df[available]

    # ── Range statistics table ────────────────────────────────
    stats = feat_data.agg(["min", "max", "mean", "median", "std"]).T
    stats["iqr"] = feat_data.quantile(0.75) - feat_data.quantile(0.25)
    stats["skewness"] = feat_data.skew()

    # Count 3-sigma outliers per feature
    z_scores = (feat_data - feat_data.mean()) / feat_data.std()
    outlier_counts = (z_scores.abs() > 3).sum()
    outlier_pcts   = outlier_counts / len(feat_data) * 100
    stats["outliers_count"] = outlier_counts
    stats["outliers_pct"]   = outlier_pcts.round(2)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    stats.to_csv(METRICS_DIR / "eda_feature_ranges.csv")
    log.info("  Feature range statistics saved → eda_feature_ranges.csv")

    # ── Figure: Outlier counts per feature ───────────────────
    outliers_sorted = outlier_pcts.sort_values(ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_colors = [
        FEATURE_GROUP_COLORS.get(
            "Transit" if f in TRANSIT_FEATURES
            else "Stellar" if f in STELLAR_FEATURES
            else "Signal" if f in SIGNAL_FEATURES
            else "FP Flags" if f in PHOTOMETRIC_FLAGS
            else "Magnitudes", "#95a5a6"
        )
        for f in outliers_sorted.index
    ]
    ax.barh(
        outliers_sorted.index[::-1],
        outliers_sorted.values[::-1],
        color=bar_colors[::-1], alpha=0.85, edgecolor="white",
    )
    ax.set_xlabel("Outliers > 3σ (%)", fontsize=11)
    ax.set_title(
        "Features with Most Outliers (> 3σ from Mean)\n"
        "Justifies median imputation and StandardScaler preprocessing",
        fontsize=12, fontweight="bold",
    )

    legend_patches = [
        mpatches.Patch(color=c, label=g)
        for g, c in FEATURE_GROUP_COLORS.items()
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc="lower right")
    fig.tight_layout()
    _savefig(fig, "eda_06_outlier_analysis.png")

    log.info(
        f"\n  Average outlier rate across all features: "
        f"{outlier_pcts.mean():.2f}%"
    )
    return stats


# ---------------------------------------------------------------------------
# SECTION 7 — Feature Importance Preview (before modelling)
# ---------------------------------------------------------------------------

def section7_class_separability(proc_df: pd.DataFrame) -> None:
    """
    Visualise the class separability of key features using violin plots
    and a statistical ANOVA F-score ranking.

    The F-score (one-way ANOVA) measures how much variance in a feature
    is explained by class membership.  Features with high F-scores are
    likely to be informative for classification.

    This is a pre-modelling EDA diagnostic — not a substitute for model
    feature importances, but a domain-guided sanity check.

    Parameters
    ----------
    proc_df : pd.DataFrame
    """
    log.info("\n" + "=" * 55)
    log.info("SECTION 7 — Class Separability Preview")
    log.info("=" * 55)

    from sklearn.feature_selection import f_classif
    from sklearn.impute import SimpleImputer

    available = [f for f in ALL_MODEL_FEATURES if f in proc_df.columns]
    X = proc_df[available].copy()
    y = proc_df["label"].copy()

    # Impute for ANOVA computation (median)
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)

    f_scores, p_values = f_classif(X_imp, y)

    anova_df = pd.DataFrame({
        "feature": available,
        "f_score": f_scores,
        "p_value": p_values,
    }).sort_values("f_score", ascending=False)

    # ── Figure A: ANOVA F-score ranking ──────────────────────
    top20 = anova_df.head(20)
    group_colors = [
        FEATURE_GROUP_COLORS.get(
            "Transit"   if f in TRANSIT_FEATURES
            else "Stellar" if f in STELLAR_FEATURES
            else "Signal"  if f in SIGNAL_FEATURES
            else "FP Flags" if f in PHOTOMETRIC_FLAGS
            else "Magnitudes", "#95a5a6"
        )
        for f in top20["feature"]
    ]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        top20["feature"][::-1],
        top20["f_score"][::-1],
        color=group_colors[::-1], alpha=0.85, edgecolor="white",
    )
    ax.set_xlabel("ANOVA F-Score (higher = more class-discriminative)", fontsize=11)
    ax.set_title(
        "Top 20 Features by Class Separability (One-Way ANOVA)\n"
        "Pre-modelling EDA diagnostic — coloured by feature group",
        fontsize=12, fontweight="bold",
    )
    legend_patches = [
        mpatches.Patch(color=c, label=g)
        for g, c in FEATURE_GROUP_COLORS.items()
        if g in [
            "Transit" if f in TRANSIT_FEATURES
            else "Stellar" if f in STELLAR_FEATURES
            else "Signal"  if f in SIGNAL_FEATURES
            else "FP Flags" if f in PHOTOMETRIC_FLAGS
            else "Magnitudes"
            for f in top20["feature"]
        ]
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc="lower right")
    fig.tight_layout()
    _savefig(fig, "eda_07a_anova_f_scores.png")

    # ── Figure B: Violin plots for top-6 features ────────────
    top6 = anova_df.head(6)["feature"].tolist()

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for i, feat in enumerate(top6):
        ax = axes[i]
        plot_data = [
            proc_df[proc_df[TARGET_COL] == cls][feat].dropna().clip(
                proc_df[feat].quantile(0.01),
                proc_df[feat].quantile(0.99),
            ).values
            for cls in CLASS_NAMES
        ]
        parts = ax.violinplot(
            plot_data,
            positions   = range(len(CLASS_NAMES)),
            showmedians = True,
            showextrema = True,
        )
        for j, (pc, cls) in enumerate(zip(parts["bodies"], CLASS_NAMES)):
            pc.set_facecolor(CLASS_COLORS.get(cls))
            pc.set_alpha(0.7)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(2)

        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels([c.replace(" ", "\n") for c in CLASS_NAMES], fontsize=8)
        f_val = anova_df[anova_df["feature"] == feat]["f_score"].values[0]
        ax.set_title(f"{feat}\n(F = {f_val:.1f})", fontsize=9, fontweight="bold")
        ax.set_axisbelow(True)

    fig.suptitle(
        "Violin Plots — Top 6 Class-Discriminative Features\n"
        "(Width = density of values; line = median)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    _savefig(fig, "eda_07b_violin_top6_features.png")

    # Save ANOVA results
    anova_df.to_csv(METRICS_DIR / "eda_anova_f_scores.csv", index=False)
    log.info("\n  Top 10 class-discriminative features (ANOVA F-score):")
    for _, row in anova_df.head(10).iterrows():
        significance = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*"
        log.info(f"    {row['feature']:30s}  F = {row['f_score']:8.1f}  {significance}")


# ---------------------------------------------------------------------------
# SECTION 8 — KOI Score Distribution (pre-modelling sanity check)
# ---------------------------------------------------------------------------

def section8_koi_score(raw_df: pd.DataFrame) -> None:
    """
    Analyse the distribution of the NASA pipeline disposition score (koi_score).

    koi_score is the probability assigned by the Robovetter pipeline (0–1).
    Although we exclude it from model features (label leakage), visualising
    it confirms that the human dispositions are consistent with automated
    scoring — a data quality check.

    Parameters
    ----------
    raw_df : pd.DataFrame  Raw KOI dataframe containing koi_score.
    """
    if "koi_score" not in raw_df.columns:
        log.warning("koi_score not found in raw data — skipping Section 8.")
        return

    log.info("\n" + "=" * 55)
    log.info("SECTION 8 — KOI Score Distribution (Data Quality Check)")
    log.info("=" * 55)

    valid = raw_df[raw_df[TARGET_COL].isin(CLASS_NAMES)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: KDE per class
    ax = axes[0]
    for cls in CLASS_NAMES:
        sub = valid[valid[TARGET_COL] == cls]["koi_score"].dropna()
        sub.plot.kde(
            ax=ax, label=cls,
            color=CLASS_COLORS.get(cls), linewidth=2.2,
        )
    ax.set_xlabel("Robovetter Disposition Score (koi_score)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(
        "Robovetter Score by Human Disposition\n"
        "(Excluded from model features — label leakage check)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.set_xlim(-0.05, 1.05)
    ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.6,
               label="Decision threshold 0.5")

    # Panel B: Box plot
    ax2 = axes[1]
    score_by_class = [
        valid[valid[TARGET_COL] == cls]["koi_score"].dropna().values
        for cls in CLASS_NAMES
    ]
    bp = ax2.boxplot(
        score_by_class, patch_artist=True, notch=False,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, cls in zip(bp["boxes"], CLASS_NAMES):
        patch.set_facecolor(CLASS_COLORS.get(cls))
        patch.set_alpha(0.7)
    ax2.set_xticks(range(1, len(CLASS_NAMES) + 1))
    ax2.set_xticklabels(CLASS_NAMES, fontsize=9)
    ax2.set_ylabel("koi_score", fontsize=11)
    ax2.set_title("Score Distribution (Box Plots)", fontsize=11, fontweight="bold")
    ax2.set_ylim(-0.05, 1.05)

    fig.suptitle(
        "Data Quality Check — Robovetter Score vs Human Disposition\n"
        "⚠ koi_score is EXCLUDED from model inputs (label leakage)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    _savefig(fig, "eda_08_koi_score_distribution.png")

    for cls in CLASS_NAMES:
        sub = valid[valid[TARGET_COL] == cls]["koi_score"].dropna()
        log.info(
            f"  {cls:20s}  median score = {sub.median():.3f}  "
            f"mean = {sub.mean():.3f}"
        )


# ---------------------------------------------------------------------------
# Master EDA orchestrator
# ---------------------------------------------------------------------------

def run_eda() -> None:
    """
    Execute all EDA sections in sequence and save a consolidated summary.

    This is the function called by ``run_pipeline.py --stage 2``.

    Output
    ------
    10 dissertation-quality figures in ``results/figures/eda_*.png``
    3 CSV summary files in ``results/metrics/``
    """
    set_all_seeds()

    log.info("\n" + "╔" + "═" * 54 + "╗")
    log.info("║  STAGE 2 — EXPLORATORY DATA ANALYSIS                  ║")
    log.info("║  Exoplanet Candidate Vetting — NASA Kepler KOI         ║")
    log.info("╚" + "═" * 54 + "╝")

    # Load data
    raw_df  = _load_raw()
    proc_df = _load_processed()

    # Run all sections
    summary      = section1_overview(raw_df, proc_df)
    section2_missing_values(raw_df, proc_df)
    feat_stats   = section3_feature_distributions(proc_df)
    section4_correlations(proc_df)
    section5_astrophysical_relationships(proc_df)
    range_stats  = section6_outliers(proc_df)
    section7_class_separability(proc_df)
    section8_koi_score(raw_df)

    # ── Save consolidated EDA summary ────────────────────────
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("\n" + "=" * 55)
    log.info("EDA COMPLETE")
    log.info(f"  Figures saved to : {FIGURES_DIR}")
    log.info(f"  Metrics saved to : {METRICS_DIR}")
    log.info("=" * 55)


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_eda()