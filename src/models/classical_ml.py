"""
classical_ml.py
===============
Stage 3 — Classical Machine Learning Models for Exoplanet Candidate Vetting.

Trains and evaluates three classical ML models on the preprocessed KOI
tabular feature set:

  1. Random Forest       — ensemble method, best interpretability
  2. Support Vector Machine (RBF) — strong baseline for tabular data
  3. Logistic Regression — linear baseline, most interpretable

Each model is:
  - Trained on train + validation splits combined (train_val)
  - Evaluated on the held-out test set
  - Saved to results/models/ as a .pkl file
  - Evaluated using the shared evaluate_model() utility
  - Visualised with confusion matrices, ROC curves, and feature importances

Usage
-----
    python -m src.models.classical_ml

    # Or called by run_pipeline.py Stage 3:
    from src.models.classical_ml import train_and_evaluate_all
    train_and_evaluate_all()

Author : MSc Data Science Dissertation
Dataset: NASA Kepler Objects of Interest (KOI) Q1-Q17 DR25
"""

from __future__ import annotations

import sys
import pickle
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc as sk_auc
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import (
    TRAIN_FILE, VALIDATION_FILE, TEST_FILE,
    MODELS_DIR, FIGURES_DIR, METRICS_DIR,
    CLASS_NAMES, RANDOM_SEED,
)
from src.utils.logger  import get_logger
from src.utils.seed    import set_all_seeds
from src.evaluation.metrics import evaluate_model

log = get_logger(__name__)

# ── Colour palette (consistent with EDA plots) ───────────────────────────────
_PALETTE = {
    "CONFIRMED":      "#2ecc71",
    "FALSE POSITIVE": "#e74c3c",
    "CANDIDATE":      "#3498db",
}
_COLORS = [_PALETTE[c] for c in CLASS_NAMES]

# ── Model definitions ─────────────────────────────────────────────────────────

def _get_models() -> Dict[str, Any]:
    """Return the three classical ML models with tuned hyperparameters."""
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators  = 300,
            max_depth      = None,
            min_samples_split = 2,
            class_weight   = "balanced",
            random_state   = RANDOM_SEED,
            n_jobs         = -1,
        ),
        "SVM": SVC(
            kernel       = "rbf",
            C            = 10,
            gamma        = "scale",
            probability  = True,          # needed for ROC-AUC
            class_weight = "balanced",
            random_state = RANDOM_SEED,
        ),
        "Logistic Regression": LogisticRegression(
            C            = 1.0,
            max_iter     = 1000,
            class_weight = "balanced",
            random_state = RANDOM_SEED,
            n_jobs       = -1,
        ),
    }


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_splits() -> Tuple[np.ndarray, ...]:
    """
    Load train / val / test CSVs produced by Stage 1 preprocessing.
    Combines train and val into a single training set for final models.

    Returns
    -------
    X_trainval, X_test, y_trainval, y_test, feature_names
    """
    log.info("Loading preprocessed splits …")
    train_df = pd.read_csv(TRAIN_FILE)
    val_df   = pd.read_csv(VALIDATION_FILE)
    test_df  = pd.read_csv(TEST_FILE)

    feature_cols = [c for c in train_df.columns if c != "label"]

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    X_val   = val_df[feature_cols].values
    y_val   = val_df["label"].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df["label"].values

    # Combine train + val for final training (val was used for EDA / CNN tuning)
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    log.info(f"  Train+Val : {X_trainval.shape[0]} samples")
    log.info(f"  Test      : {X_test.shape[0]} samples")
    log.info(f"  Features  : {len(feature_cols)}")

    return X_trainval, X_test, y_trainval, y_test, feature_cols


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _plot_confusion_matrices(
    results: Dict[str, Dict],
    y_test:  np.ndarray,
) -> None:
    """Save a 1×3 grid of normalised confusion matrices."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Confusion Matrices — Classical ML Models (Test Set)",
        fontsize=14, fontweight="bold", y=1.02,
    )

    for ax, (name, res) in zip(axes, results.items()):
        cm   = confusion_matrix(y_test, res["y_pred"])
        cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        sns.heatmap(
            cm_n, annot=cm, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.5, ax=ax, cbar=False,
            annot_kws={"size": 10, "weight": "bold"},
        )
        ax.set_title(name, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True",      fontsize=9)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.tick_params(axis="y", rotation=0,  labelsize=8)

    plt.tight_layout()
    path = FIGURES_DIR / "ml_01_confusion_matrices.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved → {path.name}")


def _plot_per_class_f1(results: Dict[str, Dict]) -> None:
    """Heatmap of per-class F1 for all three models."""
    models = list(results.keys())
    f1_data = np.array([
        [
            results[m]["metrics"][f"f1_{c.replace(' ', '_').lower()}"]
            for c in CLASS_NAMES
        ]
        for m in models
    ])

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        f1_data, annot=True, fmt=".3f", cmap="YlOrRd",
        xticklabels=CLASS_NAMES, yticklabels=models,
        vmin=0, vmax=1, linewidths=0.5, ax=ax,
        annot_kws={"size": 12, "weight": "bold"},
    )
    ax.set_title("Per-Class F1 Score — Classical ML", fontsize=12, fontweight="bold")
    ax.set_xlabel("Class", fontsize=10)
    ax.set_ylabel("Model", fontsize=10)
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    path = FIGURES_DIR / "ml_02_per_class_f1.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved → {path.name}")


def _plot_roc_curves(
    results: Dict[str, Dict],
    y_test:  np.ndarray,
) -> None:
    """One-vs-Rest ROC curves for each model."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "ROC Curves — One-vs-Rest (Test Set)",
        fontsize=14, fontweight="bold",
    )
    y_bin = label_binarize(y_test, classes=[0, 1, 2])

    for ax, (name, res) in zip(axes, results.items()):
        y_proba = res["y_proba"]
        for i, (label, color) in enumerate(zip(CLASS_NAMES, _COLORS)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            roc_auc = sk_auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f"{label} (AUC={roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        ax.set_xlabel("False Positive Rate", fontsize=9)
        ax.set_ylabel("True Positive Rate",  fontsize=9)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "ml_03_roc_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved → {path.name}")


def _plot_feature_importance(
    rf_model:      RandomForestClassifier,
    feature_names: list,
) -> None:
    """Horizontal bar chart of RF Gini importances (top 20)."""
    importances = rf_model.feature_importances_
    idx = np.argsort(importances)[::-1][:20]
    top_names  = [feature_names[i] for i in idx]
    top_values = importances[idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        range(20), top_values[::-1],
        color=plt.cm.Blues(np.linspace(0.4, 0.9, 20)),
    )
    ax.set_yticks(range(20))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Feature Importance (Gini)", fontsize=10)
    ax.set_title(
        "Random Forest — Top 20 Feature Importances",
        fontsize=12, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, top_values[::-1]):
        ax.text(
            val + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=7,
        )
    plt.tight_layout()
    path = FIGURES_DIR / "ml_04_rf_feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved → {path.name}")


def _plot_model_comparison(summary_df: pd.DataFrame) -> None:
    """Grouped bar chart comparing all three models across four metrics."""
    metrics = ["accuracy", "f1_macro", "roc_auc_macro", "cohen_kappa"]
    labels  = ["Accuracy", "F1 Macro", "ROC-AUC", "Cohen's κ"]
    models  = summary_df["model"].tolist()
    colors  = ["#2196F3", "#4CAF50", "#FF9800"]
    x       = np.arange(len(metrics))
    width   = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (model, color) in enumerate(zip(models, colors)):
        vals = [
            summary_df.loc[summary_df["model"] == model, m].values[0]
            for m in metrics
        ]
        bars = ax.bar(
            x + i * width, vals, width,
            label=model, color=color, alpha=0.85, edgecolor="white",
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold",
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_title(
        "Classical ML — Performance Comparison (Test Set)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = FIGURES_DIR / "ml_05_model_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved → {path.name}")


# ── Main training function ────────────────────────────────────────────────────

def train_and_evaluate_all() -> pd.DataFrame:
    """
    Train all three classical ML models, evaluate on the test set,
    save model artefacts, produce all figures, and return a summary DataFrame.

    Returns
    -------
    pd.DataFrame
        One row per model with key metrics (used by Stage 6 comparison).
    """
    set_all_seeds()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True,  exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    X_trainval, X_test, y_trainval, y_test, feature_names = _load_splits()

    models  = _get_models()
    results = {}
    summary = []

    # ── Train + evaluate each model ──────────────────────────────────────────
    for name, model in models.items():
        log.info(f"\n{'─'*50}\nTraining: {name}\n{'─'*50}")

        model.fit(X_trainval, y_trainval)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # Use the shared evaluate_model() utility (saves JSON + master CSV)
        metrics = evaluate_model(
            y_true     = y_test,
            y_pred     = y_pred,
            y_prob     = y_proba,
            model_name = name,
            save       = True,
        )

        results[name] = {
            "model":   model,
            "y_pred":  y_pred,
            "y_proba": y_proba,
            "metrics": metrics,
        }

        # Save model artefact
        safe_name = name.lower().replace(" ", "_")
        model_path = MODELS_DIR / f"{safe_name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        log.info(f"Model saved → {model_path.name}")

        # Collect summary row
        summary.append({
            "model":        name,
            "accuracy":     round(metrics["accuracy"],        4),
            "f1_macro":     round(metrics["f1_macro"],        4),
            "roc_auc_macro":round(metrics.get("roc_auc_macro") or 0, 4),
            "cohen_kappa":  round(metrics["cohen_kappa"],     4),
            "f1_confirmed":       round(metrics.get("f1_confirmed",       0), 4),
            "f1_false_positive":  round(metrics.get("f1_false_positive",  0), 4),
            "f1_candidate":       round(metrics.get("f1_candidate",       0), 4),
        })

    # ── Figures ───────────────────────────────────────────────────────────────
    log.info("\nGenerating figures …")
    _plot_confusion_matrices(results, y_test)
    _plot_per_class_f1(results)
    _plot_roc_curves(results, y_test)
    _plot_feature_importance(
        results["Random Forest"]["model"], feature_names
    )
    summary_df = pd.DataFrame(summary)
    _plot_model_comparison(summary_df)

    # ── Save Stage 3 summary CSV ──────────────────────────────────────────────
    summary_csv = METRICS_DIR / "classical_ml_results.csv"
    summary_df.to_csv(summary_csv, index=False)
    log.info(f"Summary saved → {summary_csv.name}")

    # ── Print results table ───────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("STAGE 3 — CLASSICAL ML RESULTS")
    log.info("=" * 60)
    log.info("\n" + summary_df[
        ["model", "accuracy", "f1_macro", "roc_auc_macro", "cohen_kappa"]
    ].to_string(index=False))

    best = summary_df.loc[summary_df["f1_macro"].idxmax()]
    log.info(
        f"\n  Best model (F1 Macro): {best['model']} "
        f"(F1={best['f1_macro']:.4f}, AUC={best['roc_auc_macro']:.4f})"
    )
    log.info("=" * 60)

    return summary_df


# ── Script entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_and_evaluate_all()