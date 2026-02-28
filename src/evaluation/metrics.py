"""
metrics.py
==========
Evaluation utilities for the Exoplanet Candidate Vetting project.

Provides a single ``evaluate_model()`` function that computes a
comprehensive set of classification metrics and saves them to disk,
plus helper functions used by the comparison notebook.

Metrics computed
----------------
- Accuracy
- Precision, Recall, F1 (macro, weighted, per-class)
- ROC-AUC (macro OvR, weighted OvR)
- Cohen's Kappa
- Matthews Correlation Coefficient
- Classification report (string)

Usage
-----
    from src.evaluation.metrics import evaluate_model

    metrics = evaluate_model(
        y_true, y_pred, y_prob,
        model_name="Genesis CNN",
        save=True,
    )
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    cohen_kappa_score,
    matthews_corrcoef,
    classification_report,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import CLASS_NAMES, METRICS_DIR, METRICS_AVERAGE
from src.utils.logger import get_logger

log = get_logger(__name__)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    model_name: str = "Model",
    save: bool = True,
) -> Dict[str, Any]:
    """
    Compute and optionally save a comprehensive set of classification metrics.

    Parameters
    ----------
    y_true     : np.ndarray  – Ground-truth integer labels (shape N,).
    y_pred     : np.ndarray  – Predicted integer labels (shape N,).
    y_prob     : np.ndarray or None
                 Predicted probabilities (shape N × C). Required for ROC-AUC.
    model_name : str         – Used for display and file naming.
    save       : bool        – If True, write metrics to JSON and CSV.

    Returns
    -------
    dict
        Dictionary of all computed metrics.
    """
    log.info(f"Evaluating: {model_name}")

    # ------------------------------------------------------------------ #
    # Core classification metrics
    # ------------------------------------------------------------------ #
    metrics = {
        "model"           : model_name,
        "accuracy"        : float(accuracy_score(y_true, y_pred)),
        "precision_macro" : float(precision_score(y_true, y_pred, average="macro",    zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_macro"    : float(recall_score(y_true, y_pred, average="macro",       zero_division=0)),
        "recall_weighted" : float(recall_score(y_true, y_pred, average="weighted",    zero_division=0)),
        "f1_macro"        : float(f1_score(y_true, y_pred, average="macro",           zero_division=0)),
        "f1_weighted"     : float(f1_score(y_true, y_pred, average="weighted",        zero_division=0)),
        "cohen_kappa"     : float(cohen_kappa_score(y_true, y_pred)),
        "mcc"             : float(matthews_corrcoef(y_true, y_pred)),
    }

    # Per-class F1 scores
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, cls_name in enumerate(CLASS_NAMES):
        safe_name = cls_name.replace(" ", "_").lower()
        if i < len(f1_per_class):
            metrics[f"f1_{safe_name}"] = float(f1_per_class[i])

    # ------------------------------------------------------------------ #
    # ROC-AUC (requires probability estimates)
    # ------------------------------------------------------------------ #
    if y_prob is not None:
        try:
            metrics["roc_auc_macro"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr",
                              average="macro", labels=list(range(len(CLASS_NAMES))))
            )
            metrics["roc_auc_weighted"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr",
                              average="weighted", labels=list(range(len(CLASS_NAMES))))
            )
        except Exception as exc:
            log.warning(f"ROC-AUC computation failed: {exc}")
            metrics["roc_auc_macro"]    = None
            metrics["roc_auc_weighted"] = None
    else:
        metrics["roc_auc_macro"]    = None
        metrics["roc_auc_weighted"] = None

    # ------------------------------------------------------------------ #
    # Classification report (human-readable string)
    # ------------------------------------------------------------------ #
    target_names = CLASS_NAMES[: len(np.unique(y_true))]
    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    metrics["classification_report"] = report

    # ------------------------------------------------------------------ #
    # Log summary
    # ------------------------------------------------------------------ #
    log.info(f"  Accuracy   : {metrics['accuracy']:.4f}")
    log.info(f"  F1 (macro) : {metrics['f1_macro']:.4f}")
    if metrics["roc_auc_macro"] is not None:
        log.info(f"  ROC-AUC    : {metrics['roc_auc_macro']:.4f}")
    log.info(f"  Kappa      : {metrics['cohen_kappa']:.4f}")
    log.info(f"\n{report}")

    # ------------------------------------------------------------------ #
    # Save metrics to disk
    # ------------------------------------------------------------------ #
    if save:
        _save_metrics(metrics, model_name)

    return metrics


def _save_metrics(metrics: dict, model_name: str) -> None:
    """
    Write metrics dict to JSON and append a summary row to the
    master metrics CSV (``results/metrics/all_models_metrics.csv``).

    Parameters
    ----------
    metrics    : dict
    model_name : str
    """
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace(" ", "_").lower()

    # --- JSON (full detail) ---
    json_path = METRICS_DIR / f"{safe_name}_metrics.json"
    # Classification report is not JSON-serialisable directly; keep as string
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {k: v for k, v in metrics.items()},
            f, indent=2, default=str,
        )
    log.info(f"Metrics saved → {json_path}")

    # --- Master CSV (summary row, excludes long string columns) ---
    scalar_keys = [
        k for k, v in metrics.items()
        if isinstance(v, (int, float, type(None))) and k != "model"
    ]
    row = {"model": model_name, **{k: metrics[k] for k in scalar_keys}}
    master_csv = METRICS_DIR / "all_models_metrics.csv"

    if master_csv.exists():
        master_df = pd.read_csv(master_csv)
        # Replace existing row for this model if present
        master_df = master_df[master_df["model"] != model_name]
        master_df = pd.concat([master_df, pd.DataFrame([row])], ignore_index=True)
    else:
        master_df = pd.DataFrame([row])

    master_df.to_csv(master_csv, index=False)
    log.info(f"Master metrics CSV updated → {master_csv}")


def load_all_metrics() -> pd.DataFrame:
    """
    Load the master metrics CSV containing summary rows for all evaluated
    models.  Used by the comparison notebook and plots module.

    Returns
    -------
    pd.DataFrame or empty DataFrame if file does not exist.
    """
    master_csv = METRICS_DIR / "all_models_metrics.csv"
    if master_csv.exists():
        return pd.read_csv(master_csv)
    log.warning("No master metrics CSV found. Run model evaluations first.")
    return pd.DataFrame()
