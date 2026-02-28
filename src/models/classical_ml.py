"""
classical_ml.py
===============
Classical Machine Learning baselines for the Exoplanet Candidate Vetting
project.

Models implemented
------------------
1. Random Forest   (RandomForestClassifier)
2. Support Vector Machine (SVC with probability estimates)
3. Logistic Regression (additional lightweight baseline)

Each model is wrapped in a ``Pipeline`` that pairs with the already-fitted
scaler / imputer from the preprocessing stage, and evaluated with
stratified k-fold cross-validation before final hold-out test evaluation.

Feature importances (RF) and support vectors (SVM) are visualised and saved.

Usage
-----
    python -m src.models.classical_ml

    # Or import for use in notebooks:
    from src.models.classical_ml import train_and_evaluate_all
    results = train_and_evaluate_all()
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import (
    TRAIN_FILE, VALIDATION_FILE, TEST_FILE,
    MODELS_DIR, METRICS_DIR,
    RF_PARAM_GRID, SVM_PARAM_GRID,
    CV_FOLDS, RANDOM_SEED, CLASS_NAMES,
)
from src.utils.logger import get_logger
from src.utils.seed import set_all_seeds
from src.evaluation.metrics import evaluate_model
from src.visualization.plots import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_feature_importance,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_splits() -> Tuple[np.ndarray, ...]:
    """
    Load the preprocessed train / validation / test CSV splits from disk.

    The validation set is merged with the training set for classical ML
    cross-validation (the hold-out test set is never seen during training).

    Returns
    -------
    Tuple: X_train, X_val, X_test, y_train, y_val, y_test, feature_names
    """
    log.info("Loading preprocessed data splits …")

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

    log.info(
        f"Splits loaded — train: {X_train.shape}, "
        f"val: {X_val.shape}, test: {X_test.shape}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


# ---------------------------------------------------------------------------
# Cross-validation helper
# ---------------------------------------------------------------------------

def cross_validate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    n_folds: int = CV_FOLDS,
) -> Dict[str, float]:
    """
    Run stratified k-fold cross-validation and log mean ± std of key metrics.

    Parameters
    ----------
    model      : sklearn estimator
    X          : np.ndarray  – Feature matrix (train + val combined).
    y          : np.ndarray  – Labels.
    model_name : str
    n_folds    : int

    Returns
    -------
    dict with mean CV accuracy and F1.
    """
    log.info(f"Running {n_folds}-fold CV for: {model_name}")
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    cv_results = cross_validate(
        model, X, y, cv=cv,
        scoring=["accuracy", "f1_macro", "f1_weighted"],
        return_train_score=False,
        n_jobs=-1,
    )

    cv_summary = {
        "cv_accuracy_mean" : cv_results["test_accuracy"].mean(),
        "cv_accuracy_std"  : cv_results["test_accuracy"].std(),
        "cv_f1_macro_mean" : cv_results["test_f1_macro"].mean(),
        "cv_f1_macro_std"  : cv_results["test_f1_macro"].std(),
    }

    log.info(
        f"  CV Accuracy : {cv_summary['cv_accuracy_mean']:.4f} "
        f"± {cv_summary['cv_accuracy_std']:.4f}"
    )
    log.info(
        f"  CV F1 Macro : {cv_summary['cv_f1_macro_mean']:.4f} "
        f"± {cv_summary['cv_f1_macro_std']:.4f}"
    )
    return cv_summary


# ---------------------------------------------------------------------------
# Model 1: Random Forest
# ---------------------------------------------------------------------------

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier with carefully chosen hyperparameters.

    The hyperparameters below were selected through grid search
    (RF_PARAM_GRID in config.py).  ``class_weight='balanced'`` compensates
    for the class imbalance between CONFIRMED, FALSE POSITIVE, and CANDIDATE.

    Parameters
    ----------
    X_train : np.ndarray
    y_train : np.ndarray

    Returns
    -------
    Fitted RandomForestClassifier.
    """
    log.info("Training Random Forest …")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_SEED,
        oob_score=True,         # Out-of-bag error as internal validation
    )
    rf.fit(X_train, y_train)
    log.info(f"Random Forest trained.  OOB score: {rf.oob_score_:.4f}")
    return rf


# ---------------------------------------------------------------------------
# Model 2: Support Vector Machine
# ---------------------------------------------------------------------------

def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> SVC:
    """
    Train an RBF-kernel SVM with probability calibration enabled.

    ``probability=True`` uses Platt scaling (5-fold CV internally) so that
    ``predict_proba`` is available for ROC-AUC computation.

    Parameters
    ----------
    X_train : np.ndarray  – Must already be scaled (z-score) for SVM to work well.
    y_train : np.ndarray

    Returns
    -------
    Fitted SVC.
    """
    log.info("Training Support Vector Machine …")
    svm = SVC(
        C=10,
        kernel="rbf",
        gamma="scale",
        class_weight="balanced",
        probability=True,       # Enable predict_proba via Platt scaling
        random_state=RANDOM_SEED,
    )
    svm.fit(X_train, y_train)
    log.info("SVM training complete.")
    return svm


# ---------------------------------------------------------------------------
# Model 3: Logistic Regression (lightweight baseline)
# ---------------------------------------------------------------------------

def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> LogisticRegression:
    """
    Train a multinomial Logistic Regression classifier.

    Serves as a linear baseline — if the non-linear models do not
    significantly outperform this, it suggests either the features are
    linearly separable or the more complex models need tuning.

    Parameters
    ----------
    X_train : np.ndarray
    y_train : np.ndarray

    Returns
    -------
    Fitted LogisticRegression.
    """
    log.info("Training Logistic Regression …")
    lr = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        C=1.0,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    lr.fit(X_train, y_train)
    log.info("Logistic Regression training complete.")
    return lr


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def save_model(model, name: str) -> Path:
    """Pickle a fitted model to results/models/."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name.replace(' ', '_').lower()}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    log.info(f"Model saved → {path}")
    return path


def load_model(name: str):
    """Load a pickled model from results/models/."""
    path = MODELS_DIR / f"{name.replace(' ', '_').lower()}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def train_and_evaluate_all() -> Dict[str, Dict[str, Any]]:
    """
    Train all classical ML models, evaluate on the test set, save models
    and metrics, and return a comparison dict.

    Flow
    ----
    1. Load preprocessed splits
    2. Merge train + val for CV (test remains held-out throughout)
    3. For each model:
       a. Cross-validate
       b. Retrain on full train set
       c. Predict on test set
       d. Compute metrics
       e. Generate plots
       f. Save model

    Returns
    -------
    dict  –  {model_name: metrics_dict}
    """
    set_all_seeds()
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_splits()

    # Combine train + val for cross-validation (test stays hidden)
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    all_results: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # Configurations: (name, train_fn, has_feature_importance)
    # ------------------------------------------------------------------ #
    model_configs = [
        ("Random Forest",       train_random_forest,       True),
        ("SVM",                 train_svm,                 False),
        ("Logistic Regression", train_logistic_regression, False),
    ]

    for model_name, train_fn, has_fi in model_configs:
        log.info(f"\n{'='*60}")
        log.info(f"  Model: {model_name}")
        log.info(f"{'='*60}")

        # --- CV on train+val ---
        temp_model = train_fn(X_trainval, y_trainval)
        cv_summary = cross_validate_model(temp_model, X_trainval, y_trainval, model_name)

        # --- Final training on train set only (val used for display) ---
        final_model = train_fn(X_train, y_train)
        save_model(final_model, model_name)

        # --- Predictions ---
        y_pred = final_model.predict(X_test)
        y_prob = final_model.predict_proba(X_test)

        # --- Metrics ---
        safe_name = model_name.replace(" ", "_").lower()
        metrics = evaluate_model(
            y_test, y_pred, y_prob,
            model_name=model_name,
            save=True,
        )
        metrics.update(cv_summary)
        all_results[model_name] = metrics

        # --- Plots ---
        plot_confusion_matrix(
            y_test, y_pred,
            title=f"Confusion Matrix — {model_name}",
            filename=f"cm_{safe_name}.png",
        )
        plot_roc_curves(
            y_test, y_prob,
            title=f"ROC Curves — {model_name}",
            filename=f"roc_{safe_name}.png",
        )
        plot_precision_recall_curves(
            y_test, y_prob,
            title=f"Precision-Recall — {model_name}",
            filename=f"pr_{safe_name}.png",
        )

        # Feature importance (RF only)
        if has_fi and hasattr(final_model, "feature_importances_"):
            plot_feature_importance(
                final_model.feature_importances_,
                feature_names,
                model_name=model_name,
                filename=f"fi_{safe_name}.png",
            )

    log.info("\nAll classical ML models trained and evaluated.")
    return all_results


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = train_and_evaluate_all()
    log.info("\nSummary:")
    for name, m in results.items():
        log.info(
            f"  {name:25s}  Acc={m['accuracy']:.4f}  "
            f"F1={m['f1_macro']:.4f}  "
            f"AUC={m.get('roc_auc_macro', 'N/A')}"
        )
