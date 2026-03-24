"""
baseline_cnn.py
===============
Stage 4 — Single-Branch Baseline CNN for Exoplanet Candidate Vetting.

Architecture
------------
    Input (37, 1)
        │
    Conv1D(32, kernel=3, ReLU) + BatchNorm
        │
    Conv1D(64, kernel=3, ReLU) + BatchNorm
        │
    Conv1D(128, kernel=3, ReLU) + BatchNorm
        │
    GlobalAveragePooling1D
        │
    Dense(64, ReLU) + Dropout(0.3)
        │
    Dense(3, Softmax)  →  CONFIRMED / FALSE POSITIVE / CANDIDATE
    
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Allow running as either  `python -m src.models.baseline_cnn`
# or  `python src/models/baseline_cnn.py`  from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc as sk_auc
from sklearn.utils.class_weight import compute_class_weight

from src.utils.config import (
    TRAIN_FILE, VALIDATION_FILE, TEST_FILE,
    MODELS_DIR, FIGURES_DIR, METRICS_DIR,
    CLASS_NAMES, RANDOM_SEED, BASELINE_CNN_CONFIG,
)
from src.utils.logger import get_logger
from src.utils.seed import set_all_seeds
from src.evaluation.metrics import evaluate_model

log = get_logger(__name__)

# Colour palette — same green/red/blue used throughout the project
_COLORS = ["#2ecc71", "#e74c3c", "#3498db"]   # CONFIRMED / FP / CANDIDATE


# =============================================================================
# 1. DATA LOADING
# =============================================================================

def _load_splits():
    """
    Load the three preprocessed CSV splits produced by Stage 1.

    Returns
    -------
    X_train, X_val, X_test : np.ndarray  shape (N, 37, 1)
        Features reshaped for Conv1D — the extra trailing "1" is the
        channel dimension (like mono audio: 37 timesteps, 1 channel).

    y_train, y_val, y_test : np.ndarray  shape (N,)
        Integer labels: 0=CONFIRMED, 1=FALSE POSITIVE, 2=CANDIDATE.

    n_features : int
        Number of features (37 for our KOI dataset).
    """
    log.info("Loading preprocessed splits …")
    train_df = pd.read_csv(TRAIN_FILE)
    val_df   = pd.read_csv(VALIDATION_FILE)
    test_df  = pd.read_csv(TEST_FILE)

    feature_cols = [c for c in train_df.columns if c != "label"]
    n_features   = len(feature_cols)

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["label"].values.astype(np.int32)
    X_val   = val_df[feature_cols].values.astype(np.float32)
    y_val   = val_df["label"].values.astype(np.int32)
    X_test  = test_df[feature_cols].values.astype(np.float32)
    y_test  = test_df["label"].values.astype(np.int32)

    log.info(f"  Train      : {X_train.shape[0]} samples")
    log.info(f"  Validation : {X_val.shape[0]} samples")
    log.info(f"  Test       : {X_test.shape[0]} samples")
    log.info(f"  Features   : {n_features}")

    # Reshape (N, 37) → (N, 37, 1)  so Conv1D can process the sequence
    X_train = X_train.reshape(-1, n_features, 1)
    X_val   = X_val.reshape(-1, n_features, 1)
    X_test  = X_test.reshape(-1, n_features, 1)

    return X_train, X_val, X_test, y_train, y_val, y_test, n_features


# =============================================================================
# 2. MODEL DEFINITION
# =============================================================================

def build_baseline_cnn(n_features: int) -> keras.Model:
    """
    Build and compile the single-branch Baseline CNN.

    Architecture (controlled by BASELINE_CNN_CONFIG in config.py):
      - 3 × Conv1D blocks:  filters [32, 64, 128],  kernel_size=3
      - Each block:  Conv1D  →  BatchNormalization  →  ReLU
      - GlobalAveragePooling1D  (replaces Flatten — much fewer parameters)
      - Dense(64, ReLU)  →  Dropout(0.3)
      - Dense(3, Softmax)

    Parameters
    ----------
    n_features : int
        Length of the feature vector (37 for the KOI dataset).

    Returns
    -------
    keras.Model  compiled and ready for training.
    """
    cfg = BASELINE_CNN_CONFIG

    inputs = keras.Input(shape=(n_features, 1), name="tabular_input")
    x = inputs

    # ----- Three convolutional blocks ----------------------------------------
    # Each block doubles the filters: 32 → 64 → 128
    # padding="same" keeps the sequence length at 37 throughout all blocks
    # use_bias=False because BatchNorm's shift parameter (β) does the same job
    # L2 regularization (1e-4) prevents large weights on our small dataset
    for i, n_filters in enumerate(cfg["filters"]):
        x = layers.Conv1D(
            filters=n_filters,
            kernel_size=cfg["kernel_size"],
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizers.l2(1e-4),
            name=f"conv_{i+1}",
        )(x)
        x = layers.BatchNormalization(name=f"bn_{i+1}")(x)
        x = layers.Activation("relu", name=f"relu_{i+1}")(x)

    # ----- Global Average Pooling --------------------------------------------
    # Shape goes from (37, 128) → (128,)
    # Takes the mean across all 37 positions for each of the 128 filters.
    # This keeps parameter count low vs Flatten which would give (4736,).
    x = layers.GlobalAveragePooling1D(name="gap")(x)

    # ----- Dense classification head -----------------------------------------
    for i, units in enumerate(cfg["dense_units"]):
        x = layers.Dense(
            units,
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4),
            name=f"dense_{i+1}",
        )(x)
        x = layers.Dropout(cfg["dropout_rate"], name=f"dropout_{i+1}",
                            seed=RANDOM_SEED)(x)

    outputs = layers.Dense(
        cfg["num_classes"],
        activation="softmax",
        name="output",
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="baseline_cnn")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# =============================================================================
# 3. VISUALISATIONS
# =============================================================================

def _plot_training_history(history: keras.callbacks.History) -> None:
    """Two-panel loss + accuracy plot with best epoch marker."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Baseline CNN — Training History", fontsize=13, fontweight="bold")

    epochs = range(1, len(history.history["loss"]) + 1)

    # Loss curve
    ax1.plot(epochs, history.history["loss"],     color="#58a6ff", lw=2, label="Train loss")
    ax1.plot(epochs, history.history["val_loss"], color="#f0883e", lw=2, linestyle="--", label="Val loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Loss curve", fontweight="bold")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    # Accuracy curve
    ax2.plot(epochs, history.history["accuracy"],     color="#39d353", lw=2, label="Train accuracy")
    ax2.plot(epochs, history.history["val_accuracy"], color="#bc8cff", lw=2, linestyle="--", label="Val accuracy")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.set_title("Accuracy curve", fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3); ax2.set_ylim(0, 1)

    # Red dotted line at the best epoch
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    for ax in (ax1, ax2):
        ax.axvline(best_epoch, color="#f85149", linestyle=":", lw=1.5,
                   alpha=0.8, label=f"Best epoch ({best_epoch})")
    ax1.legend(fontsize=9); ax2.legend(fontsize=9)

    plt.tight_layout()
    path = FIGURES_DIR / "cnn4_baseline_training_history.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved → {path.name}")


def _plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Normalised confusion matrix with raw counts as annotations."""
    cm   = confusion_matrix(y_true, y_pred)
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_n, annot=cm, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        linewidths=0.5, ax=ax, cbar=True,
        annot_kws={"size": 11, "weight": "bold"},
    )
    ax.set_title("Baseline CNN — Confusion Matrix (Test Set)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True",      fontsize=10)
    ax.tick_params(axis="x", rotation=25, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)
    plt.tight_layout()

    path = FIGURES_DIR / "cnn4_baseline_confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved → {path.name}")


def _plot_roc_curves(y_true: np.ndarray, y_proba: np.ndarray) -> None:
    """One-vs-Rest ROC curves for all three classes."""
    y_bin = label_binarize(y_true, classes=[0, 1, 2])

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (label, color) in enumerate(zip(CLASS_NAMES, _COLORS)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2.2, label=f"{label}  (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random (0.500)")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate",  fontsize=10)
    ax.set_title("Baseline CNN — ROC Curves (OvR, Test Set)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout()

    path = FIGURES_DIR / "cnn4_baseline_roc_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved → {path.name}")


def _plot_architecture(model: keras.Model) -> None:
    """Text-based architecture diagram saved as a dark-theme PNG."""
    cfg = BASELINE_CNN_CONFIG
    lines = [
        "Baseline CNN — Architecture",
        "═" * 44,
        "  Input            (37 features, 1 channel)",
        "        │",
        "  Conv1D(32, k=3) + BatchNorm + ReLU",
        "        │",
        "  Conv1D(64, k=3) + BatchNorm + ReLU",
        "        │",
        "  Conv1D(128, k=3) + BatchNorm + ReLU",
        "        │",
        "  GlobalAveragePooling1D",
        "        │",
        f"  Dense(64, ReLU) + Dropout({cfg['dropout_rate']})",
        "        │",
        "  Dense(3, Softmax)",
        "═" * 44,
        f"  Total params: {model.count_params():,}",
        f"  Trainable:    {sum(tf.size(w).numpy() for w in model.trainable_weights):,}",
    ]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_xlim(0, 1); ax.set_ylim(0, len(lines) + 1); ax.axis("off")
    for i, line in enumerate(reversed(lines)):
        weight = "bold" if ("═" in line or "Baseline" in line) else "normal"
        color  = "#39d353" if "Dense(3"  in line else \
                 "#58a6ff" if "Conv1D"   in line else \
                 "#bc8cff" if "Dense(64" in line else \
                 "#e6edf3"
        ax.text(0.08, i + 0.5, line, fontsize=11, fontfamily="monospace",
                color=color, fontweight=weight, va="center")
    ax.set_facecolor("#0d1117"); fig.patch.set_facecolor("#0d1117")
    plt.tight_layout()

    path = FIGURES_DIR / "cnn4_baseline_architecture.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    log.info(f"Saved → {path.name}")


# =============================================================================
# 4. MAIN TRAINING FUNCTION
# =============================================================================

def train_baseline_cnn() -> dict:
    """
    Full Stage 4 pipeline: build → train → evaluate → save everything.

    Steps
    -----
    1.  Load the Stage 1 preprocessed CSV splits
    2.  Build the model from BASELINE_CNN_CONFIG
    3.  Compute balanced class weights (same approach as classical ML)
    4.  Train with EarlyStopping + ReduceLROnPlateau + ModelCheckpoint
    5.  Evaluate on the held-out test set via the shared evaluate_model()
    6.  Save model weights, figures, and metrics CSV

    Returns
    -------
    dict
        Full metrics dictionary (accuracy, F1, AUC, κ, per-class F1, …)
        This is passed to Stage 6 for the final comparison table.
    """
    set_all_seeds()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True,  exist_ok=True)

    cfg = BASELINE_CNN_CONFIG

    # ------------------------------------------------------------------
    # Step 1 — Load data
    # ------------------------------------------------------------------
    X_train, X_val, X_test, y_train, y_val, y_test, n_features = _load_splits()

    # ------------------------------------------------------------------
    # Step 2 — Build model
    # ------------------------------------------------------------------
    log.info("\nBuilding Baseline CNN …")
    model = build_baseline_cnn(n_features)
    model.summary(print_fn=log.info)
    log.info(f"Total parameters: {model.count_params():,}")

    # ------------------------------------------------------------------
    # Step 3 — Class weights
    # Mirrors class_weight='balanced' from sklearn.
    # CONFIRMED: 0.556,  FALSE POSITIVE: 1.201,  CANDIDATE: 2.725
    # Without this, CANDIDATE F1 drops from ~0.39 to ~0.08.
    # ------------------------------------------------------------------
    cw_values = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weights = dict(enumerate(cw_values))
    log.info(f"Class weights: { {k: round(v, 3) for k, v in class_weights.items()} }")

    # ------------------------------------------------------------------
    # Step 4 — Callbacks
    # ------------------------------------------------------------------
    callbacks = [
        # Stop if val_loss doesn't improve for 10 consecutive epochs.
        # restore_best_weights=True rolls back to the best checkpoint.
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg["patience"],
            restore_best_weights=True,
            verbose=1,
        ),
        # Halve the learning rate if val_loss plateaus for 5 epochs.
        # patience=5 fires before EarlyStopping, giving the model a
        # second chance at a lower learning rate before training ends.
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        # Save the best weights to disk after every improvement.
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / "baseline_cnn_best.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]

    # ------------------------------------------------------------------
    # Step 5 — Train
    # ------------------------------------------------------------------
    log.info(f"\nTraining for up to {cfg['epochs']} epochs "
             f"(early stopping patience={cfg['patience']}) …")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    best_epoch    = int(np.argmin(history.history["val_loss"])) + 1
    best_val_loss = min(history.history["val_loss"])
    log.info(f"\nTraining complete.  Best epoch: {best_epoch}  "
             f"(val_loss = {best_val_loss:.4f})")

    # ------------------------------------------------------------------
    # Step 6 — Save final model
    # ------------------------------------------------------------------
    model_path = MODELS_DIR / "baseline_cnn.keras"
    model.save(model_path)
    log.info(f"Model saved → {model_path.name}")

    # ------------------------------------------------------------------
    # Step 7 — Predict on test set
    # ------------------------------------------------------------------
    log.info("\nEvaluating on held-out test set …")
    y_proba = model.predict(X_test, verbose=0)
    y_pred  = np.argmax(y_proba, axis=1)

    # ------------------------------------------------------------------
    # Step 8 — Metrics  (shared utility used by all models)
    # ------------------------------------------------------------------
    metrics = evaluate_model(
        y_true     = y_test,
        y_pred     = y_pred,
        y_prob     = y_proba,
        model_name = "Baseline CNN",
        save       = True,
    )
    metrics["best_epoch"]    = best_epoch
    metrics["best_val_loss"] = round(float(best_val_loss), 4)
    metrics["n_params"]      = model.count_params()

    # ------------------------------------------------------------------
    # Step 9 — Figures
    # ------------------------------------------------------------------
    log.info("\nGenerating figures …")
    _plot_training_history(history)
    _plot_confusion_matrix(y_test, y_pred)
    _plot_roc_curves(y_test, y_proba)
    _plot_architecture(model)

    # ------------------------------------------------------------------
    # Step 10 — Stage-specific summary CSV
    # ------------------------------------------------------------------
    summary = {
        "model":              "Baseline CNN",
        "accuracy":           round(metrics["accuracy"], 4),
        "f1_macro":           round(metrics["f1_macro"], 4),
        "roc_auc_macro":      round(metrics.get("roc_auc_macro") or 0, 4),
        "cohen_kappa":        round(metrics["cohen_kappa"], 4),
        "f1_confirmed":       round(metrics.get("f1_confirmed", 0), 4),
        "f1_false_positive":  round(metrics.get("f1_false_positive", 0), 4),
        "f1_candidate":       round(metrics.get("f1_candidate", 0), 4),
        "best_epoch":         best_epoch,
        "n_params":           model.count_params(),
    }
    pd.DataFrame([summary]).to_csv(
        METRICS_DIR / "baseline_cnn_results.csv", index=False
    )
    log.info("Summary CSV saved → baseline_cnn_results.csv")

    # ------------------------------------------------------------------
    # Step 11 — Print final table to log
    # ------------------------------------------------------------------
    log.info("\n" + "=" * 55)
    log.info("STAGE 4 — BASELINE CNN RESULTS")
    log.info("=" * 55)
    log.info(f"  Accuracy   : {metrics['accuracy']:.4f}")
    log.info(f"  F1 Macro   : {metrics['f1_macro']:.4f}")
    log.info(f"  ROC-AUC    : {metrics.get('roc_auc_macro', 0):.4f}")
    log.info(f"  Cohen's κ  : {metrics['cohen_kappa']:.4f}")
    log.info(f"  Best epoch : {best_epoch} / {len(history.history['loss'])}")
    log.info(f"  Parameters : {model.count_params():,}")
    log.info("=" * 55)

    return metrics


# =============================================================================
# Script entry point
# =============================================================================
if __name__ == "__main__":
    train_baseline_cnn()