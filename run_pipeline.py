"""
run_pipeline.py
===============
Master pipeline runner for the Exoplanet Candidate Vetting project.

Stages
------
  Stage 1  — Data Preprocessing         (tabular cleaning, imputation, scaling, splits)
  Stage 2  — EDA & Visualisation        (class distribution, correlation, distributions)

Usage
-----
  python run_pipeline.py                     # Run ALL stages
  python run_pipeline.py --stage 1 2 3       # Run specific stages
Examples
--------
  # Full dissertation pipeline
  python run_pipeline.py

  # Fast test (no CNNs)
  python run_pipeline.py --stage 1 2 3 6 --skip-cnn
"""

import argparse
import sys
import time
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.logger import get_logger
from src.utils.seed import set_all_seeds

log = get_logger("run_pipeline")


# ---------------------------------------------------------------------------
# Stage executor helper
# ---------------------------------------------------------------------------

def run_stage(stage_num: int, stage_name: str, fn, *args, **kwargs):
    """Execute one pipeline stage with timing and clean error reporting."""
    log.info(f"\n{'#' * 62}")
    log.info(f"#  STAGE {stage_num}: {stage_name}")
    log.info(f"{'#' * 62}")
    t0 = time.time()
    try:
        result  = fn(*args, **kwargs)
        elapsed = time.time() - t0
        log.info(f"Stage {stage_num} completed successfully in {elapsed:.1f}s")
        return result
    except Exception as exc:
        log.error(f"Stage {stage_num} FAILED: {exc}", exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Individual stage functions
# ---------------------------------------------------------------------------

def stage1_preprocessing():
    """Load, clean, impute, scale and split the KOI tabular data."""
    from src.data.preprocess import run_preprocessing
    return run_preprocessing()


def stage2_eda(processed_file):
    """Generate exploratory data analysis figures."""
    import pandas as pd
    from src.utils.config import (
        TRANSIT_FEATURES, STELLAR_FEATURES, ALL_MODEL_FEATURES,
        TARGET_COL,
    )
    from src.visualization.plots import (
        plot_class_distribution,
        plot_missing_values,
        plot_correlation_heatmap,
        plot_feature_distributions,
    )

    df = pd.read_csv(processed_file)

    plot_class_distribution(df, save=True)
    log.info("  Class distribution plot saved.")

    plot_missing_values(df, save=True)
    log.info("  Missing values plot saved.")

    available = [f for f in ALL_MODEL_FEATURES if f in df.columns]
    plot_correlation_heatmap(df, available, save=True)
    log.info("  Correlation heatmap saved.")

    plot_feature_distributions(df, TRANSIT_FEATURES, save=True)
    log.info("  Feature distributions plot saved.")

    log.info("EDA complete — 4 figures saved to results/figures/")


def stage3_classical_ml():
    """Train and evaluate Random Forest, SVM and Logistic Regression."""
    from src.models.classical_ml import train_and_evaluate_all
    return train_and_evaluate_all()