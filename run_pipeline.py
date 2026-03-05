"""
run_pipeline.py
===============
Master pipeline runner for the Exoplanet Candidate Vetting project.

Stages
------
  Stage 1  — Data Preprocessing
  Stage 2  — EDA & Visualisation (8-section comprehensive EDA)
  Stage 3  — Classical ML Models (RF, SVM, LR)
  Stage 4  — Baseline CNN
  Stage 5  — Genesis CNN (tabular)
  Stage 6  — Model Comparison

Usage
-----
  python run_pipeline.py                  # All stages
  python run_pipeline.py --stage 1 2      # Preprocessing + EDA only
  python run_pipeline.py --skip-cnn       # Skip CNN stages

Author : MSc Data Science Dissertation
Dataset: NASA Kepler Objects of Interest (KOI) Q1-Q17 DR25
"""

from __future__ import annotations
import argparse, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.logger import get_logger
from src.utils.seed   import set_all_seeds
from src.utils.config import PROCESSED_FILE

log = get_logger("run_pipeline")


def run_stage(stage_num, stage_name, fn, *args, **kwargs):
    """Execute one pipeline stage with timing and error reporting."""
    log.info(f"\n{'#'*62}\n#  STAGE {stage_num}: {stage_name}\n{'#'*62}")
    t0 = time.time()
    try:
        result = fn(*args, **kwargs)
        log.info(f"Stage {stage_num} completed in {time.time()-t0:.1f}s")
        return result
    except Exception as exc:
        log.error(f"Stage {stage_num} FAILED: {exc}", exc_info=True)
        raise


def stage1_preprocessing():
    from src.data.preprocess import run_preprocessing
    return run_preprocessing()


def stage2_eda():
    """
    Full 8-section EDA using src/data/eda.py.
    Generates 10 dissertation-quality figures and 3 CSV summaries.
    """
    from src.eda.run_eda import run_eda
    run_eda()


def stage3_classical_ml():
    try:
        from src.models.classical_ml import train_and_evaluate_all
        return train_and_evaluate_all()
    except ImportError:
        log.warning("classical_ml.py not yet implemented — skipping Stage 3.")


def stage4_baseline_cnn():
    try:
        from src.models.baseline_cnn import train_baseline_cnn
        return train_baseline_cnn()
    except ImportError:
        log.warning("baseline_cnn.py not yet implemented — skipping Stage 4.")


def stage5_genesis_cnn():
    try:
        from src.models.genesis_cnn import train_genesis_cnn
        return train_genesis_cnn()
    except ImportError:
        log.warning("genesis_cnn.py not yet implemented — skipping Stage 5.")


def stage6_comparison():
    try:
        from src.evaluation.compare import run_comparison
        return run_comparison()
    except ImportError:
        log.warning("compare.py not yet implemented — skipping Stage 6.")


def parse_args():
    p = argparse.ArgumentParser(description="Exoplanet Vetting Pipeline")
    p.add_argument("--stage", "-s", type=int, nargs="+",
                   choices=[1,2,3,4,5,6], default=None, metavar="N",
                   help="Stage(s) to run: 1=Preprocess 2=EDA 3=ClassML 4=BaseCNN 5=GenesisCNN 6=Compare")
    p.add_argument("--skip-cnn", action="store_true",
                   help="Skip CNN stages (4 and 5) for fast runs.")
    return p.parse_args()


def main():
    set_all_seeds()
    args = parse_args()
    t_start = time.time()

    log.info("\n╔══════════════════════════════════════════════════════╗")
    log.info("║  EXOPLANET CANDIDATE VETTING — PIPELINE RUNNER      ║")
    log.info("║  MSc Data Science Dissertation                       ║")
    log.info("╚══════════════════════════════════════════════════════╝\n")

    stages = sorted(set(args.stage)) if args.stage else [1,2,3,4,5,6]
    if args.skip_cnn:
        stages = [s for s in stages if s not in (4,5)]
        log.info("--skip-cnn active: stages 4 & 5 skipped.")

    log.info(f"Running stages: {stages}\n")

    if 1 in stages:
        run_stage(1, "Data Preprocessing", stage1_preprocessing)

    if 2 in stages:
        if not PROCESSED_FILE.exists():
            log.error("koi_processed.csv not found. Run Stage 1 first.")
            sys.exit(1)
        run_stage(2, "Exploratory Data Analysis", stage2_eda)

    if 3 in stages:
        run_stage(3, "Classical ML Models", stage3_classical_ml)

    if 4 in stages:
        run_stage(4, "Baseline CNN", stage4_baseline_cnn)

    if 5 in stages:
        run_stage(5, "Genesis CNN (Tabular)", stage5_genesis_cnn)

    if 6 in stages:
        run_stage(6, "Model Comparison", stage6_comparison)

    elapsed = time.time() - t_start
    log.info(f"\n{'='*55}\nPIPELINE DONE — {elapsed:.1f}s\nFigures: results/figures/\nMetrics: results/metrics/\n{'='*55}")


if __name__ == "__main__":
    main()