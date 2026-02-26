"""
config.py
=========
Central configuration module for the Exoplanet Candidate Vetting project.

All file paths, hyperparameters, feature lists, and experiment settings
are defined here so that changes propagate consistently across the entire
codebase.  Import this module wherever configuration is needed:

    from src.utils.config import Config

Author : MSc Dissertation Project
Dataset: NASA Kepler Objects of Interest (KOI)
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Root paths — all paths are resolved relative to the project root so the
# project works regardless of where it is cloned.
# ---------------------------------------------------------------------------

# Project root is two levels above this file (src/utils/config.py → project/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR       = PROJECT_ROOT / "data"
RAW_DIR        = DATA_DIR / "raw"
INTERIM_DIR    = DATA_DIR / "interim"
PROCESSED_DIR  = DATA_DIR / "processed"

RESULTS_DIR    = PROJECT_ROOT / "results"
FIGURES_DIR    = RESULTS_DIR / "figures"
METRICS_DIR    = RESULTS_DIR / "metrics"
MODELS_DIR     = RESULTS_DIR / "models"

NOTEBOOKS_DIR  = PROJECT_ROOT / "notebooks"
DOCS_DIR       = PROJECT_ROOT / "docs"

# ---------------------------------------------------------------------------
# Raw data file
# ---------------------------------------------------------------------------
RAW_DATA_FILE  = RAW_DIR / "koi_data.csv"
RAW_DATA_FILE2 = RAW_DIR / "kepler_light_data.csv" 

# Processed / interim artefacts
PROCESSED_FILE         = PROCESSED_DIR / "koi_processed.csv"
TRAIN_FILE             = PROCESSED_DIR / "train.csv"
VALIDATION_FILE        = PROCESSED_DIR / "val.csv"
TEST_FILE              = PROCESSED_DIR / "test.csv"
SCALER_FILE            = MODELS_DIR / "scaler.pkl"
LABEL_ENCODER_FILE     = MODELS_DIR / "label_encoder.pkl"

# ---------------------------------------------------------------------------
# Target column
# ---------------------------------------------------------------------------
TARGET_COL = "koi_disposition"      # CONFIRMED | FALSE POSITIVE | CANDIDATE

# Binary label mapping used for binary-classification experiments
BINARY_LABEL_MAP = {
    "CONFIRMED"    : 1,   # Planet
    "FALSE POSITIVE": 0,  # Not a planet
    "CANDIDATE"    : -1,  # Excluded from binary experiments
}

# Multi-class integer mapping
MULTICLASS_LABEL_MAP = {
    "CONFIRMED"    : 0,
    "FALSE POSITIVE": 1,
    "CANDIDATE"    : 2,
}

CLASS_NAMES = ["CONFIRMED", "FALSE POSITIVE", "CANDIDATE"]

# ---------------------------------------------------------------------------
# Feature engineering — columns to DROP before modelling
# These are identifiers, leaky columns, or fully-empty columns discovered
# during EDA.
# ---------------------------------------------------------------------------

# Columns that are completely empty in this dataset
FULLY_EMPTY_COLS = [
    "koi_ingress", "koi_ingress_err1", "koi_ingress_err2",
    "koi_eccen", "koi_eccen_err1", "koi_eccen_err2",
    "koi_longp", "koi_longp_err1", "koi_longp_err2",
    "koi_incl_err1", "koi_incl_err2",
    "koi_teq_err1", "koi_teq_err2",
    "koi_sma_err1", "koi_sma_err2",
    "koi_model_chisq", "koi_model_dof",
    "koi_sage", "koi_sage_err1", "koi_sage_err2",
]

# Identifier / metadata columns — not predictive
ID_COLS = [
    "rowid", "kepid", "kepoi_name", "kepler_name",
    "koi_vet_stat", "koi_vet_date", "koi_disp_prov",
    "koi_comment", "koi_tce_delivname", "koi_quarters",
    "koi_datalink_dvr", "koi_datalink_dvs",
    "koi_fittype", "koi_limbdark_mod", "koi_parm_prov",
    "koi_trans_mod", "koi_sparprov",
]

# koi_pdisposition and koi_score are semi-leaky (derived from same pipeline)
# They are kept for experiments that intentionally include them but excluded
# from the "clean" feature set.
LEAKY_COLS = ["koi_pdisposition", "koi_score"]

# All columns to drop for the *clean* modelling feature set
DROP_COLS = FULLY_EMPTY_COLS + ID_COLS + LEAKY_COLS

# ---------------------------------------------------------------------------
# Selected feature groups (domain knowledge driven)
# These are the astrophysical features used as primary model inputs
# ---------------------------------------------------------------------------

TRANSIT_FEATURES = [
    "koi_period",          # Orbital period (days)
    "koi_time0bk",         # Transit epoch (BKJD)
    "koi_impact",          # Impact parameter
    "koi_duration",        # Transit duration (hours)
    "koi_depth",           # Transit depth (ppm)
    "koi_ror",             # Planet-to-star radius ratio
    "koi_srho",            # Stellar density from transit fit
    "koi_prad",            # Planet radius (Earth radii)
    "koi_sma",             # Semi-major axis (AU)
    "koi_incl",            # Orbital inclination (degrees)
    "koi_teq",             # Equilibrium temperature (K)
    "koi_insol",           # Insolation flux (Earth units)
    "koi_dor",             # Planet-to-star distance over stellar radius
]

STELLAR_FEATURES = [
    "koi_steff",           # Stellar effective temperature (K)
    "koi_slogg",           # Stellar surface gravity (log g)
    "koi_smet",            # Stellar metallicity [Fe/H]
    "koi_srad",            # Stellar radius (Solar radii)
    "koi_smass",           # Stellar mass (Solar masses)
    "ra", "dec",           # Sky coordinates
    "koi_kepmag",          # Kepler-band magnitude
]

PHOTOMETRIC_FLAGS = [
    "koi_fpflag_nt",       # Not transit-like flag
    "koi_fpflag_ss",       # Significant secondary flag
    "koi_fpflag_co",       # Centroid offset flag
    "koi_fpflag_ec",       # Ephemeris match flag
]

SIGNAL_FEATURES = [
    "koi_max_sngle_ev",    # Maximum single-event statistic
    "koi_max_mult_ev",     # Maximum multiple-event statistic
    "koi_model_snr",       # Signal-to-noise ratio of model fit
    "koi_num_transits",    # Number of observed transits
    "koi_bin_oedp_sig",    # Odd-even depth difference significance
]

MAGNITUDE_FEATURES = [
    "koi_gmag", "koi_rmag", "koi_imag",
    "koi_zmag", "koi_jmag", "koi_hmag", "koi_kmag",
]

# Combined feature set for modelling
ALL_MODEL_FEATURES = (
    TRANSIT_FEATURES
    + STELLAR_FEATURES
    + PHOTOMETRIC_FLAGS
    + SIGNAL_FEATURES
    + MAGNITUDE_FEATURES
)

# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------
TRAIN_SIZE      = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE       = 0.15
RANDOM_SEED     = 42           # Fixed for full reproducibility

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
IMPUTATION_STRATEGY = "median"  # Strategy for SimpleImputer on numeric cols
SCALER_TYPE         = "standard" # 'standard' | 'minmax' | 'robust'

# ---------------------------------------------------------------------------
# Classical ML hyperparameters (used in GridSearchCV)
# ---------------------------------------------------------------------------
RF_PARAM_GRID = {
    "n_estimators"      : [100, 200, 300],
    "max_depth"         : [None, 10, 20, 30],
    "min_samples_split" : [2, 5],
    "min_samples_leaf"  : [1, 2],
    "class_weight"      : ["balanced"],
}

SVM_PARAM_GRID = {
    "C"      : [0.1, 1, 10, 100],
    "gamma"  : ["scale", "auto"],
    "kernel" : ["rbf", "poly"],
    "class_weight": ["balanced"],
}

# ---------------------------------------------------------------------------
# Genesis CNN hyperparameters
# ---------------------------------------------------------------------------
CNN_CONFIG = {
    # Input
    "input_length"    : 50,     # Length of synthetic/tabular feature vector
                                # reshaped to (N, input_length, 1) for Conv1D

    # Architecture — simplified Genesis multi-branch design
    "branches"        : 2,      # Number of parallel convolutional branches

    # Branch 1: captures local transit features
    "b1_filters"      : [32, 64],
    "b1_kernels"      : [3, 3],

    # Branch 2: captures global/period features
    "b2_filters"      : [32, 64],
    "b2_kernels"      : [7, 5],

    # Merged dense head
    "dense_units"     : [128, 64],
    "dropout_rate"    : 0.4,

    # Output
    "num_classes"     : 3,       # CONFIRMED / FALSE POSITIVE / CANDIDATE

    # Training
    "epochs"          : 60,
    "batch_size"      : 32,
    "learning_rate"   : 1e-3,
    "patience"        : 10,      # Early stopping patience

    # Regularisation
    "l2_lambda"       : 1e-4,
}

# Baseline CNN (simpler architecture for comparison)
BASELINE_CNN_CONFIG = {
    "filters"       : [32, 64, 128],
    "kernel_size"   : 3,
    "dense_units"   : [64],
    "dropout_rate"  : 0.3,
    "num_classes"   : 3,
    "epochs"        : 60,
    "batch_size"    : 32,
    "learning_rate" : 1e-3,
    "patience"      : 10,
}

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
CV_FOLDS          = 5       # Cross-validation folds for classical ML
METRICS_AVERAGE   = "macro" # Averaging for multi-class metrics
ROC_PLOT_CLASSES  = CLASS_NAMES

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = "INFO"
LOG_FILE  = PROJECT_ROOT / "results" / "experiment.log"

# ---------------------------------------------------------------------------
# Matplotlib / plotting style
# ---------------------------------------------------------------------------
PLOT_STYLE   = "seaborn-v0_8-whitegrid"
PLOT_DPI     = 150
PLOT_FIGSIZE = (10, 6)
COLOR_PALETTE = ["#2ecc71", "#e74c3c", "#3498db"]   # green/red/blue

# ---------------------------------------------------------------------------
# Ensure all output directories exist at import time
# ---------------------------------------------------------------------------
for _dir in [INTERIM_DIR, PROCESSED_DIR, FIGURES_DIR, METRICS_DIR, MODELS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)
