"""
preprocess.py
=============
End-to-end preprocessing pipeline for the NASA Kepler KOI dataset.

Pipeline steps
--------------
1. Load raw CSV (skipping NASA comment header lines beginning with '#')
2. Drop non-informative, identifier, and fully-empty columns
3. Encode the target label (koi_disposition) as integers
4. Impute missing numeric values with the column median
5. Scale numeric features with StandardScaler
6. Stratified train / validation / test split (70 / 15 / 15)
7. Persist processed splits and fitted transformers to disk

Usage
-----
    python -m src.data.preprocess

    # Or import for use in notebooks:
    from src.data.preprocess import run_preprocessing
    X_train, X_val, X_test, y_train, y_val, y_test = run_preprocessing()
"""

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Allow running as a script from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import (
    RAW_DATA_FILE, PROCESSED_FILE,
    TRAIN_FILE, VALIDATION_FILE, TEST_FILE,
    SCALER_FILE, LABEL_ENCODER_FILE,
    TARGET_COL, DROP_COLS, ALL_MODEL_FEATURES,
    TRAIN_SIZE, VALIDATION_SIZE, TEST_SIZE,
    RANDOM_SEED, IMPUTATION_STRATEGY, MULTICLASS_LABEL_MAP,
    PROCESSED_DIR, MODELS_DIR,
)
from src.utils.logger import get_logger
from src.utils.seed import set_all_seeds

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Step 1 – Load raw data
# ---------------------------------------------------------------------------

def load_raw_data(filepath: Path = RAW_DATA_FILE) -> pd.DataFrame:
    """
    Load the NASA KOI CSV file, skipping comment lines that begin with '#'.

    The NASA Exoplanet Archive prepends several comment / column-description
    lines before the actual CSV header; ``comment='#'`` handles this cleanly.

    Parameters
    ----------
    filepath : Path
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with all 141 columns as downloaded.
    """
    log.info(f"Loading raw data from: {filepath}")
    df = pd.read_csv(filepath, comment="#", low_memory=False)
    log.info(f"Raw data loaded — shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Step 2 – Drop non-informative columns
# ---------------------------------------------------------------------------

def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove identifier, metadata, leaky, and fully-empty columns defined in
    ``config.DROP_COLS``.  Only drops columns that actually exist in the
    dataframe to avoid KeyErrors if the schema changes.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Dataframe with reduced column set.
    """
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    log.info(f"Dropping {len(cols_to_drop)} non-informative columns.")
    return df.drop(columns=cols_to_drop)


# ---------------------------------------------------------------------------
# Step 3 – Filter & encode target
# ---------------------------------------------------------------------------

def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map the string target ``koi_disposition`` to integer class labels using
    ``MULTICLASS_LABEL_MAP`` defined in config.

    Also drops any rows where the target is NaN or not in the map.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Dataframe with an added integer column ``label`` and the original
        ``koi_disposition`` column still present for reference.
    """
    initial_len = len(df)
    df = df.dropna(subset=[TARGET_COL])

    # Keep only known classes
    df = df[df[TARGET_COL].isin(MULTICLASS_LABEL_MAP.keys())].copy()
    dropped = initial_len - len(df)
    if dropped:
        log.warning(f"Dropped {dropped} rows with unknown/NaN target values.")

    df["label"] = df[TARGET_COL].map(MULTICLASS_LABEL_MAP)
    log.info(f"Target distribution after encoding:\n{df['label'].value_counts()}")
    return df


# ---------------------------------------------------------------------------
# Step 4 – Select model features
# ---------------------------------------------------------------------------

def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate the feature matrix X from the target vector y.

    Only columns listed in ``ALL_MODEL_FEATURES`` (config) that are actually
    present in the dataframe are selected.  This makes the pipeline robust
    to minor schema differences.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Integer-encoded target labels.
    """
    available_features = [f for f in ALL_MODEL_FEATURES if f in df.columns]
    missing = set(ALL_MODEL_FEATURES) - set(available_features)
    if missing:
        log.warning(f"Features not found in data (will be skipped): {missing}")

    X = df[available_features].copy()
    y = df["label"].copy()
    log.info(f"Feature matrix shape: {X.shape}  |  Target shape: {y.shape}")
    return X, y


# ---------------------------------------------------------------------------
# Step 5 – Impute missing values
# ---------------------------------------------------------------------------

def impute_missing(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    strategy: str = IMPUTATION_STRATEGY,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a ``SimpleImputer`` on the training set only and transform all splits.

    Using only training statistics avoids data leakage from validation/test
    sets, which is essential for honest evaluation.

    Parameters
    ----------
    X_train, X_val, X_test : pd.DataFrame
        Feature splits.
    strategy : str
        Imputation strategy ('median', 'mean', 'most_frequent').

    Returns
    -------
    Tuple of three np.ndarray (train, val, test).
    """
    log.info(f"Imputing missing values with strategy='{strategy}'")
    imputer = SimpleImputer(strategy=strategy)
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp   = imputer.transform(X_val)
    X_test_imp  = imputer.transform(X_test)
    log.info("Imputation complete.")
    return X_train_imp, X_val_imp, X_test_imp, imputer


# ---------------------------------------------------------------------------
# Step 6 – Scale features
# ---------------------------------------------------------------------------

def scale_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit a ``StandardScaler`` on the training set and transform all splits.

    StandardScaler (z-score normalisation) is appropriate for the mixed
    astrophysical features in this dataset.  The fitted scaler is returned
    so it can be persisted and applied to new data at inference time.

    Parameters
    ----------
    X_train, X_val, X_test : np.ndarray

    Returns
    -------
    Scaled arrays and the fitted scaler object.
    """
    log.info("Fitting StandardScaler on training data.")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)
    log.info("Feature scaling complete.")
    return X_train_sc, X_val_sc, X_test_sc, scaler


# ---------------------------------------------------------------------------
# Step 7 – Train / Val / Test split
# ---------------------------------------------------------------------------

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple:
    """
    Stratified split into train (70%), validation (15%), and test (15%).

    Stratification preserves the class distribution across all three splits,
    which is important given the imbalance between CONFIRMED, FALSE POSITIVE,
    and CANDIDATE classes.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series

    Returns
    -------
    Six-tuple: X_train, X_val, X_test, y_train, y_val, y_test
    """
    log.info(
        f"Splitting data — train:{TRAIN_SIZE:.0%}  "
        f"val:{VALIDATION_SIZE:.0%}  test:{TEST_SIZE:.0%}"
    )

    # First split off the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    # Split the remainder into train + validation
    # Relative validation size within the temp set
    relative_val_size = VALIDATION_SIZE / (TRAIN_SIZE + VALIDATION_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=relative_val_size,
        stratify=y_temp,
        random_state=RANDOM_SEED,
    )

    log.info(
        f"Split sizes — train: {len(X_train)}  "
        f"val: {len(X_val)}  test: {len(X_test)}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Step 8 – Persist processed data
# ---------------------------------------------------------------------------

def save_splits(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
    y_train: pd.Series, y_val: pd.Series, y_test: pd.Series,
    feature_names: list[str],
) -> None:
    """
    Save the processed train/val/test splits as CSV files so they can be
    loaded directly by model training scripts without re-running the full
    pipeline.

    Parameters
    ----------
    X_train, X_val, X_test : np.ndarray
        Scaled feature arrays.
    y_train, y_val, y_test : pd.Series
        Integer label arrays.
    feature_names : list[str]
        Column names matching the feature arrays.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    def _to_df(X, y):
        df = pd.DataFrame(X, columns=feature_names)
        df["label"] = y.values
        return df

    _to_df(X_train, y_train).to_csv(TRAIN_FILE, index=False)
    _to_df(X_val,   y_val).to_csv(VALIDATION_FILE, index=False)
    _to_df(X_test,  y_test).to_csv(TEST_FILE, index=False)
    log.info(f"Splits saved to {PROCESSED_DIR}")


def save_transformers(scaler: StandardScaler, imputer) -> None:
    """
    Pickle the fitted scaler and imputer so they can be reloaded for
    inference on new data without re-fitting.

    Parameters
    ----------
    scaler  : StandardScaler
    imputer : SimpleImputer
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SCALER_FILE, "wb") as f:
        pickle.dump(scaler, f)
    with open(MODELS_DIR / "imputer.pkl", "wb") as f:
        pickle.dump(imputer, f)
    log.info(f"Scaler saved → {SCALER_FILE}")
    log.info(f"Imputer saved → {MODELS_DIR / 'imputer.pkl'}")


# ---------------------------------------------------------------------------
# Main pipeline orchestrator
# ---------------------------------------------------------------------------

def run_preprocessing() -> tuple:
    """
    Execute the full preprocessing pipeline from raw CSV to scaled,
    split numpy arrays.

    Returns
    -------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test, feature_names)

    Notes
    -----
    All intermediate artefacts (splits, scaler, imputer) are saved to disk
    for downstream use by training scripts and notebooks.
    """
    set_all_seeds()

    # --- Load ---
    df = load_raw_data()

    # --- Clean ---
    df = drop_columns(df)
    df = encode_target(df)

    # --- Features / target ---
    X, y = select_features(df)
    feature_names = X.columns.tolist()

    # --- Split (before scaling to prevent leakage) ---
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # --- Impute ---
    X_train_imp, X_val_imp, X_test_imp, imputer = impute_missing(
        X_train, X_val, X_test
    )

    # --- Scale ---
    X_train_sc, X_val_sc, X_test_sc, scaler = scale_features(
        X_train_imp, X_val_imp, X_test_imp
    )

    # --- Persist ---
    save_splits(
        X_train_sc, X_val_sc, X_test_sc,
        y_train, y_val, y_test,
        feature_names,
    )
    save_transformers(scaler, imputer)

    # Also save the full processed dataframe for EDA notebook
    df_processed = df.copy()
    df_processed.to_csv(PROCESSED_FILE, index=False)
    log.info(f"Full processed dataframe → {PROCESSED_FILE}")

    log.info("=" * 60)
    log.info("Preprocessing pipeline complete.")
    log.info(f"  Training samples  : {len(X_train_sc)}")
    log.info(f"  Validation samples: {len(X_val_sc)}")
    log.info(f"  Test samples      : {len(X_test_sc)}")
    log.info(f"  Features          : {len(feature_names)}")
    log.info("=" * 60)

    return X_train_sc, X_val_sc, X_test_sc, y_train, y_val, y_test, feature_names


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_preprocessing()
