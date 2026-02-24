"""
utils.py — Shared utility functions for the Used Car Valuation System.

Provides preprocessing, feature engineering, and helper functions
used across training, inference, and adapter modules.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict, Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORICAL_COLS = ["fuelType", "transmission", "model", "brand"]
NUMERIC_COLS = ["year", "mileage", "engineSize"]
TARGET_COL = "price"
TEMPORAL_SPLIT_YEAR = 2018

# Columns considered irrelevant for modelling (drop if present)
DROP_COLS = ["tax", "tax(£)", "mpg"]

# Reproducibility
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Data loading & cleaning
# ---------------------------------------------------------------------------

def load_and_clean(filepath: str) -> pd.DataFrame:
    """Load CSV, drop irrelevant columns, and remove rows with missing price."""
    df = pd.read_csv(filepath)

    # Normalise column names to lower-camelCase expected by the pipeline
    col_map = {c: c.strip() for c in df.columns}
    df.rename(columns=col_map, inplace=True)

    # Drop columns that are not useful
    existing_drop = [c for c in DROP_COLS if c in df.columns]
    if existing_drop:
        df.drop(columns=existing_drop, inplace=True)

    # Remove rows where target is missing
    df.dropna(subset=[TARGET_COL], inplace=True)

    # Basic sanity filters
    df = df[df[TARGET_COL] > 0].copy()

    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def apply_log_target(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p transformation to the price column."""
    df = df.copy()
    df[TARGET_COL] = np.log1p(df[TARGET_COL])
    return df


def temporal_split(
    df: pd.DataFrame, split_year: int = TEMPORAL_SPLIT_YEAR
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data temporally: train on year <= split_year, test on year > split_year."""
    train = df[df["year"] <= split_year].copy()
    test = df[df["year"] > split_year].copy()
    return train, test


def one_hot_encode(
    train: pd.DataFrame,
    test: pd.DataFrame,
    categorical_cols: List[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """One-hot encode categoricals and align train/test columns."""
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_COLS

    # Only encode columns that exist
    cols_to_encode = [c for c in categorical_cols if c in train.columns]

    train = pd.get_dummies(train, columns=cols_to_encode, dtype=float)
    test = pd.get_dummies(test, columns=cols_to_encode, dtype=float)

    # Align columns — keep only columns present in train, fill missing with 0
    train, test = train.align(test, join="left", axis=1, fill_value=0)

    return train, test


def scale_features(
    X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit StandardScaler on train and transform both splits."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# ---------------------------------------------------------------------------
# Preprocessing pipeline (for inference)
# ---------------------------------------------------------------------------

def preprocess_single(
    car_dict: Dict[str, Any],
    feature_columns: List[str],
    scaler: StandardScaler,
) -> np.ndarray:
    """
    Preprocess a single car dictionary for prediction.

    Converts the input dict to a dataframe, one-hot encodes it,
    aligns to the training feature columns, scales, and returns
    a numpy array ready for model.predict().
    """
    df = pd.DataFrame([car_dict])

    # One-hot encode categorical columns present in the input
    cols_to_encode = [c for c in CATEGORICAL_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=cols_to_encode, dtype=float)

    # Align to training columns
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Scale
    X = scaler.transform(df.values)
    return X


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Compute MAE, RMSE, and R² on real-price scale."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {"MAE": mae, "RMSE": rmse, "R2": r2}
