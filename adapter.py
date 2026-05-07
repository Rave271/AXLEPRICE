"""
adapter.py — Hugging Face Hub adapter for the Used Car Valuation model.

Downloads the trained Keras model, scaler, and feature configuration
from a Hugging Face repository and returns ready-to-use objects.

This is a lightweight module — it only downloads small artefacts from HF
and loads them locally.  No GPU required.
"""

import json
import os
from typing import Tuple, Dict, List, Any

import joblib
from huggingface_hub import hf_hub_download


# ---------------------------------------------------------------------------
# Default artefact filenames (must match what train.py uploads)
# ---------------------------------------------------------------------------

MODEL_FILENAME        = "car_price_model.keras"
SCALER_FILENAME       = "scaler.joblib"
CONFIG_FILENAME       = "model_config.json"
NUMERIC_IDX_FILENAME  = "numeric_idx.joblib"
MODEL_ENCODER_FILENAME = "model_encoder.joblib"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_model_from_hf(
    repo_id: str,
    cache_dir: str = None,
    token: str = None,
) -> Tuple["keras.Model", Any, List[str]]:
    """
    Download model artefacts from Hugging Face Hub and load them.

    Parameters
    ----------
    repo_id : str
        Hugging Face repository identifier, e.g. ``"username/uk-used-car-nn"``.
    cache_dir : str, optional
        Local directory to cache downloaded files. Defaults to the HF cache.
    token : str, optional
        Hugging Face access token. Falls back to the cached login token.

    Returns
    -------
    model : keras.Model
        The loaded Keras neural network (runs on CPU).
    scaler : sklearn.preprocessing.StandardScaler
        The fitted StandardScaler used during training.
    feature_columns : list[str]
        Ordered list of feature column names expected by the model.
    numeric_idx : list[int]
        Indices of numeric columns that should be scaled.
    model_encoder : dict
        Smoothed target-encoding map for the 'model' column.
    reference_year : int
        Reference year used to compute car_age during training.
    """
    # Download each artefact (small files, cached after first call) ----------
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=MODEL_FILENAME,
        cache_dir=cache_dir,
        token=token,
    )
    scaler_path = hf_hub_download(
        repo_id=repo_id,
        filename=SCALER_FILENAME,
        cache_dir=cache_dir,
        token=token,
    )
    config_path = hf_hub_download(
        repo_id=repo_id,
        filename=CONFIG_FILENAME,
        cache_dir=cache_dir,
        token=token,
    )

    numeric_idx_path = hf_hub_download(
        repo_id=repo_id,
        filename=NUMERIC_IDX_FILENAME,
        cache_dir=cache_dir,
        token=token,
    )
    model_encoder_path = hf_hub_download(
        repo_id=repo_id,
        filename=MODEL_ENCODER_FILENAME,
        cache_dir=cache_dir,
        token=token,
    )

    from tensorflow import keras
    model = keras.models.load_model(model_path)

    scaler        = joblib.load(scaler_path)
    numeric_idx   = joblib.load(numeric_idx_path)
    model_encoder = joblib.load(model_encoder_path)

    with open(config_path, "r") as f:
        config = json.load(f)
    feature_columns = config["feature_columns"]
    reference_year  = config.get("reference_year", 2020)

    print(f"[adapter] Loaded model, scaler, and config from '{repo_id}'")
    print(f"[adapter] Feature count: {len(feature_columns)}, numeric_idx: {numeric_idx}")

    return model, scaler, feature_columns, numeric_idx, model_encoder, reference_year
