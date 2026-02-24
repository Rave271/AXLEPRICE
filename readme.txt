================================================================================
       AI-POWERED UK USED CAR VALUATION SYSTEM — README
================================================================================

Author  : Raghav Verma
Date    : February 2026
License : MIT

================================================================================
  1. PROJECT OVERVIEW
================================================================================

This project implements a two-stage AI system for used car price valuation.

  ALL heavy computation runs on Kaggle. The local code is a bare frontend.

  ┌─────────────────────────────────────────────────────────┐
  │                   KAGGLE  (heavy)                       │
  │  • Train Keras MLP neural network on 100K cars          │
  │  • Load & demo Qwen2.5-3B-Instruct LLM (GPU)           │
  │  • Push model + scaler + config to Hugging Face Hub     │
  └─────────────────────────────────────────────────────────┘
                           │
                     HF Hub (storage)
                           │
  ┌─────────────────────────────────────────────────────────┐
  │               LOCAL  (bare frontend)                    │
  │  • Download tiny NN model from HF  (cached, ~100 KB)   │
  │  • Run price prediction on CPU     (instant)            │
  │  • Call Qwen via HF Inference API  (serverless, no GPU) │
  │  • Display results (CLI or Streamlit UI)                │
  └─────────────────────────────────────────────────────────┘

  The local machine needs NO GPU and does NOT download the LLM.
  LLM explanations are generated serverless via HF Inference API.

================================================================================
  ★  QUICK START GUIDE  (step-by-step)
================================================================================

  This walks you through the full flow: Kaggle → Hugging Face → your Mac.

  ─────────────────────────────────────────────────────────────────────────
  STEP 1 — Get a Hugging Face token  (one-time, 2 min)
  ─────────────────────────────────────────────────────────────────────────

    1. Go to  https://huggingface.co/join  and create a free account.
    2. Go to  https://huggingface.co/settings/tokens
    3. Click  "Create new token".
       • Name : anything (e.g. "kaggle-car")
       • Type : **Write**  (needed to upload the model)
    4. Copy the token — it starts with  hf_...
       Keep it safe. You'll paste it in two places below.

  ─────────────────────────────────────────────────────────────────────────
  STEP 2 — Upload the notebook to Kaggle  (5 min)
  ─────────────────────────────────────────────────────────────────────────

    1. Go to  https://www.kaggle.com  (sign in or create free account).
    2. Click  "+ Create"  →  "New Notebook".
    3. Click  "File"  →  "Import Notebook"  →  upload  train_notebook.ipynb
       from this project folder.

  ─────────────────────────────────────────────────────────────────────────
  STEP 3 — Add the dataset to Kaggle  (2 min)
  ─────────────────────────────────────────────────────────────────────────

    1. In the right sidebar of the notebook, click  "Add data".
    2. Search for:  100000 UK Used Car Data Set
       (by Aditya — ~100K rows across multiple brand CSV files)
    3. Click  "Add"  to attach it to the notebook.
    4. Verify the dataset appears under  /kaggle/input/
       (The notebook auto-detects CSV files in that folder.)

  ─────────────────────────────────────────────────────────────────────────
  STEP 4 — Add your HF token as a Kaggle secret  (1 min)
  ─────────────────────────────────────────────────────────────────────────

    1. In the notebook, click the  "Add-ons"  menu →  "Secrets".
    2. Click  "Add a new secret".
       • Label :  HF_TOKEN
       • Value :  hf_...  (paste your token from Step 1)
    3. Toggle the secret  ON  (so the notebook can access it).

  ─────────────────────────────────────────────────────────────────────────
  STEP 5 — Enable GPU and set your repo name  (1 min)
  ─────────────────────────────────────────────────────────────────────────

    1. In the right sidebar, under "Session options":
       • Set  Accelerator  →  GPU T4 x2  (needed for the Qwen demo).
       • Set  Persistence  →  "Files only"  (optional, saves outputs).
    2. In  Cell 32  (the "Push to HF Hub" cell), find this line:
         REPO_ID = "YOUR_USERNAME/uk-used-car-nn"
       Replace  YOUR_USERNAME  with your Hugging Face username.
       Example:  REPO_ID = "raghavverma/uk-used-car-nn"

  ─────────────────────────────────────────────────────────────────────────
  STEP 6 — Run the notebook  (10–15 min)
  ─────────────────────────────────────────────────────────────────────────

    1. Click  "Run All"  (or  Ctrl+Shift+Enter / ⌘+Shift+Enter).
    2. The notebook will:
       ✓  Load & clean ~100K rows of car data
       ✓  Apply log1p price transformation
       ✓  Temporal split (year ≤ 2018 train / > 2018 test)
       ✓  One-hot encode categoricals, scale features
       ✓  Train Keras MLP (Dense 64 → 32 → 1) with EarlyStopping
       ✓  Evaluate: MAE, RMSE, R² — plot loss curves & scatter
       ✓  Save model, scaler, config as files
       ✓  Upload all 3 artefacts to your HF Hub repo
       ✓  Load Qwen LLM and demo a prediction + explanation
    3. Wait for all cells to finish. Check cell 32 prints:
         "All artefacts pushed to https://huggingface.co/your_user/..."

  ─────────────────────────────────────────────────────────────────────────
  STEP 7 — Verify on Hugging Face  (30 sec)
  ─────────────────────────────────────────────────────────────────────────

    1. Go to  https://huggingface.co/YOUR_USERNAME/uk-used-car-nn
    2. You should see 3 files:
       • car_price_model.keras
       • scaler.joblib
       • model_config.json
    3. If all three are there, training is done. You won't need
       Kaggle again (unless you retrain).

  ─────────────────────────────────────────────────────────────────────────
  STEP 8 — Set up your local machine  (3 min)
  ─────────────────────────────────────────────────────────────────────────

    Open Terminal on your Mac. Run:

      cd ~/Downloads/used\ car\ eval

      python3 -m venv .venv
      source .venv/bin/activate

      pip install pandas numpy scikit-learn tensorflow joblib \
                  huggingface_hub streamlit

    Set your HF token so the LLM explanation works:

      export HF_TOKEN="hf_paste_your_token_here"

    (Or add it to your  ~/.zshrc  so it persists across sessions.)

  ─────────────────────────────────────────────────────────────────────────
  STEP 9a — Run the Streamlit UI  (recommended)
  ─────────────────────────────────────────────────────────────────────────

      streamlit run app.py

    This opens a web page in your browser (usually http://localhost:8501):
      1. Fill in the sidebar: paste your HF repo ID and token.
      2. Enter car details (brand, model, year, mileage, etc.).
      3. Click  "💰 Get Valuation".
      4. Instantly see the predicted price.
      5. Wait ~2–3 sec for the Qwen LLM explanation (remote API call).

    Press  Ctrl+C  in Terminal to stop.

  ─────────────────────────────────────────────────────────────────────────
  STEP 9b — Run via CLI  (alternative)
  ─────────────────────────────────────────────────────────────────────────

      python inference.py \
          --repo YOUR_USERNAME/uk-used-car-nn \
          --brand BMW \
          --model "3 Series" \
          --year 2019 \
          --mileage 30000 \
          --engineSize 2.0 \
          --fuelType Petrol \
          --transmission Automatic \
          --token hf_your_token_here

    Expected output:

      ============================================================
        UK Used Car Valuation System
      ============================================================
        Loading model from HF: YOUR_USERNAME/uk-used-car-nn ...
        Predicting ...

        Predicted Price : £21,345.67

        Explanation:
        The 2019 BMW 3 Series is valued at approximately £21,345 ...
      ============================================================

    For price only (skip LLM), add  --no-explain.

  ─────────────────────────────────────────────────────────────────────────
  DONE!
  ─────────────────────────────────────────────────────────────────────────

  Summary of what lives where:

    KAGGLE   → training + Qwen demo  (run once, then done)
    HF HUB   → stores model artefacts  (permanent, public or private)
    YOUR MAC → bare frontend  (no GPU, no LLM download, instant)

================================================================================
  2. PROJECT STRUCTURE
================================================================================

  /used car eval
  ├── train_notebook.ipynb    # Kaggle notebook — ALL heavy lifting
  ├── app.py                  # Streamlit UI frontend (local, lightweight)
  ├── inference.py            # CLI frontend + predict_and_explain (local)
  ├── adapter.py              # HF Hub adapter — download tiny NN + scaler
  ├── utils.py                # Shared preprocessing utilities
  ├── Main.py                 # (empty placeholder)
  └── readme.txt              # This file

  Artefacts produced by training (pushed to HF Hub by Kaggle notebook):
  ├── car_price_model.keras   # Trained Keras neural network (~100 KB)
  ├── scaler.joblib           # Fitted StandardScaler (~few KB)
  └── model_config.json       # Feature column names (JSON)

================================================================================
  3. DATASET
================================================================================

  Name    : 100,000 UK Used Car Data Set
  Author  : Aditya
  Source  : https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes
  Size    : ~100,000 rows across multiple brand-specific CSV files

  Expected columns:
    price         — Target variable (GBP)
    year          — Registration year
    mileage       — Odometer reading (miles)
    engineSize    — Engine displacement (litres)
    fuelType      — Petrol / Diesel / Hybrid / Electric / Other
    transmission  — Manual / Automatic / Semi-Auto
    model         — Specific model name (e.g. "3 Series", "A Class")
    brand         — Manufacturer (e.g. "BMW", "Mercedes")

  Columns dropped during preprocessing:
    tax, tax(£), mpg — Not useful for price prediction

================================================================================
  4. ENVIRONMENT SETUP
================================================================================

  4.1 Python Version
  ------------------
  Python 3.9 or later is required.

  4.2 Install Dependencies
  ------------------------
  Run from the project root:

  LOCAL dependencies (lightweight — no torch, no transformers):

    pip install pandas numpy scikit-learn tensorflow joblib huggingface_hub streamlit

  KAGGLE dependencies (installed in notebook cell 2):

    pip install pandas numpy scikit-learn tensorflow matplotlib joblib \
                huggingface_hub transformers torch

  Or create a local virtual environment:

    python -m venv .venv
    source .venv/bin/activate          # macOS / Linux
    .venv\Scripts\activate             # Windows
    pip install pandas numpy scikit-learn tensorflow joblib huggingface_hub streamlit

  4.3 GPU — NOT needed locally
  -------------------------------------------
  All GPU work (NN training + Qwen LLM demo) runs on Kaggle.
  On Kaggle, enable GPU under Settings → Accelerator → GPU T4 x2.

  Locally, the NN runs on CPU (instant forward pass) and the LLM
  explanation is handled serverless via HF Inference API — zero GPU.

  4.4 Hugging Face Token
  -----------------------
  To push/pull models from Hugging Face Hub:

    1. Create a free account at https://huggingface.co
    2. Generate a token at https://huggingface.co/settings/tokens
       - Use a "Write" token if you plan to upload artefacts
    3. Set it as an environment variable:
         export HF_TOKEN="hf_your_token_here"
       Or on Kaggle: Add it as a Secret named "HF_TOKEN"

================================================================================
  5. USAGE INSTRUCTIONS
================================================================================

  5.1 Training (Kaggle Notebook)
  ------------------------------
  1. Upload train_notebook.ipynb to Kaggle.
  2. Add the dataset:
       - Search: "100000 UK Used Car Data Set" by Aditya
       - Add it to the notebook's input data
  3. Enable GPU accelerator in notebook settings.
  4. Add your Hugging Face token as a Kaggle Secret (key: "HF_TOKEN").
  5. Update the REPO_ID variable in cell 14 to your username:
       REPO_ID = "your_hf_username/uk-used-car-nn"
  6. Run All Cells.

  The notebook will:
    • Load and clean the data
    • Apply log1p transformation to prices
    • Split data temporally (year ≤ 2018 train / year > 2018 test)
    • One-hot encode categoricals
    • Scale features with StandardScaler
    • Train a Keras MLP with EarlyStopping
    • Evaluate and visualise results
    • Save artefacts to HF Hub
    • Load Qwen and run an end-to-end prediction with explanation

  5.2 Local Frontend — Streamlit UI  (recommended)
  --------------------------------------------------
  The simplest way to use the system locally:

    pip install streamlit
    export HF_TOKEN="hf_your_token_here"
    streamlit run app.py

  This opens a web UI where you fill in car details and click
  "Get Valuation". Predicted price appears instantly; the LLM
  explanation streams from HF Inference API (no local GPU).

  5.3 Local Frontend — CLI
  -------------------------
  After the model is on Hugging Face Hub:

    python inference.py \
        --repo your_hf_username/uk-used-car-nn \
        --brand BMW \
        --model "3 Series" \
        --year 2019 \
        --mileage 30000 \
        --engineSize 2.0 \
        --fuelType Petrol \
        --transmission Automatic \
        --token hf_your_token_here

  Output:
    ============================================================
      UK Used Car Valuation System
    ============================================================
      Loading model from HF: your_hf_username/uk-used-car-nn ...
      Predicting ...

      Predicted Price : £21,345.67

      Explanation:
      The 2019 BMW 3 Series is valued at approximately £21,345 ...
    ============================================================

  5.4 Skip LLM Explanation
  -------------------------
  If you just want the price prediction (no API call):

    python inference.py \
        --repo your_hf_username/uk-used-car-nn \
        --brand BMW --model "3 Series" --year 2019 \
        --mileage 30000 --engineSize 2.0 \
        --fuelType Petrol --transmission Automatic \
        --no-explain

  5.5 Programmatic Usage
  -----------------------
  In your own Python code:

    from adapter import load_model_from_hf
    from inference import predict_and_explain

    model, scaler, features = load_model_from_hf("your_user/uk-used-car-nn")

    result = predict_and_explain(
        car_dict={
            "brand": "Mercedes", "model": "C Class",
            "year": 2020, "mileage": 15000,
            "engineSize": 2.0, "fuelType": "Diesel",
            "transmission": "Automatic",
        },
        model=model, scaler=scaler,
        feature_columns=features,
        explain=True,
        hf_token="hf_your_token_here",
    )

    print(f"Price: £{result['predicted_price']:,.2f}")
    print(f"Explanation: {result['explanation_text']}")

================================================================================
  6. TECHNICAL DETAILS
================================================================================

  6.1 Preprocessing Pipeline
  --------------------------
  1. Load all brand CSV files and concatenate them.
  2. Drop irrelevant columns: tax, tax(£), mpg, unnamed index cols.
  3. Remove rows where price is missing or zero.
  4. Apply log1p(price) to reduce right-skewness.
     - log1p is used instead of log to handle price = 0 edge cases.
     - Inverse: expm1() converts predictions back to real prices.
  5. Temporal split on year:
     - Train: year ≤ 2018  (historical data)
     - Test:  year > 2018  (newer cars, simulates real deployment)
     - This avoids data leakage from future data into training.
  6. One-hot encode: fuelType, transmission, model, brand.
     - pd.get_dummies() with dtype=float.
     - Test columns aligned to train columns (missing filled with 0).
  7. StandardScaler fitted on train, applied to both train and test.
     - Scaler saved as scaler.joblib for inference.

  6.2 Neural Network Architecture
  --------------------------------
    Input (n features)
      → Dense(64, ReLU)
      → Dense(32, ReLU)
      → Dense(1, Linear)

    Compiler:
      Optimiser : Adam (default lr=0.001)
      Loss      : Mean Squared Error (MSE)
      Metric    : Mean Absolute Error (MAE)

    Training:
      Epochs          : up to 50
      Batch size      : 32
      Validation split: 0.2
      EarlyStopping   : patience=5, restore_best_weights=True

    Why this architecture?
      - Two hidden layers provide enough capacity for tabular data.
      - ReLU avoids vanishing gradients and is computationally efficient.
      - Linear output layer for regression.
      - EarlyStopping prevents overfitting on this relatively simple task.

  6.3 Evaluation Metrics
  ----------------------
    MAE  — Mean Absolute Error  : Average prediction error in pounds.
    RMSE — Root Mean Sq Error   : Penalises large errors more heavily.
    R²   — Coefficient of Det.  : Proportion of variance explained (1.0 = perfect).

    Metrics are computed on real-price scale (after expm1 inverse transform).

  6.4 Qwen LLM Integration
  -------------------------
    Model : Qwen/Qwen2.5-3B-Instruct

    ON KAGGLE (demo / training notebook):
      Loaded locally with torch float16, device_map="auto".
      Used to demonstrate end-to-end prediction + explanation.

    ON LOCAL MACHINE (production frontend):
      Called via HF Inference API (huggingface_hub.InferenceClient).
      Serverless — no model download, no GPU, no torch needed.
      Requires HF_TOKEN (free tier is sufficient).
      max_tokens = 150.

    The LLM receives a structured prompt with all car details and the
    predicted price, and generates a professional valuation explanation.
    It is NOT involved in the price prediction itself — only explanation.

  6.5 Hugging Face Hub Integration
  ---------------------------------
    Three artefacts are uploaded:
      1. car_price_model.keras  — The trained Keras model
      2. scaler.joblib          — The fitted StandardScaler
      3. model_config.json      — Feature column names (ordered list)

    The adapter.py module uses hf_hub_download() to fetch these files
    and loads them into memory for inference.

================================================================================
  7. MODULE REFERENCE
================================================================================

  utils.py
  --------
  │ load_and_clean(filepath)          → pd.DataFrame
  │ apply_log_target(df)              → pd.DataFrame
  │ temporal_split(df, split_year)    → (train_df, test_df)
  │ one_hot_encode(train, test)       → (train_encoded, test_encoded)
  │ scale_features(X_train, X_test)   → (X_train_s, X_test_s, scaler)
  │ preprocess_single(car_dict, ...)  → np.ndarray
  │ compute_metrics(y_true, y_pred)   → dict {MAE, RMSE, R2}

  adapter.py  (lightweight — downloads tiny files from HF)
  ----------
  │ load_model_from_hf(repo_id, cache_dir, token)
  │   → (keras.Model, StandardScaler, list[str])

  inference.py  (bare frontend — no local LLM)
  ------------
  │ predict_and_explain(car_dict, model, scaler, feature_columns,
  │                     explain, hf_token)
  │   → dict {predicted_price, explanation_text}
  │
  │ generate_explanation(car_dict, predicted_price, hf_token)
  │   → str  (calls HF Inference API, no local GPU)
  │
  │ CLI: python inference.py --repo ... --brand ... --token ...

  app.py  (Streamlit UI frontend)
  ------
  │ streamlit run app.py
  │ Web form → predicted price + LLM explanation
  │ Uses adapter.py + HF Inference API under the hood

================================================================================
  8. TROUBLESHOOTING
================================================================================

  Q: "No CSV files found" error in the notebook.
  A: The DATA_DIR path may differ. Check your Kaggle input path:
       !ls /kaggle/input/
     Update DATA_DIR accordingly.

  Q: "brand" column is missing.
  A: The dataset may use "manufacturer" or the filename-derived brand.
     The notebook handles this automatically by renaming.

  Q: CUDA out of memory when loading Qwen on Kaggle.
  A: Use a T4 or P100 GPU on Kaggle. If still OOM:
       - Reduce LLM to a smaller model (e.g. Qwen2.5-1.5B-Instruct)
       - This only affects the Kaggle demo; locally the LLM is called
         via HF Inference API (no local GPU needed).

  Q: LLM explanation says "Set HF_TOKEN to enable..."
  A: Export your Hugging Face token:
       export HF_TOKEN="hf_your_token_here"
     Or pass --token to the CLI. Free tier tokens work fine.

  Q: "ModuleNotFoundError: No module named 'huggingface_hub'"
  A: Run: pip install huggingface_hub

  Q: Model predictions seem off.
  A: Ensure you're using the same preprocessing pipeline (same scaler,
     same feature columns, same one-hot encoding). The model_config.json
     file stores the exact feature column order.

  Q: How to use a different HF repo name?
  A: Change REPO_ID in the notebook and --repo in the CLI.
     Format: "username/repo-name"

  Q: Can I retrain on a different dataset?
  A: Yes — update DATA_DIR and ensure columns match the expected names.
     Re-run the full notebook.

================================================================================
  9. REQUIREMENTS SUMMARY
================================================================================

  LOCAL (lightweight):
    pandas>=1.5
    numpy>=1.23
    scikit-learn>=1.2
    tensorflow>=2.13
    joblib>=1.2
    huggingface_hub>=0.19
    streamlit>=1.30       (for app.py frontend)

  KAGGLE (notebook only — NOT needed locally):
    matplotlib>=3.6
    transformers>=4.36
    torch>=2.1

  Note: torch and transformers are NOT required locally.
  The LLM is called serverless via HF Inference API.

================================================================================
  10. LICENCE
================================================================================

  This project is released under the MIT License.
  Dataset licence: refer to the Kaggle dataset page for terms.

================================================================================
