AXLEPRICE
=========

AxlePrice is a two-stage system for used car price valuation.
Heavy training runs on Kaggle, while the local app serves the website
and runs fast CPU-only inference using the tiny model from Hugging Face.

Quick Start
-----------
1) Create a virtual environment and install dependencies:
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

2) (Optional) set your Hugging Face token for explanations:
   echo 'HF_TOKEN=hf_your_token_here' > .env

3) Run the web app:
   python -m uvicorn app:app --reload --port 8000

Open:
  http://127.0.0.1:8000
  http://127.0.0.1:8000/valuation

Training (Kaggle)
-----------------
Use train_notebook.ipynb on Kaggle to train and push the model artefacts
to your Hugging Face repo (model, scaler, config). You only need to do
this when you want to retrain.

Project Structure
-----------------
app.py            FastAPI server
inference.py      CLI runner
adapter.py        Hugging Face model fetcher
utils.py          Shared preprocessing
train_notebook.ipynb  Kaggle training notebook
requirements.txt  Local dependencies
web/              HTML templates + static assets

Dataset
-------
Name: 100,000 UK Used Car Data Set (Aditya)
Source: https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes

Notes
-----
- No GPU is required locally.
- AI explanations use the Hugging Face Inference API when HF_TOKEN is set.
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
      AxlePrice
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

  6.4 LLM Integration
  -------------------------
    ON KAGGLE (demo only):
      Model : Qwen/Qwen2.5-3B-Instruct
      Loaded locally with torch float16, device_map="auto".
      Used to demonstrate end-to-end prediction + explanation in notebook.
      Requires GPU (T4 or better).

    ON LOCAL MACHINE (production frontend):
      Model : meta-llama/Meta-Llama-3-8B-Instruct
      Called via HF Inference API (huggingface_hub.InferenceClient).
      Serverless — HF runs the model on their infrastructure.
      No local GPU, no model download, no torch/transformers needed.
      Requires HF_TOKEN (free tier is sufficient).
      max_tokens = 400, timeout = 60 seconds.

    Prompt engineering:
      - System prompt: "You are a used car pricing expert. Give short,
        specific, data-driven answers. No headers or preamble."
      - User prompt: provides car details + predicted price, asks for
        3-4 concise bullet points with specific numbers.

    The LLM receives a structured prompt with all car details and the
    predicted price, and generates a professional valuation explanation.
    It is NOT involved in the price prediction itself — only explanation.

    Note: Qwen2.5-3B-Instruct and Mistral-7B-Instruct-v0.3 were tested
    for local serverless use but were not available as chat models on
    the HF Inference API. Meta-Llama-3-8B-Instruct works reliably.

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

  app.py  (website backend)
  ------
  │ python -m uvicorn app:app --reload --port 8000
  │ Web pages + API → predicted price + optional LLM explanation
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

  Q: Prediction hangs when using TensorFlow (model.predict() freezes).
  A: In some environments, TensorFlow's model.predict() can hang due to
     thread/execution context interactions. Use the direct call instead:
       model(X, training=False).numpy()
     This is already used in app.py.

  Q: LLM explanation is cut off or too generic.
  A: Ensure max_tokens is set to at least 400. The prompt should request
     concise bullet points with specific numbers.

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
    fastapi>=0.110
    uvicorn>=0.23
    jinja2>=3.1
    python-dotenv>=1.0    (auto-loads .env file in app.py)

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
