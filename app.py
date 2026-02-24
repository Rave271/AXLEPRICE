"""
app.py — Streamlit frontend for the UK Used Car Valuation System.

A bare-bones local UI that:
  • Takes car details via form inputs
  • Sends them through the local NN model (downloaded from HF, tiny)
  • Calls HF Inference API for the Qwen LLM explanation (serverless)
  • Displays the predicted price and explanation

Run:
    streamlit run app.py
"""

import os
import streamlit as st
import numpy as np

# Auto-load .env file
from dotenv import load_dotenv
load_dotenv()

from adapter import load_model_from_hf
from utils import preprocess_single

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="UK Used Car Valuation", page_icon="🚗", layout="centered")
st.title("🚗 UK Used Car Valuation")
st.caption("AI-powered price prediction + professional explanation")

# ---------------------------------------------------------------------------
# Sidebar — settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    repo_id = st.text_input("HF Repo ID", value="Rave271/uk-used-car-nn",
                            help="Hugging Face repo where the model is hosted")
    hf_token = st.text_input("HF Token", type="password",
                             value=os.environ.get("HF_TOKEN", ""),
                             help="Needed for model download and LLM explanations")
    want_explanation = st.checkbox("Generate LLM explanation", value=True)

# ---------------------------------------------------------------------------
# Load model (cached so it only downloads once)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model(repo_id: str, token: str):
    """Download and cache the NN model + scaler from HF Hub."""
    return load_model_from_hf(repo_id=repo_id, token=token or None)


# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
st.subheader("Enter Car Details")

col1, col2 = st.columns(2)
with col1:
    brand = st.text_input("Brand", value="BMW")
    car_model = st.text_input("Model", value="3 Series")
    year = st.number_input("Year", min_value=1990, max_value=2026, value=2019)
    mileage = st.number_input("Mileage", min_value=0, max_value=500000, value=30000)

with col2:
    engine_size = st.number_input("Engine Size (L)", min_value=0.0, max_value=8.0,
                                   value=2.0, step=0.1)
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid", "Electric", "Other"])
    transmission = st.selectbox("Transmission", ["Automatic", "Manual", "Semi-Auto"])

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
if st.button("💰 Get Valuation", type="primary", use_container_width=True):
    car_dict = {
        "brand": brand,
        "model": car_model,
        "year": year,
        "mileage": mileage,
        "engineSize": engine_size,
        "fuelType": fuel_type,
        "transmission": transmission,
    }

    with st.spinner("Loading model from Hugging Face..."):
        try:
            nn_model, scaler, feature_columns = load_model(repo_id, hf_token)
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.stop()

    # ── Price prediction (local, instant) ──
    # Use direct model call instead of .predict() to avoid TF threading hang
    with st.spinner("Predicting price..."):
        X = preprocess_single(car_dict, feature_columns, scaler)
        log_price = float(nn_model(X, training=False).numpy().flatten()[0])
        predicted_price = float(np.expm1(log_price))

    st.divider()
    st.metric(label="Predicted Price", value=f"£{predicted_price:,.2f}")

    # ── LLM explanation (HF Inference API, serverless) ──
    if want_explanation:
        with st.spinner("Generating explanation (Qwen via HF API)..."):
            try:
                from huggingface_hub import InferenceClient

                token = hf_token or os.environ.get("HF_TOKEN")
                if not token:
                    st.warning("Set HF_TOKEN to enable LLM explanations.")
                else:
                    client = InferenceClient(token=token, timeout=60)

                    prompt = f"""A {year} {brand} {car_model} with {mileage:,} miles, {engine_size}L {fuel_type} engine, {transmission} transmission is valued at £{predicted_price:,.2f}.

In 3-4 concise bullet points, explain the key factors behind this specific price. Be specific with numbers — compare to typical values for this brand/model. No generic filler."""

                    messages = [
                        {"role": "system", "content": "You are a used car pricing expert. Give short, specific, data-driven answers. No headers or preamble."},
                        {"role": "user", "content": prompt},
                    ]

                    response = client.chat_completion(
                        model="meta-llama/Meta-Llama-3-8B-Instruct",
                        messages=messages,
                        max_tokens=400,
                    )

                    explanation = response.choices[0].message.content.strip()
                    st.subheader("AI Explanation")
                    st.write(explanation)

            except Exception as e:
                st.warning(f"LLM explanation failed: {e}")
