"""
inference.py — Lightweight local frontend for the Used Car Valuation System.

All heavy computation (NN training, LLM weights) lives on Kaggle / HF Hub.
Locally we only:
  1. Download the tiny Keras NN + scaler from Hugging Face (cached after first use)
  2. Run a single forward-pass for price prediction (CPU, instant)
  3. Call HF Inference API for Qwen LLM explanation (serverless, no local GPU)

Usage
-----
    python inference.py \
        --repo  username/uk-used-car-nn \
        --brand BMW --model "3 Series" --year 2019 \
        --mileage 30000 --engineSize 2.0 \
        --fuelType Diesel --transmission Automatic
"""

import argparse
import os
import sys
from typing import Any, Dict

import numpy as np

from adapter import load_model_from_hf
from utils import preprocess_single


# ---------------------------------------------------------------------------
# LLM explanation via HF Inference API  (zero local GPU required)
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """\
A {year} {brand} {model} with {mileage} miles, {engine}L {fuel} engine, {transmission} transmission is valued at £{price}.

In 3-4 concise bullet points, explain the key factors behind this specific price. Be specific with numbers — compare to typical values for this brand/model. No generic filler."""


def generate_explanation(
    car_dict: Dict[str, Any],
    predicted_price: float,
    hf_token: str = None,
) -> str:
    """
    Call the Qwen LLM via Hugging Face's serverless Inference API.

    No local GPU, no model download — the request is processed on HF servers.
    Requires a Hugging Face token (free tier is sufficient).
    """
    from huggingface_hub import InferenceClient

    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        return "(Explanation unavailable — set HF_TOKEN to enable LLM explanations)"

    client = InferenceClient(token=token)

    prompt = PROMPT_TEMPLATE.format(
        brand=car_dict.get("brand", "Unknown"),
        model=car_dict.get("model", "Unknown"),
        year=car_dict.get("year", "N/A"),
        mileage=car_dict.get("mileage", "N/A"),
        engine=car_dict.get("engineSize", "N/A"),
        fuel=car_dict.get("fuelType", "N/A"),
        transmission=car_dict.get("transmission", "N/A"),
        price=f"{predicted_price:,.2f}",
    )

    messages = [
        {"role": "system", "content": "You are a used car pricing expert. Give short, specific, data-driven answers. No headers or preamble."},
        {"role": "user", "content": prompt},
    ]

    response = client.chat_completion(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=messages,
        max_tokens=400,
    )

    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def predict_and_explain(
    car_dict: Dict[str, Any],
    model,
    scaler,
    feature_columns,
    explain: bool = True,
    hf_token: str = None,
) -> Dict[str, Any]:
    """
    Predict price locally (CPU) and optionally get an LLM explanation
    via the HF Inference API (serverless, no local GPU).
    """
    X = preprocess_single(car_dict, feature_columns, scaler)
    log_price = model.predict(X, verbose=0)[0][0]
    predicted_price = float(np.expm1(log_price))

    result: Dict[str, Any] = {"predicted_price": round(predicted_price, 2)}

    if explain:
        result["explanation_text"] = generate_explanation(
            car_dict, predicted_price, hf_token=hf_token
        )

    return result


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Predict used-car price and get an AI explanation."
    )
    parser.add_argument("--repo", type=str, required=True,
                        help="HF repo id, e.g. username/uk-used-car-nn")
    parser.add_argument("--brand", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, dest="car_model")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--mileage", type=int, required=True)
    parser.add_argument("--engineSize", type=float, required=True)
    parser.add_argument("--fuelType", type=str, required=True)
    parser.add_argument("--transmission", type=str, required=True)
    parser.add_argument("--no-explain", action="store_true", default=False,
                        help="Skip the LLM explanation.")
    parser.add_argument("--token", type=str, default=None,
                        help="HF token (or set HF_TOKEN env var).")
    args = parser.parse_args()

    car_dict = {
        "brand": args.brand,
        "model": args.car_model,
        "year": args.year,
        "mileage": args.mileage,
        "engineSize": args.engineSize,
        "fuelType": args.fuelType,
        "transmission": args.transmission,
    }

    print("=" * 60)
    print("  UK Used Car Valuation System")
    print("=" * 60)

    print(f"\n  Loading model from HF: {args.repo} ...")
    nn_model, scaler, feature_columns = load_model_from_hf(
        repo_id=args.repo, token=args.token
    )

    print("  Predicting ...")
    result = predict_and_explain(
        car_dict=car_dict,
        model=nn_model,
        scaler=scaler,
        feature_columns=feature_columns,
        explain=not args.no_explain,
        hf_token=args.token or os.environ.get("HF_TOKEN"),
    )

    print(f"\n  Predicted Price : £{result['predicted_price']:,.2f}")
    if "explanation_text" in result:
        print(f"\n  Explanation:\n  {result['explanation_text']}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
