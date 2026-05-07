# AxlePrice

AxlePrice is a two-stage system for used car price valuation. Heavy training runs
on Kaggle, while the local app serves the website and runs fast CPU-only inference
using the tiny model from Hugging Face.

## Quick Start

1. Create a virtual environment and install dependencies:

   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. (Optional) set your Hugging Face token for explanations:

   ```bash
   echo 'HF_TOKEN=hf_your_token_here' > .env
   ```

3. Run the web app:

   ```bash
   python -m uvicorn app:app --reload --port 8000
   ```

Open:
- http://127.0.0.1:8000
- http://127.0.0.1:8000/valuation

## Training (Kaggle)

Use `train_notebook.ipynb` on Kaggle to train and push the model artefacts to
your Hugging Face repo (model, scaler, config). You only need to do this when
retraining.

## Project Structure

- `app.py` FastAPI server
- `inference.py` CLI runner
- `adapter.py` Hugging Face model fetcher
- `utils.py` Shared preprocessing
- `train_notebook.ipynb` Kaggle training notebook
- `requirements.txt` Local dependencies
- `web/` HTML templates + static assets

## Dataset

- Name: 100,000 UK Used Car Data Set (Aditya)
- Source: https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes

## Render Deploy (Web Service)

- Root directory: repo root (with `app.py` and `requirements.txt`)
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- Environment variables:
  - `HF_TOKEN` (optional, only for explanations)

## Notes

- No GPU is required locally.
- AI explanations use the Hugging Face Inference API when `HF_TOKEN` is set.
