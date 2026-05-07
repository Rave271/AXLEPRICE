"""app.py — FastAPI server for the AxlePrice website.

This provides a small, proper website frontend:
- Home:        GET /
- Valuation:   GET /valuation
- About:       GET /about

API:
- POST /api/valuate  -> {predicted_price, explanation?, input}

Run (dev):
    uvicorn app:app --reload --port 8000
"""

from __future__ import annotations

import datetime as _dt
import os
import secrets
import time
from functools import lru_cache
from typing import Any, Dict, Optional

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.requests import Request

from adapter import load_model_from_hf
from utils import preprocess_single


load_dotenv()

DEFAULT_REPO_ID = "Rave271/uk-used-car-nn"
LLM_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
CURRENT_YEAR = _dt.date.today().year
SESSION_TTL_SECONDS = 60 * 60
SESSION_COOKIE = "axleprice_session"

_SESSIONS: Dict[str, Dict[str, Any]] = {}

_USERS = {
    "raghav": "pass",
}

_SAMPLE_HISTORY = [
    {
        "brand": "BMW",
        "model": "3 Series",
        "year": 2019,
        "mileage": 30000,
        "engineSize": 2.0,
        "fuelType": "Petrol",
        "transmission": "Automatic",
        "predicted_price": 19678.77,
        "confidence": 82.4,
    },
    {
        "brand": "Audi",
        "model": "A3",
        "year": 2018,
        "mileage": 42000,
        "engineSize": 1.6,
        "fuelType": "Diesel",
        "transmission": "Manual",
        "predicted_price": 15420.55,
        "confidence": 79.1,
    },
    {
        "brand": "Mercedes",
        "model": "C Class",
        "year": 2020,
        "mileage": 21000,
        "engineSize": 2.0,
        "fuelType": "Hybrid",
        "transmission": "Automatic",
        "predicted_price": 26305.2,
        "confidence": 88.7,
    },
]


class ValuationRequest(BaseModel):
    brand: str = Field(min_length=1, max_length=80)
    model: str = Field(min_length=1, max_length=120)
    year: int = Field(ge=1990, le=CURRENT_YEAR)
    mileage: int = Field(ge=0, le=500_000)
    engineSize: float = Field(ge=0, le=8)
    fuelType: str = Field(min_length=1, max_length=30)
    transmission: str = Field(min_length=1, max_length=30)
    repoId: Optional[str] = None
    hfToken: Optional[str] = None
    explain: bool = True


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=1, max_length=120)


def _clean_text(value: str) -> str:
    return (value or "").strip()


def _create_session(username: str) -> str:
    token = secrets.token_urlsafe(24)
    _SESSIONS[token] = {
        "username": username,
        "expires": time.time() + SESSION_TTL_SECONDS,
        "history": list(_SAMPLE_HISTORY),
    }
    return token


def _get_session(request: Request) -> Optional[Dict[str, Any]]:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    session = _SESSIONS.get(token)
    if not session:
        return None
    if session["expires"] < time.time():
        _SESSIONS.pop(token, None)
        return None
    return session


def _require_login(request: Request) -> Dict[str, Any]:
    session = _get_session(request)
    if not session:
        raise HTTPException(status_code=401, detail="Login required.")
    return session


@lru_cache(maxsize=4)
def _load_model(repo_id: str, token: str):
    # Cache per repo/token combo (token may be empty string)
    return load_model_from_hf(repo_id=repo_id, token=(token or None))


def _predict_price(*, model_bundle: Any, car_dict: Dict[str, Any]) -> float:
    nn_model, scaler, feature_columns, numeric_idx, model_encoder, reference_year = model_bundle

    X = preprocess_single(
        car_dict,
        feature_columns,
        scaler,
        numeric_idx=numeric_idx,
        model_encoder=model_encoder,
        reference_year=reference_year,
    )

    # Use direct model call instead of .predict() to avoid TF threading hang
    log_price = float(nn_model(X, training=False).numpy().flatten()[0])
    return float(np.expm1(log_price))


def _generate_explanation(*, car_dict: Dict[str, Any], predicted_price: float, hf_token: str) -> str:
    from huggingface_hub import InferenceClient

    token = _clean_text(hf_token) or os.environ.get("HF_TOKEN")
    if not token:
        return "(Explanation unavailable — set HF_TOKEN to enable it.)"

    client = InferenceClient(token=token, timeout=60)
    prompt = (
        f"A {car_dict['year']} {car_dict['brand']} {car_dict['model']} with {car_dict['mileage']:,} miles, "
        f"{car_dict['engineSize']}L {car_dict['fuelType']} engine, {car_dict['transmission']} transmission "
        f"is valued at £{predicted_price:,.2f}.\n\n"
        "In 3-4 concise bullet points, explain the key factors behind this specific price. "
        "Be specific with numbers — compare to typical values for this brand/model. No generic filler."
    )

    messages = [
        {
            "role": "system",
            "content": "You are a used car pricing expert. Give short, specific, data-driven answers. No headers or preamble.",
        },
        {"role": "user", "content": prompt},
    ]

    response = client.chat_completion(
        model=LLM_MODEL_ID,
        messages=messages,
        max_tokens=400,
    )
    return response.choices[0].message.content.strip()


def _confidence_score(*, car_dict: Dict[str, Any]) -> float:
    age = max(CURRENT_YEAR - int(car_dict.get("year", CURRENT_YEAR)), 0)
    mileage = max(float(car_dict.get("mileage", 0)), 0.0)
    mileage_per_year = mileage / max(age, 1)

    score = 0.88

    if age >= 18:
        score -= 0.20
    elif age >= 10:
        score -= 0.12
    elif age >= 6:
        score -= 0.06

    if mileage_per_year > 22000:
        score -= 0.12
    elif mileage_per_year < 1500:
        score -= 0.06

    engine = float(car_dict.get("engineSize", 0))
    if engine >= 4.5:
        score -= 0.05

    fuel = (car_dict.get("fuelType") or "").strip().lower()
    if fuel not in {"petrol", "diesel", "hybrid", "electric"}:
        score -= 0.04

    return max(0.55, min(0.95, score))


app = FastAPI(title="AxlePrice")

app.mount("/static", StaticFiles(directory="web/static"), name="static")
templates = Jinja2Templates(directory="web/templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "request": request,
            "title": "AxlePrice",
            "active_page": "home",
            "is_logged_in": bool(_get_session(request)),
        },
    )


@app.get("/valuation", response_class=HTMLResponse)
def valuation_page(request: Request):
    if not _get_session(request):
        return RedirectResponse(url="/login")
    return templates.TemplateResponse(
        request,
        "valuation.html",
        {
            "request": request,
            "title": "Valuation | AxlePrice",
            "active_page": "valuation",
            "current_year": CURRENT_YEAR,
            "default_repo_id": DEFAULT_REPO_ID,
            "is_logged_in": True,
        },
    )


@app.get("/about", response_class=HTMLResponse)
def about(request: Request):
    return templates.TemplateResponse(
        request,
        "about.html",
        {
            "request": request,
            "title": "About | AxlePrice",
            "active_page": "about",
            "is_logged_in": bool(_get_session(request)),
        },
    )


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    if _get_session(request):
        return RedirectResponse(url="/valuation")
    return templates.TemplateResponse(
        request,
        "login.html",
        {
            "request": request,
            "title": "Login | AxlePrice",
            "active_page": "login",
            "is_logged_in": False,
        },
    )


@app.post("/api/login")
def login(req: LoginRequest, response: Response):
    username = _clean_text(req.username)
    password = _clean_text(req.password)
    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password required.")
    if _USERS.get(username) != password:
        raise HTTPException(status_code=401, detail="Invalid credentials.")

    token = _create_session(username)
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        samesite="lax",
    )
    return {"ok": True}


@app.post("/api/logout")
def logout(request: Request, response: Response):
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        _SESSIONS.pop(token, None)
    response.delete_cookie(SESSION_COOKIE)
    return {"ok": True}


@app.get("/logout")
def logout_page(request: Request):
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        _SESSIONS.pop(token, None)
    response = RedirectResponse(url="/login")
    response.delete_cookie(SESSION_COOKIE)
    return response


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/history")
def history(request: Request):
    session = _require_login(request)
    return {"items": session.get("history", [])}


@app.get("/api/history/export")
def export_history(request: Request):
    session = _require_login(request)
    items = session.get("history", [])

    output = [
        "brand,model,year,mileage,engineSize,fuelType,transmission,predicted_price,confidence"
    ]
    for item in items:
        output.append(
            "{brand},{model},{year},{mileage},{engineSize},{fuelType},{transmission},{predicted_price},{confidence}".format(
                brand=str(item.get("brand", "")),
                model=str(item.get("model", "")),
                year=item.get("year", ""),
                mileage=item.get("mileage", ""),
                engineSize=item.get("engineSize", ""),
                fuelType=str(item.get("fuelType", "")),
                transmission=str(item.get("transmission", "")),
                predicted_price=item.get("predicted_price", ""),
                confidence=item.get("confidence", ""),
            )
        )

    csv_data = "\n".join(output)
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=axleprice_history.csv"},
    )


@app.post("/api/valuate")
def valuate(request: Request, req: ValuationRequest):
    session = _require_login(request)
    repo_id = _clean_text(req.repoId) or DEFAULT_REPO_ID
    hf_token = _clean_text(req.hfToken)

    car_dict = {
        "brand": _clean_text(req.brand),
        "model": _clean_text(req.model),
        "year": int(req.year),
        "mileage": int(req.mileage),
        "engineSize": float(req.engineSize),
        "fuelType": _clean_text(req.fuelType),
        "transmission": _clean_text(req.transmission),
    }

    if not car_dict["brand"] or not car_dict["model"]:
        raise HTTPException(status_code=400, detail="Brand and model are required.")

    try:
        model_bundle = _load_model(repo_id, hf_token)
        predicted_price = _predict_price(model_bundle=model_bundle, car_dict=car_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    explanation: Optional[str] = None
    if bool(req.explain):
        try:
            explanation = _generate_explanation(
                car_dict=car_dict,
                predicted_price=predicted_price,
                hf_token=hf_token,
            )
        except Exception as e:
            explanation = f"(Explanation failed: {e})"

    confidence = _confidence_score(car_dict=car_dict)

    history_item = {
        **car_dict,
        "predicted_price": round(float(predicted_price), 2),
        "confidence": round(confidence * 100, 1),
    }
    session.setdefault("history", []).insert(0, history_item)

    return {
        "predicted_price": round(float(predicted_price), 2),
        "confidence": round(confidence * 100, 1),
        "explanation": explanation,
        "input": car_dict,
    }
