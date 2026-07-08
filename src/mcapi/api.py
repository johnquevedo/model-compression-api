"""FastAPI inference service for the compressed model.

Configuration via environment variables (read at startup):
  MCAPI_MODEL_DIR   path to the served model directory (default: artifacts/student-quantized)
  MCAPI_BACKEND     torch | quantized | onnx        (default: quantized)
  MCAPI_MAX_LENGTH  tokenizer max length            (default: 128)
  MCAPI_QUANT_DTYPE qint8 | quint8                  (default: qint8)

Run with:  uvicorn mcapi.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .predictor import Predictor

app = FastAPI(
    title="Model Compression & Inference API",
    version="0.1.0",
    description="Serve a distilled + quantized text classifier with per-request latency.",
)

_predictor: Optional[Predictor] = None


class PredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, description="One or more input texts.")


class Prediction(BaseModel):
    label: str
    label_id: int
    score: float
    probabilities: List[float]


class PredictResponse(BaseModel):
    predictions: List[Prediction]
    latency_ms: float
    backend: str
    batch_size: int


def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        _predictor = Predictor(
            model_dir=os.environ.get("MCAPI_MODEL_DIR", "artifacts/student-quantized"),
            backend=os.environ.get("MCAPI_BACKEND", "quantized"),
            max_length=int(os.environ.get("MCAPI_MAX_LENGTH", "128")),
            quant_dtype=os.environ.get("MCAPI_QUANT_DTYPE", "qint8"),
        )
    return _predictor


@app.on_event("startup")
def _startup() -> None:
    # Fail fast (and warm the model) if the artifact is missing/misconfigured.
    try:
        get_predictor()
    except Exception as exc:  # pragma: no cover - startup diagnostics
        print(f"[api] WARNING: could not load model at startup: {exc}")


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "model_loaded": _predictor is not None}


@app.get("/info")
def info() -> dict:
    p = get_predictor()
    return {
        "model_dir": p.model_dir,
        "backend": p.backend,
        "device": str(p.device),
        "max_length": p.max_length,
        "labels": p.id2label,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    p = get_predictor()
    try:
        t0 = time.perf_counter()
        preds = p.predict(req.texts)
        latency_ms = (time.perf_counter() - t0) * 1000.0
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return PredictResponse(
        predictions=[Prediction(**pr) for pr in preds],
        latency_ms=latency_ms,
        backend=p.backend,
        batch_size=len(req.texts),
    )
