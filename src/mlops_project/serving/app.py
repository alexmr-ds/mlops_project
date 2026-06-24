"""REST API for water potability predictions.

The best-performing model (Random Forest) is loaded once at startup from its
persisted ModelBundle.  Each prediction request passes the raw water quality
measurements through the bundle, which handles imputation, scaling, and feature
selection internally before calling the estimator.

Run locally:
    uv run uvicorn mlops_project.serving.app:app --reload

With Docker:
    docker compose up
"""

from __future__ import annotations

import pickle
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Feature engineering must be applied to raw measurements before the model bundle
# can preprocess and predict — the bundle's internal preprocessor was fitted on
# the 32-column engineered frame, not the original 9 water quality measurements.
from mlops_project.pipelines.preprocessing.nodes import _engineer_feature_frame

# ---------------------------------------------------------------------------
# Model loading — done once at import time so every request reuses the same
# fitted bundle without the overhead of re-deserialising from disk.
# ---------------------------------------------------------------------------

_MODEL_PATH = Path("data/06_models/random_forest_model.pkl")
_model: Any = None


def _load_model() -> Any:
    if not _MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {_MODEL_PATH}. "
            "Run 'kedro run --pipeline=modeling' first to train and persist the model."
        )
    with _MODEL_PATH.open("rb") as f:
        return pickle.load(f)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    global _model
    _model = _load_model()
    yield


app = FastAPI(
    title="Water Potability API",
    description=(
        "Predicts whether a water sample is potable (safe to drink) "
        "using a tuned Random Forest classifier trained on the Kaggle "
        "Water Potability dataset."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class WaterSample(BaseModel):
    """Raw water quality measurements for one sample.

    The three nullable fields (ph, Sulfate, Trihalomethanes) may be omitted
    when the measurement is unavailable — the model's imputer will fill them
    with the training-set mean automatically.
    """

    ph: float | None = Field(default=None, description="pH level (0–14)")
    Hardness: float = Field(description="Water hardness in mg/L")
    Solids: float = Field(description="Total dissolved solids in ppm")
    Chloramines: float = Field(description="Chloramines concentration in ppm")
    Sulfate: float | None = Field(default=None, description="Sulfate concentration in mg/L")
    Conductivity: float = Field(description="Electrical conductivity in μS/cm")
    Organic_carbon: float = Field(description="Organic carbon content in ppm")
    Trihalomethanes: float | None = Field(default=None, description="Trihalomethanes in μg/L")
    Turbidity: float = Field(description="Water turbidity in NTU")

    model_config = {"json_schema_extra": {
        "example": {
            "ph": 7.0,
            "Hardness": 204.8,
            "Solids": 20791.3,
            "Chloramines": 7.3,
            "Sulfate": 368.5,
            "Conductivity": 564.3,
            "Organic_carbon": 10.4,
            "Trihalomethanes": 86.9,
            "Turbidity": 2.96,
        }
    }}


class PredictionResponse(BaseModel):
    prediction: int = Field(description="0 = not potable, 1 = potable")
    probability: float = Field(description="Probability that the sample is potable")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check — confirms the model is loaded and the API is running."""
    return {"status": "ok", "model": "random_forest"}


@app.post("/predict", response_model=PredictionResponse)
def predict(sample: WaterSample) -> PredictionResponse:
    """Predict whether the given water sample is potable.

    The input features are passed through the model bundle's fitted preprocessing
    pipeline (outlier-aware imputation, scaling, feature selection) before the
    Random Forest estimator produces its prediction.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # Build a single-row feature frame from the raw measurements. When a
    # nullable field (ph, Sulfate, Trihalomethanes) is omitted, model_dump()
    # produces a None, which makes pandas infer an object dtype for that
    # column -- coerce everything to numeric (None -> NaN) before feature
    # engineering so the bundle's preprocessor sees float columns throughout.
    raw_frame = pd.DataFrame([sample.model_dump()]).apply(pd.to_numeric, errors="coerce")
    feature_frame = _engineer_feature_frame(raw_frame)

    prediction = int(_model.predict(feature_frame)[0])
    probability = float(_model.predict_proba(feature_frame)[0][1])

    return PredictionResponse(prediction=prediction, probability=round(probability, 4))
