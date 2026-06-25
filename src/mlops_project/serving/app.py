"""REST API for water potability predictions.

The best-performing model (Extra Trees) is loaded once at startup from its
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

# Feature engineering must run on the raw measurements before the model bundle
# can preprocess and predict: the bundle's internal preprocessor was fitted on
# the 32-column engineered frame, not the original 9 water quality measurements.
from mlops_project.pipelines.preprocessing.nodes import _engineer_feature_frame

# Champion selection is a deliberate manual promotion gate, not derived at
# runtime. Extra Trees is the top-ranked model in data/08_reporting/
# model_comparison.csv, so it is promoted here on purpose. The same manual
# choice is mirrored by the drift and SHAP pipeline inputs. If a re-tune
# re-ranks the models, re-promote by changing CHAMPION_MODEL_NAME here and the
# matching catalog inputs; tests/test_champion_promotion.py fails fast if this
# name ever drifts from the top of the committed comparison table.
CHAMPION_MODEL_NAME = "extra_trees"

# Loaded once in the lifespan startup hook below so every request reuses the
# same fitted bundle instead of re-deserialising it from disk each time.
_MODEL_PATH = Path(f"data/06_models/{CHAMPION_MODEL_NAME}_model.pkl")
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
        "using a tuned Extra Trees classifier trained on the Kaggle "
        "Water Potability dataset."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


class WaterSample(BaseModel):
    """Raw water quality measurements for one sample.

    ph, Sulfate, and Trihalomethanes may be omitted when the measurement is
    unavailable; the model's imputer fills them in with the training-set mean.
    """

    ph: float | None = Field(default=None, description="pH level (0-14)")
    Hardness: float = Field(description="Water hardness in mg/L")
    Solids: float = Field(description="Total dissolved solids in ppm")
    Chloramines: float = Field(description="Chloramines concentration in ppm")
    Sulfate: float | None = Field(default=None, description="Sulfate concentration in mg/L")
    Conductivity: float = Field(description="Electrical conductivity in uS/cm")
    Organic_carbon: float = Field(description="Organic carbon content in ppm")
    Trihalomethanes: float | None = Field(default=None, description="Trihalomethanes in ug/L")
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


@app.get("/health")
def health() -> dict[str, str]:
    """Confirm the model is loaded and the API is up."""
    return {"status": "ok", "model": CHAMPION_MODEL_NAME}


@app.post("/predict", response_model=PredictionResponse)
def predict(sample: WaterSample) -> PredictionResponse:
    """Predict whether the given water sample is potable.

    Runs the input through the model bundle's fitted preprocessing pipeline
    (outlier-aware imputation, scaling, feature selection) before the Extra
    Trees estimator produces its prediction.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # model_dump() turns an omitted/null ph, Sulfate, or Trihalomethanes into a
    # Python None, and a single-row DataFrame with a None infers object dtype
    # for that column. Coerce to numeric first (None becomes NaN) so the
    # engineered frame stays float throughout and the imputer can fill the gap.
    raw_frame = pd.DataFrame([sample.model_dump()]).apply(pd.to_numeric, errors="coerce")
    feature_frame = _engineer_feature_frame(raw_frame)

    prediction = int(_model.predict(feature_frame)[0])
    probability = float(_model.predict_proba(feature_frame)[0][1])

    return PredictionResponse(prediction=prediction, probability=round(probability, 4))
