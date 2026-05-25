"""Prediction-ready bundles for persisted trained models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from mlops_project.modeling import preprocessing


@dataclass
class ModelBundle:
    """Bundle a fitted estimator with its fitted learned preprocessing."""

    model_name: str
    preprocessor: preprocessing.ModelPreprocessor
    estimator: Any

    def __post_init__(self) -> None:
        """Validate the persisted bundle interface."""
        if not isinstance(self.model_name, str) or not self.model_name:
            raise ValueError("model_name must be a non-empty string.")
        if not isinstance(self.preprocessor, preprocessing.ModelPreprocessor):
            raise ValueError("preprocessor must be a ModelPreprocessor.")
        if not hasattr(self.estimator, "predict"):
            raise ValueError("estimator must expose predict.")
        if not hasattr(self.estimator, "predict_proba"):
            raise ValueError("estimator must expose predict_proba.")
        if not hasattr(self.estimator, "get_params"):
            raise ValueError("estimator must expose get_params.")

    @property
    def selected_features(self) -> list[str]:
        """Return final selected feature names."""
        return list(self.preprocessor.selected_features)

    @property
    def filtered_training_row_count(self) -> int:
        """Return the number of rows used after final outlier filtering."""
        return self.preprocessor.filtered_training_row_count

    @property
    def classes_(self) -> np.ndarray:
        """Return fitted estimator classes for sklearn compatibility."""
        return self.estimator.classes_

    def transform_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted learned preprocessing to engineered features."""
        return self.preprocessor.transform(features)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict labels from engineered features."""
        return self.estimator.predict(self.transform_features(features))

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities from engineered features."""
        return self.estimator.predict_proba(self.transform_features(features))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return estimator parameters for logging compatibility."""
        return self.estimator.get_params(deep=deep)
