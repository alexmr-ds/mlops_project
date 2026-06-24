"""Unit tests for the FastAPI serving layer."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import HTTPException
from sklearn.ensemble import RandomForestClassifier

from mlops_project.modeling.model_bundle import ModelBundle
from mlops_project.modeling.preprocessing import ModelPreprocessor
from mlops_project.pipelines.preprocessing.nodes import _engineer_feature_frame
from mlops_project.serving import app as serving_app


def _make_fitted_bundle() -> ModelBundle:
    """Fit a minimal ModelBundle on engineered features, like the real one."""
    rng = np.random.default_rng(11)
    raw_columns = [
        "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
        "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity",
    ]
    raw = pd.DataFrame(rng.uniform(1, 100, size=(40, len(raw_columns))), columns=raw_columns)
    X = _engineer_feature_frame(raw)
    y = pd.Series([0, 1] * (len(X) // 2), name="Potability")

    preprocessing_parameters = {
        "zscore_threshold": 99.0,
        "rfe": {
            "enabled": False,
            "step": 1,
            "cv_folds": 2,
            "scoring": "roc_auc",
            "n_jobs": 1,
            "logistic_max_iter": 100,
            "random_state": 11,
        },
    }
    preprocessor = ModelPreprocessor(preprocessing_parameters)
    X_filtered, y_filtered = preprocessor.fit_resample(X, y)

    estimator = RandomForestClassifier(n_estimators=5, random_state=11)
    estimator.fit(X_filtered, y_filtered)

    return ModelBundle(model_name="random_forest", preprocessor=preprocessor, estimator=estimator)


class PredictEndpointTests(unittest.TestCase):
    """Tests for the /predict handler, called directly as a plain function."""

    def setUp(self) -> None:
        self._original_model = serving_app._model
        serving_app._model = _make_fitted_bundle()

    def tearDown(self) -> None:
        serving_app._model = self._original_model

    def _full_sample(self) -> serving_app.WaterSample:
        return serving_app.WaterSample(
            ph=7.0,
            Hardness=200.0,
            Solids=15000.0,
            Chloramines=7.0,
            Sulfate=330.0,
            Conductivity=420.0,
            Organic_carbon=14.0,
            Trihalomethanes=70.0,
            Turbidity=4.0,
        )

    def test_full_sample_returns_valid_prediction(self) -> None:
        response = serving_app.predict(self._full_sample())

        self.assertIn(response.prediction, (0, 1))
        self.assertTrue(0.0 <= response.probability <= 1.0)

    def test_omitted_nullable_fields_does_not_500(self) -> None:
        sample = serving_app.WaterSample(
            Hardness=200.0,
            Solids=15000.0,
            Chloramines=7.0,
            Conductivity=420.0,
            Organic_carbon=14.0,
            Turbidity=4.0,
        )

        response = serving_app.predict(sample)

        self.assertIn(response.prediction, (0, 1))
        self.assertTrue(0.0 <= response.probability <= 1.0)

    def test_explicit_null_nullable_fields_does_not_500(self) -> None:
        sample = serving_app.WaterSample(
            ph=None,
            Hardness=200.0,
            Solids=15000.0,
            Chloramines=7.0,
            Sulfate=None,
            Conductivity=420.0,
            Organic_carbon=14.0,
            Trihalomethanes=None,
            Turbidity=4.0,
        )

        response = serving_app.predict(sample)

        self.assertIn(response.prediction, (0, 1))
        self.assertTrue(0.0 <= response.probability <= 1.0)

    def test_predict_raises_503_when_model_not_loaded(self) -> None:
        serving_app._model = None

        with self.assertRaises(HTTPException) as raised:
            serving_app.predict(self._full_sample())

        self.assertEqual(raised.exception.status_code, 503)


class HealthEndpointTests(unittest.TestCase):
    """Tests for the /health handler."""

    def test_health_returns_ok_status_and_model_name(self) -> None:
        response = serving_app.health()

        self.assertEqual(response, {"status": "ok", "model": "extra_trees"})


class LoadModelTests(unittest.TestCase):
    """Tests for _load_model's startup loading behaviour."""

    def test_raises_file_not_found_when_model_artifact_is_missing(self) -> None:
        original_path = serving_app._MODEL_PATH
        serving_app._MODEL_PATH = Path("data/06_models/does_not_exist.pkl")
        try:
            with self.assertRaisesRegex(FileNotFoundError, "Run 'kedro run --pipeline=modeling'"):
                serving_app._load_model()
        finally:
            serving_app._MODEL_PATH = original_path

    def test_loads_the_persisted_model_bundle(self) -> None:
        loaded = serving_app._load_model()

        self.assertTrue(hasattr(loaded, "predict"))
        self.assertTrue(hasattr(loaded, "predict_proba"))


if __name__ == "__main__":
    unittest.main()
