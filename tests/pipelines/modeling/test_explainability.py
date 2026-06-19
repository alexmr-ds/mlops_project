"""Unit tests for SHAP-based feature explainability."""

import unittest

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestClassifier

from mlops_project.modeling import explainability as modeling_explainability
from mlops_project.modeling.model_bundle import ModelBundle
from mlops_project.modeling.preprocessing import ModelPreprocessor


def _make_model_bundle(n_rows: int = 40) -> tuple[ModelBundle, pd.DataFrame]:
    """Fit a minimal ModelBundle and return it with a matching test set."""
    feature_names = ["ph", "Hardness", "Solids", "Conductivity", "Sulfate"]
    rng = np.random.default_rng(73)

    X = pd.DataFrame(rng.uniform(0, 10, size=(n_rows, len(feature_names))), columns=feature_names)
    y = pd.Series([0, 1] * (n_rows // 2), name="Potability")

    preprocessing_parameters = {
        "zscore_threshold": 99.0,
        "rfe": {
            "enabled": False,
            "step": 1,
            "cv_folds": 2,
            "scoring": "roc_auc",
            "n_jobs": 1,
            "logistic_max_iter": 100,
            "random_state": 73,
        },
    }

    preprocessor = ModelPreprocessor(preprocessing_parameters)
    X_filtered, y_filtered = preprocessor.fit_resample(X, y)

    estimator = RandomForestClassifier(n_estimators=5, random_state=73)
    estimator.fit(X_filtered, y_filtered)

    bundle = ModelBundle(
        model_name="random_forest",
        preprocessor=preprocessor,
        estimator=estimator,
    )

    X_test = pd.DataFrame(
        rng.uniform(0, 10, size=(8, len(feature_names))), columns=feature_names
    )
    return bundle, X_test


class ComputeShapValuesTests(unittest.TestCase):
    """Tests for compute_shap_values."""

    def setUp(self) -> None:
        self.bundle, self.X_test = _make_model_bundle()

    def test_shap_values_shape_matches_transformed_test_data(self) -> None:
        shap_values, summary = modeling_explainability.compute_shap_values(
            self.bundle, self.X_test
        )

        X_transformed = self.bundle.preprocessor.transform(self.X_test)
        self.assertEqual(shap_values.shape, (len(self.X_test), X_transformed.shape[1]))

    def test_summary_frame_has_expected_columns(self) -> None:
        _, summary = modeling_explainability.compute_shap_values(self.bundle, self.X_test)

        self.assertListEqual(summary.columns.tolist(), ["feature", "mean_abs_shap"])

    def test_summary_frame_has_one_row_per_selected_feature(self) -> None:
        _, summary = modeling_explainability.compute_shap_values(self.bundle, self.X_test)

        X_transformed = self.bundle.preprocessor.transform(self.X_test)
        self.assertEqual(len(summary), X_transformed.shape[1])

    def test_summary_frame_is_sorted_by_importance_descending(self) -> None:
        _, summary = modeling_explainability.compute_shap_values(self.bundle, self.X_test)

        values = summary["mean_abs_shap"].tolist()
        self.assertEqual(values, sorted(values, reverse=True))

    def test_mean_abs_shap_values_are_non_negative(self) -> None:
        _, summary = modeling_explainability.compute_shap_values(self.bundle, self.X_test)

        self.assertTrue((summary["mean_abs_shap"] >= 0).all())


class CreateShapSummaryPlotTests(unittest.TestCase):
    """Tests for create_shap_summary_plot."""

    def setUp(self) -> None:
        bundle, X_test = _make_model_bundle()
        self.shap_values, self.shap_summary = modeling_explainability.compute_shap_values(
            bundle, X_test
        )

    def test_returns_matplotlib_figure(self) -> None:
        plot = modeling_explainability.create_shap_summary_plot(
            self.shap_values, self.shap_summary
        )

        self.assertIsInstance(plot, Figure)

    def test_plot_has_correct_axis_labels(self) -> None:
        plot = modeling_explainability.create_shap_summary_plot(
            self.shap_values, self.shap_summary
        )

        ax = plot.axes[0]
        self.assertIn("SHAP", ax.get_title())
        self.assertIn("SHAP", ax.get_xlabel())


if __name__ == "__main__":
    unittest.main()
