"""Unit tests for data drift detection nodes."""

import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from mlops_project.modeling.evaluation import METRIC_NAMES
from mlops_project.modeling.model_bundle import ModelBundle
from mlops_project.modeling.preprocessing import ModelPreprocessor
from mlops_project.pipelines.data_drift import nodes
from mlops_project.pipelines.preprocessing.nodes import _engineer_feature_frame


def _make_identical_splits(n_rows: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two splits sampled from the same distribution; expect no drift."""
    rng = np.random.default_rng(73)
    features = ["ph", "Hardness", "Solids"]
    X_train = pd.DataFrame(rng.normal(5.0, 1.0, size=(n_rows, len(features))), columns=features)
    X_test = pd.DataFrame(rng.normal(5.0, 1.0, size=(n_rows, len(features))), columns=features)
    return X_train, X_test


def _parameters(threshold: float = 0.05) -> dict:
    return {"significance_threshold": threshold}


class DetectFeatureDriftTests(unittest.TestCase):
    """Tests for the detect_feature_drift node."""

    def test_report_has_expected_columns(self) -> None:
        X_train, X_test = _make_identical_splits()
        report = nodes.detect_feature_drift(X_train, X_test, _parameters())

        self.assertListEqual(
            report.columns.tolist(), ["feature", "ks_statistic", "p_value", "drifted"]
        )

    def test_report_has_one_row_per_feature(self) -> None:
        X_train, X_test = _make_identical_splits()
        report = nodes.detect_feature_drift(X_train, X_test, _parameters())

        self.assertEqual(len(report), len(X_train.columns))
        self.assertSetEqual(set(report["feature"]), set(X_train.columns))

    def test_no_drift_on_same_distribution(self) -> None:
        X_train, X_test = _make_identical_splits(n_rows=200)
        report = nodes.detect_feature_drift(X_train, X_test, _parameters(threshold=0.05))

        # With the same underlying distribution and large samples, all p-values should be high
        self.assertFalse(report["drifted"].any(), msg=report.to_string())

    def test_drift_detected_on_shifted_distribution(self) -> None:
        X_train, X_test = _make_identical_splits(n_rows=200)
        # Shift one feature by a large amount to guarantee the KS test picks it up
        X_test = X_test.copy()
        X_test["Hardness"] = X_test["Hardness"] + 100.0

        report = nodes.detect_feature_drift(X_train, X_test, _parameters(threshold=0.05))

        hardness_row = report.loc[report["feature"] == "Hardness"].iloc[0]
        self.assertTrue(hardness_row["drifted"])
        self.assertLess(hardness_row["p_value"], 0.05)

    def test_non_shifted_features_are_not_flagged(self) -> None:
        X_train, X_test = _make_identical_splits(n_rows=200)
        X_test = X_test.copy()
        X_test["Hardness"] = X_test["Hardness"] + 100.0

        report = nodes.detect_feature_drift(X_train, X_test, _parameters(threshold=0.05))

        non_drifted = report.loc[report["feature"] != "Hardness"]
        self.assertFalse(non_drifted["drifted"].any(), msg=non_drifted.to_string())

    def test_ks_statistics_are_between_zero_and_one(self) -> None:
        X_train, X_test = _make_identical_splits()
        report = nodes.detect_feature_drift(X_train, X_test, _parameters())

        self.assertTrue((report["ks_statistic"] >= 0).all())
        self.assertTrue((report["ks_statistic"] <= 1).all())

    def test_p_values_are_between_zero_and_one(self) -> None:
        X_train, X_test = _make_identical_splits()
        report = nodes.detect_feature_drift(X_train, X_test, _parameters())

        self.assertTrue((report["p_value"] >= 0).all())
        self.assertTrue((report["p_value"] <= 1).all())

    def test_significance_threshold_controls_drift_flag(self) -> None:
        """A very strict threshold (0) flags everything; a lax one (1) flags nothing."""
        X_train, X_test = _make_identical_splits(n_rows=50)

        report_strict = nodes.detect_feature_drift(X_train, X_test, _parameters(threshold=0.0))
        report_lax = nodes.detect_feature_drift(X_train, X_test, _parameters(threshold=1.0))

        self.assertFalse(report_strict["drifted"].any())
        self.assertTrue(report_lax["drifted"].all())


def _make_engineered_frame(n_rows: int = 60, seed: int = 5) -> pd.DataFrame:
    """Raw measurements run through the real feature engineering, like X_test."""
    rng = np.random.default_rng(seed)
    raw_columns = nodes._RAW_MEASUREMENT_COLUMNS
    raw = pd.DataFrame(rng.uniform(1.0, 100.0, size=(n_rows, len(raw_columns))), columns=raw_columns)
    return _engineer_feature_frame(raw)


class SimulateProductionDriftTests(unittest.TestCase):
    """Tests for simulate_production_drift."""

    def test_shifted_raw_columns_change_by_the_configured_amount(self) -> None:
        X_test = _make_engineered_frame()
        shifts = {"ph": -1.2, "Sulfate": 80.0}

        simulated = nodes.simulate_production_drift(X_test, {"simulated_shift": shifts})

        pd.testing.assert_series_equal(
            simulated["ph"], X_test["ph"] - 1.2, check_names=False
        )
        pd.testing.assert_series_equal(
            simulated["Sulfate"], X_test["Sulfate"] + 80.0, check_names=False
        )

    def test_engineered_columns_are_recomputed_consistently(self) -> None:
        X_test = _make_engineered_frame()
        shifts = {"ph": -1.2, "Sulfate": 80.0}

        simulated = nodes.simulate_production_drift(X_test, {"simulated_shift": shifts})

        expected_interaction = simulated["Chloramines"] * simulated["ph"]
        pd.testing.assert_series_equal(
            simulated["chloramines_ph_interaction"],
            expected_interaction,
            check_names=False,
        )

    def test_output_has_same_columns_as_input(self) -> None:
        X_test = _make_engineered_frame()

        simulated = nodes.simulate_production_drift(X_test, {"simulated_shift": {"ph": -1.0}})

        self.assertListEqual(list(simulated.columns), list(X_test.columns))

    def test_unshifted_raw_columns_are_unchanged(self) -> None:
        X_test = _make_engineered_frame()

        simulated = nodes.simulate_production_drift(X_test, {"simulated_shift": {"ph": -1.0}})

        pd.testing.assert_series_equal(simulated["Hardness"], X_test["Hardness"])


class SimulatedScenarioShowsMoreDriftTests(unittest.TestCase):
    """The simulated production scenario should flag far more drift than X_train vs X_test."""

    def test_simulated_drift_flags_more_features_than_the_train_test_baseline(self) -> None:
        X_train = _make_engineered_frame(n_rows=150, seed=1)
        X_test = _make_engineered_frame(n_rows=150, seed=2)
        parameters = {
            "significance_threshold": 0.05,
            "simulated_shift": {"ph": -1.2, "Chloramines": 3.0, "Sulfate": 80.0, "Trihalomethanes": 40.0},
        }

        baseline_report = nodes.detect_feature_drift(X_train, X_test, parameters)
        simulated_X_test = nodes.simulate_production_drift(X_test, parameters)
        simulated_report = nodes.detect_feature_drift(X_train, simulated_X_test, parameters)

        self.assertGreater(simulated_report["drifted"].sum(), baseline_report["drifted"].sum())


class EvaluateModelUnderSimulatedDriftTests(unittest.TestCase):
    """Tests for evaluate_model_under_simulated_drift."""

    def setUp(self) -> None:
        rng = np.random.default_rng(21)
        raw_columns = nodes._RAW_MEASUREMENT_COLUMNS
        raw_train = pd.DataFrame(
            rng.uniform(1.0, 100.0, size=(50, len(raw_columns))), columns=raw_columns
        )
        X_train = _engineer_feature_frame(raw_train)
        y_train = pd.Series(rng.integers(0, 2, size=50), name="Potability")

        preprocessing_parameters = {
            "zscore_threshold": 99.0,
            "rfe": {
                "enabled": False,
                "step": 1,
                "cv_folds": 2,
                "scoring": "roc_auc",
                "n_jobs": 1,
                "logistic_max_iter": 100,
                "random_state": 21,
            },
        }
        preprocessor = ModelPreprocessor(preprocessing_parameters)
        X_filtered, y_filtered = preprocessor.fit_resample(X_train, y_train)

        estimator = RandomForestClassifier(n_estimators=5, random_state=21)
        estimator.fit(X_filtered, y_filtered)

        self.model = ModelBundle(
            model_name="random_forest", preprocessor=preprocessor, estimator=estimator
        )
        self.simulated_X_test = _engineer_feature_frame(raw_train.iloc[:20])
        self.y_test = y_train.iloc[:20]

    def test_returns_one_row_with_expected_metric_columns(self) -> None:
        metrics = nodes.evaluate_model_under_simulated_drift(
            self.model, self.simulated_X_test, self.y_test
        )

        self.assertEqual(len(metrics), 1)
        self.assertListEqual(metrics.columns.tolist(), list(METRIC_NAMES))


if __name__ == "__main__":
    unittest.main()
