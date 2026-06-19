"""Unit tests for data drift detection nodes."""

import unittest

import numpy as np
import pandas as pd

from mlops_project.pipelines.data_drift import nodes


def _make_identical_splits(n_rows: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two splits sampled from the same distribution — expect no drift."""
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


if __name__ == "__main__":
    unittest.main()
