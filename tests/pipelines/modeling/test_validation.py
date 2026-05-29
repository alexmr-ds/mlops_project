"""Unit tests for model-ready validation contracts."""

import unittest

import numpy as np
import pandas as pd

from mlops_project.modeling import validation


class ModelReadyValidationTests(unittest.TestCase):
    """Tests for transformed estimator-input validation."""

    def test_validate_model_ready_features_returns_valid_frame(self) -> None:
        features = _valid_features()

        validated_features = validation.validate_model_ready_features(
            features,
            ["feature_a", "feature_b"],
            artifact_name="fold validation features",
        )

        self.assertIs(validated_features, features)

    def test_validate_model_ready_features_fails_for_reordered_columns(self) -> None:
        features = _valid_features().loc[:, ["feature_b", "feature_a"]]

        with self.assertRaisesRegex(ValueError, "model-ready feature validation failed"):
            validation.validate_model_ready_features(
                features,
                ["feature_a", "feature_b"],
                artifact_name="fold validation features",
            )

    def test_validate_model_ready_features_fails_for_missing_columns(self) -> None:
        features = _valid_features().drop(columns=["feature_b"])

        with self.assertRaisesRegex(ValueError, "model-ready feature validation failed"):
            validation.validate_model_ready_features(
                features,
                ["feature_a", "feature_b"],
                artifact_name="fold validation features",
            )

    def test_validate_model_ready_features_fails_for_empty_frame(self) -> None:
        features = _valid_features().iloc[0:0]

        with self.assertRaisesRegex(ValueError, "model-ready feature validation failed"):
            validation.validate_model_ready_features(
                features,
                ["feature_a", "feature_b"],
                artifact_name="fold validation features",
            )

    def test_validate_model_ready_features_fails_for_null_values(self) -> None:
        features = _valid_features()
        features.loc[0, "feature_a"] = np.nan

        with self.assertRaisesRegex(ValueError, "model-ready feature validation failed"):
            validation.validate_model_ready_features(
                features,
                ["feature_a", "feature_b"],
                artifact_name="fold validation features",
            )

    def test_validate_model_ready_features_fails_for_infinite_values(self) -> None:
        features = _valid_features()
        features.loc[0, "feature_a"] = np.inf

        with self.assertRaisesRegex(ValueError, "infinite feature values"):
            validation.validate_model_ready_features(
                features,
                ["feature_a", "feature_b"],
                artifact_name="fold validation features",
            )

    def test_validate_model_ready_features_fails_for_non_numeric_values(self) -> None:
        features = _valid_features()
        features["feature_b"] = ["bad", "values", "here"]

        with self.assertRaisesRegex(ValueError, "model-ready feature validation failed"):
            validation.validate_model_ready_features(
                features,
                ["feature_a", "feature_b"],
                artifact_name="fold validation features",
            )

    def test_validate_model_ready_training_artifacts_returns_valid_inputs(self) -> None:
        features = _valid_features()
        labels = pd.Series([0, 1, 0], index=features.index)

        validated_features, validated_labels = (
            validation.validate_model_ready_training_artifacts(
                features,
                labels,
                ["feature_a", "feature_b"],
                artifact_name="fold training",
            )
        )

        self.assertIs(validated_features, features)
        self.assertIs(validated_labels, labels)

    def test_validate_model_ready_training_artifacts_fails_for_wrong_label_length(
        self,
    ) -> None:
        features = _valid_features()
        labels = pd.Series([0, 1], index=features.index[:2])

        with self.assertRaisesRegex(ValueError, "row count must match features"):
            validation.validate_model_ready_training_artifacts(
                features,
                labels,
                ["feature_a", "feature_b"],
                artifact_name="fold training",
            )

    def test_validate_model_ready_training_artifacts_fails_for_misaligned_labels(
        self,
    ) -> None:
        features = _valid_features()
        labels = pd.Series([0, 1, 0], index=[10, 11, 12])

        with self.assertRaisesRegex(ValueError, "index must match features"):
            validation.validate_model_ready_training_artifacts(
                features,
                labels,
                ["feature_a", "feature_b"],
                artifact_name="fold training",
            )

    def test_validate_model_ready_training_artifacts_fails_for_null_labels(
        self,
    ) -> None:
        features = _valid_features()
        labels = pd.Series([0, np.nan, 1], index=features.index)

        with self.assertRaisesRegex(ValueError, "null label values"):
            validation.validate_model_ready_training_artifacts(
                features,
                labels,
                ["feature_a", "feature_b"],
                artifact_name="fold training",
            )

    def test_validate_model_ready_training_artifacts_fails_for_invalid_labels(
        self,
    ) -> None:
        features = _valid_features()
        labels = pd.Series([0, 2, 1], index=features.index)

        with self.assertRaisesRegex(ValueError, "binary labels"):
            validation.validate_model_ready_training_artifacts(
                features,
                labels,
                ["feature_a", "feature_b"],
                artifact_name="fold training",
            )


def _valid_features() -> pd.DataFrame:
    """Return a valid transformed feature matrix for tests."""
    return pd.DataFrame(
        {
            "feature_a": [0.1, -0.2, 0.3],
            "feature_b": [1.0, 0.0, -1.0],
        },
        index=[3, 4, 5],
    )


if __name__ == "__main__":
    unittest.main()
