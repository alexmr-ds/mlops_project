"""Unit tests for model-local learned preprocessing."""

import unittest
from unittest import mock

import numpy as np
import pandas as pd

from mlops_project.modeling import preprocessing


class ModelPreprocessorTests(unittest.TestCase):
    """Tests for leakage-safe learned preprocessing behavior."""

    def test_fit_resample_transform_uses_training_statistics(self) -> None:
        X_train = pd.DataFrame(
            {
                'feature_a': [1.0, 3.0, 5.0, 7.0],
                'feature_b': [2.0, 4.0, 6.0, 8.0],
            },
            index=[10, 11, 12, 13],
        )
        y_train = pd.Series([0, 1, 0, 1], index=X_train.index)
        X_evaluation = pd.DataFrame(
            {
                'feature_a': [np.nan, 9.0],
                'feature_b': [5.0, 10.0],
            },
            index=[20, 21],
        )
        preprocessor = preprocessing.ModelPreprocessor(_parameters(rfe_enabled=False))

        transformed_train, transformed_y = preprocessor.fit_resample(X_train, y_train)
        transformed_evaluation = preprocessor.transform(X_evaluation)

        self.assertListEqual(preprocessor.selected_features, ['feature_a', 'feature_b'])
        self.assertListEqual(transformed_train.index.tolist(), X_train.index.tolist())
        self.assertListEqual(transformed_y.index.tolist(), y_train.index.tolist())
        self.assertEqual(len(transformed_evaluation), len(X_evaluation))
        self.assertAlmostEqual(transformed_evaluation.loc[20, 'feature_a'], 0.0, places=6)

    def test_fit_resample_validates_model_ready_training_artifacts(self) -> None:
        X_train = pd.DataFrame(
            {
                'feature_a': [1.0, 3.0, 5.0, 7.0],
                'feature_b': [2.0, 4.0, 6.0, 8.0],
            }
        )
        y_train = pd.Series([0, 1, 0, 1], index=X_train.index)
        preprocessor = preprocessing.ModelPreprocessor(_parameters(rfe_enabled=False))

        with mock.patch.object(
            preprocessing.modeling_validation,
            'validate_model_ready_training_artifacts',
            side_effect=lambda features, labels, expected_columns, artifact_name: (
                features,
                labels,
            ),
        ) as validator:
            preprocessor.fit_resample(X_train, y_train)

        self.assertEqual(validator.call_count, 1)
        self.assertListEqual(
            list(validator.call_args.args[2]),
            ['feature_a', 'feature_b'],
        )
        self.assertEqual(
            validator.call_args.kwargs['artifact_name'],
            'model-ready training',
        )

    def test_transform_validates_model_ready_evaluation_features(self) -> None:
        X_train = pd.DataFrame(
            {
                'feature_a': [1.0, 3.0, 5.0, 7.0],
                'feature_b': [2.0, 4.0, 6.0, 8.0],
            }
        )
        y_train = pd.Series([0, 1, 0, 1], index=X_train.index)
        X_evaluation = pd.DataFrame(
            {
                'feature_a': [9.0],
                'feature_b': [10.0],
            }
        )
        preprocessor = preprocessing.ModelPreprocessor(_parameters(rfe_enabled=False))
        preprocessor.fit_resample(X_train, y_train)

        with mock.patch.object(
            preprocessing.modeling_validation,
            'validate_model_ready_features',
            side_effect=lambda features, expected_columns, artifact_name: features,
        ) as validator:
            preprocessor.transform(X_evaluation)

        self.assertEqual(validator.call_count, 1)
        self.assertListEqual(
            list(validator.call_args.args[1]),
            ['feature_a', 'feature_b'],
        )
        self.assertEqual(
            validator.call_args.kwargs['artifact_name'],
            'model-ready evaluation',
        )

    def test_fit_resample_removes_outliers_only_from_training_rows(self) -> None:
        X_train = pd.DataFrame(
            {
                'feature_a': [0.0, 0.0, 0.0, 100.0],
                'feature_b': [1.0, 1.0, 1.0, 1.0],
            },
            index=[10, 11, 12, 13],
        )
        y_train = pd.Series([0, 1, 0, 1], index=X_train.index)
        X_evaluation = pd.DataFrame(
            {
                'feature_a': [1000.0],
                'feature_b': [1.0],
            },
            index=[20],
        )
        preprocessor = preprocessing.ModelPreprocessor(
            _parameters(rfe_enabled=False, zscore_threshold=1.0)
        )

        transformed_train, transformed_y = preprocessor.fit_resample(X_train, y_train)
        transformed_evaluation = preprocessor.transform(X_evaluation)

        self.assertEqual(preprocessor.filtered_training_row_count, 3)
        self.assertListEqual(transformed_train.index.tolist(), [10, 11, 12])
        self.assertListEqual(transformed_y.index.tolist(), [10, 11, 12])
        self.assertEqual(len(transformed_evaluation), 1)

    def test_transform_fails_before_fit(self) -> None:
        preprocessor = preprocessing.ModelPreprocessor(_parameters(rfe_enabled=False))

        with self.assertRaisesRegex(ValueError, 'must be fitted'):
            preprocessor.transform(pd.DataFrame({'feature_a': [1.0]}))

    def test_fit_resample_selects_non_empty_rfecv_features(self) -> None:
        signal = np.array([0] * 12 + [1] * 12)
        X_train = pd.DataFrame(
            {
                'signal': signal.astype(float),
                'weak_signal': signal.astype(float) * 0.5,
                'noise': np.resize([-0.1, 0.0, 0.1], 24),
            }
        )
        y_train = pd.Series(signal)
        preprocessor = preprocessing.ModelPreprocessor(_parameters(rfe_enabled=True))

        transformed_train, transformed_y = preprocessor.fit_resample(X_train, y_train)

        self.assertGreaterEqual(len(preprocessor.selected_features), 1)
        self.assertListEqual(transformed_train.columns.tolist(), preprocessor.selected_features)
        self.assertListEqual(transformed_y.index.tolist(), transformed_train.index.tolist())


def _parameters(
    *,
    rfe_enabled: bool,
    zscore_threshold: float = 99.0,
) -> dict[str, object]:
    """Return learned preprocessing parameters for tests."""
    return {
        'feature_columns': None,
        'zscore_threshold': zscore_threshold,
        'rfe': {
            'enabled': rfe_enabled,
            'step': 1,
            'cv_folds': 2,
            'scoring': 'roc_auc',
            'n_jobs': -1,
            'logistic_max_iter': 1000,
            'random_state': 73,
        },
    }


if __name__ == '__main__':
    unittest.main()
