"""Unit tests for modeling nodes."""

import tempfile
import unittest

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from mlops_project.modeling import model_bundle
from mlops_project.pipelines.modeling import nodes


class ModelingNodeTests(unittest.TestCase):
    """Tests for cross-validation, final testing, and logging behavior."""

    def test_cross_validate_logistic_regression_returns_cv_and_test_outputs(self) -> None:
        X_train, X_test, y_train, y_test = _modeling_artifacts()

        model, cv_metrics, cv_fold_metrics, test_metrics, matrix, selected_features = (
            nodes.cross_validate_and_train_logistic_regression(
                X_train,
                X_test,
                y_train,
                y_test,
                _preprocessing_parameters(),
                _modeling_parameters(),
            )
        )

        self.assertIsInstance(model, model_bundle.ModelBundle)
        self.assertIsInstance(model.estimator, LogisticRegression)
        self.assertTrue(hasattr(model, 'classes_'))
        self.assertIsNotNone(model.preprocessor.imputer)
        self.assertIsNotNone(model.preprocessor.scaler)
        self.assertEqual(len(model.predict(X_test)), len(X_test))
        self.assertEqual(cv_metrics.shape, (1, 12))
        self.assertIn('cv_mean_f1', cv_metrics.columns)
        self.assertEqual(len(cv_fold_metrics), 3)
        self.assertEqual(test_metrics.shape, (1, 6))
        self.assertEqual(matrix.shape, (2, 2))
        self.assertListEqual(matrix.index.tolist(), [0, 1])
        self.assertListEqual(matrix.columns.tolist(), [0, 1])
        self.assertGreaterEqual(len(selected_features), 1)
        self.assertTrue(set(selected_features).issubset(X_train.columns))

    def test_cross_validate_random_forest_returns_cv_and_test_outputs(self) -> None:
        X_train, X_test, y_train, y_test = _modeling_artifacts()

        model, cv_metrics, cv_fold_metrics, test_metrics, matrix, selected_features = (
            nodes.cross_validate_and_train_random_forest(
                X_train,
                X_test,
                y_train,
                y_test,
                _preprocessing_parameters(),
                _modeling_parameters(),
            )
        )

        self.assertIsInstance(model, model_bundle.ModelBundle)
        self.assertIsInstance(model.estimator, RandomForestClassifier)
        self.assertTrue(hasattr(model, 'classes_'))
        self.assertIsNotNone(model.preprocessor.imputer)
        self.assertIsNotNone(model.preprocessor.scaler)
        self.assertEqual(len(model.predict(X_test)), len(X_test))
        self.assertEqual(model.estimator.n_estimators, 10)
        self.assertEqual(model.estimator.random_state, 73)
        self.assertEqual(model.estimator.n_jobs, -1)
        self.assertEqual(cv_metrics.shape, (1, 12))
        self.assertEqual(len(cv_fold_metrics), 3)
        self.assertEqual(test_metrics.shape, (1, 6))
        self.assertEqual(matrix.shape, (2, 2))
        self.assertGreaterEqual(len(selected_features), 1)

    def test_cross_validation_fails_when_n_splits_exceeds_smallest_class_count(
        self,
    ) -> None:
        X_train, X_test, y_train, y_test = _modeling_artifacts()
        modeling_parameters = _modeling_parameters()
        modeling_parameters['cross_validation']['n_splits'] = 20

        with self.assertRaisesRegex(ValueError, 'n_splits cannot exceed'):
            nodes.cross_validate_and_train_logistic_regression(
                X_train,
                X_test,
                y_train,
                y_test,
                _preprocessing_parameters(),
                modeling_parameters,
            )

    def test_evaluate_model_fails_for_missing_selected_features(self) -> None:
        X_train, X_test, y_train, y_test = _modeling_artifacts()
        model, _, _, _, _, selected_features = (
            nodes.cross_validate_and_train_logistic_regression(
                X_train,
                X_test,
                y_train,
                y_test,
                _preprocessing_parameters(),
                _modeling_parameters(),
            )
        )

        with self.assertRaisesRegex(ValueError, 'selected_features are missing'):
            nodes.evaluate_model(
                model,
                X_test.drop(columns=selected_features[0]),
                y_test,
                selected_features,
            )

    def test_create_test_confusion_matrix_plot_returns_figure(self) -> None:
        matrix = pd.DataFrame([[2, 1], [1, 2]], index=[0, 1], columns=[0, 1])

        figure = nodes.create_test_confusion_matrix_plot(matrix)

        self.assertIsInstance(figure, Figure)

    def test_create_test_confusion_matrix_plot_accepts_csv_string_columns(self) -> None:
        matrix = pd.DataFrame([[2, 1], [1, 2]], index=[0, 1], columns=['0', '1'])

        figure = nodes.create_test_confusion_matrix_plot(matrix)

        self.assertIsInstance(figure, Figure)

    def test_log_model_to_mlflow_returns_run_info(self) -> None:
        X_train, X_test, y_train, y_test = _modeling_artifacts()
        parameters = _modeling_parameters()
        model, cv_metrics, cv_fold_metrics, test_metrics, matrix, selected_features = (
            nodes.cross_validate_and_train_logistic_regression(
                X_train,
                X_test,
                y_train,
                y_test,
                _preprocessing_parameters(),
                parameters,
            )
        )
        figure = nodes.create_test_confusion_matrix_plot(matrix)

        with tempfile.TemporaryDirectory() as tracking_dir:
            parameters['mlflow']['tracking_uri'] = tracking_dir
            run_info = nodes.log_model_to_mlflow(
                model,
                selected_features,
                cv_metrics,
                cv_fold_metrics,
                test_metrics,
                matrix,
                figure,
                parameters,
            )

        self.assertEqual(run_info.shape[0], 1)
        self.assertIn('run_id', run_info.columns)
        self.assertEqual(
            run_info.loc[0, 'experiment_name'],
            'water_potability_modeling',
        )

    def test_log_random_forest_to_mlflow_returns_run_info(self) -> None:
        X_train, X_test, y_train, y_test = _modeling_artifacts()
        parameters = _modeling_parameters()
        model, cv_metrics, cv_fold_metrics, test_metrics, matrix, selected_features = (
            nodes.cross_validate_and_train_random_forest(
                X_train,
                X_test,
                y_train,
                y_test,
                _preprocessing_parameters(),
                parameters,
            )
        )
        figure = nodes.create_test_confusion_matrix_plot(matrix)

        with tempfile.TemporaryDirectory() as tracking_dir:
            parameters['mlflow']['tracking_uri'] = tracking_dir
            run_info = nodes.log_random_forest_to_mlflow(
                model,
                selected_features,
                cv_metrics,
                cv_fold_metrics,
                test_metrics,
                matrix,
                figure,
                parameters,
            )

        self.assertEqual(run_info.shape[0], 1)
        self.assertIn('run_id', run_info.columns)
        self.assertEqual(
            run_info.loc[0, 'run_name'],
            'random_forest_nonlinear_probe',
        )

    def test_cross_validate_model_fails_for_empty_training_data(self) -> None:
        X_train, X_test, y_train, y_test = _modeling_artifacts()

        with self.assertRaisesRegex(ValueError, 'X_train must not be empty'):
            nodes.cross_validate_and_train_logistic_regression(
                X_train.iloc[0:0],
                X_test,
                y_train.iloc[0:0],
                y_test,
                _preprocessing_parameters(),
                _modeling_parameters(),
            )


def _preprocessing_parameters() -> dict[str, object]:
    """Return preprocessing parameters for model-local fold transforms."""
    return {
        'feature_columns': None,
        'zscore_threshold': 99.0,
        'rfe': {
            'enabled': True,
            'step': 1,
            'cv_folds': 2,
            'scoring': 'roc_auc',
            'n_jobs': -1,
            'logistic_max_iter': 1000,
            'random_state': 73,
        },
    }


def _modeling_parameters() -> dict[str, object]:
    """Return modeling parameters for tests."""
    return {
        'cross_validation': {
            'n_splits': 3,
            'shuffle': True,
            'random_state': 73,
        },
        'logistic_regression': {
            'max_iter': 1000,
            'solver': 'lbfgs',
            'random_state': 73,
        },
        'random_forest': {
            'n_estimators': 10,
            'max_depth': None,
            'min_samples_leaf': 1,
            'random_state': 73,
            'n_jobs': -1,
            'class_weight': None,
        },
        'mlflow': {
            'tracking_uri': 'mlruns',
            'experiment_name': 'water_potability_modeling',
            'run_name': 'logistic_regression_baseline',
            'random_forest_run_name': 'random_forest_nonlinear_probe',
        },
    }


def _modeling_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return separable train/test modeling artifacts."""
    train_signal = np.array([0] * 12 + [1] * 12)
    train_feature_a = np.concatenate(
        [np.linspace(-3.0, -0.5, 12), np.linspace(0.5, 3.0, 12)]
    )
    train_feature_b = train_feature_a * 0.8
    train_noise = np.resize([-0.2, 0.0, 0.2], 24)
    X_train = pd.DataFrame(
        {
            'feature_a': train_feature_a,
            'feature_b': train_feature_b,
            'noise': train_noise,
        },
        index=np.arange(100, 124),
    )
    y_train = pd.Series(train_signal, index=X_train.index)

    test_signal = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    test_feature_a = np.array([-2.5, -1.8, -1.1, -0.6, 0.6, 1.1, 1.8, 2.5])
    X_test = pd.DataFrame(
        {
            'feature_a': test_feature_a,
            'feature_b': test_feature_a * 0.8,
            'noise': np.resize([-0.1, 0.1], 8),
        },
        index=np.arange(200, 208),
    )
    y_test = pd.Series(test_signal, index=X_test.index)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    unittest.main()
