"""Unit tests for LogisticRegression modeling nodes."""

import tempfile
import unittest

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.linear_model import LogisticRegression

from mlops_project.pipelines.modeling import nodes


class LogisticRegressionModelingTests(unittest.TestCase):
    """Tests for baseline modeling behavior."""

    def test_train_logistic_regression_model_returns_fitted_model(self) -> None:
        X_train, _, _, y_train, _, _, selected_features = _modeling_artifacts()

        model = nodes.train_logistic_regression_model(
            X_train,
            y_train,
            selected_features,
            _parameters(),
        )

        self.assertIsInstance(model, LogisticRegression)
        self.assertTrue(hasattr(model, 'classes_'))
        self.assertEqual(model.max_iter, 1000)
        self.assertEqual(model.solver, 'lbfgs')
        self.assertEqual(model.random_state, 73)

    def test_evaluate_model_returns_required_metrics_and_confusion_matrix(self) -> None:
        X_train, X_validation, _, y_train, y_validation, _, selected_features = (
            _modeling_artifacts()
        )
        model = nodes.train_logistic_regression_model(
            X_train,
            y_train,
            selected_features,
            _parameters(),
        )

        metrics, matrix = nodes.evaluate_validation_model(
            model,
            X_validation,
            y_validation,
            selected_features,
        )

        self.assertListEqual(
            metrics.columns.tolist(),
            ['accuracy', 'precision', 'recall', 'f1', 'f1_weighted', 'roc_auc'],
        )
        self.assertEqual(metrics.shape, (1, 6))
        self.assertEqual(matrix.shape, (2, 2))
        self.assertListEqual(matrix.columns.tolist(), ['predicted_0', 'predicted_1'])
        self.assertListEqual(matrix.index.tolist(), ['actual_0', 'actual_1'])

    def test_create_confusion_matrix_plot_returns_figure(self) -> None:
        matrix = pd.DataFrame(
            [[2, 1], [1, 2]],
            index=['actual_0', 'actual_1'],
            columns=['predicted_0', 'predicted_1'],
        )

        figure = nodes.create_validation_confusion_matrix_plot(matrix)

        self.assertIsInstance(figure, Figure)

    def test_log_model_to_mlflow_returns_run_info(self) -> None:
        X_train, X_validation, X_test, y_train, y_validation, y_test, selected_features = (
            _modeling_artifacts()
        )
        parameters = _parameters()
        model = nodes.train_logistic_regression_model(
            X_train,
            y_train,
            selected_features,
            parameters,
        )
        validation_metrics, validation_matrix = nodes.evaluate_validation_model(
            model,
            X_validation,
            y_validation,
            selected_features,
        )
        test_metrics, test_matrix = nodes.evaluate_test_model(
            model,
            X_test,
            y_test,
            selected_features,
        )
        validation_figure = nodes.create_validation_confusion_matrix_plot(
            validation_matrix
        )
        test_figure = nodes.create_test_confusion_matrix_plot(test_matrix)

        with tempfile.TemporaryDirectory() as tracking_dir:
            parameters['mlflow']['tracking_uri'] = tracking_dir
            run_info = nodes.log_model_to_mlflow(
                model,
                selected_features,
                validation_metrics,
                test_metrics,
                validation_matrix,
                test_matrix,
                validation_figure,
                test_figure,
                parameters,
            )

        self.assertEqual(run_info.shape[0], 1)
        self.assertIn('run_id', run_info.columns)
        self.assertEqual(
            run_info.loc[0, 'experiment_name'],
            'water_potability_modeling',
        )

    def test_train_logistic_regression_model_fails_for_empty_training_data(self) -> None:
        X_train, _, _, y_train, _, _, selected_features = _modeling_artifacts()

        with self.assertRaisesRegex(ValueError, 'train features must not be empty'):
            nodes.train_logistic_regression_model(
                X_train.iloc[0:0],
                y_train.iloc[0:0],
                selected_features,
                _parameters(),
            )

    def test_evaluate_validation_model_fails_for_empty_validation_data(self) -> None:
        X_train, X_validation, _, y_train, y_validation, _, selected_features = (
            _modeling_artifacts()
        )
        model = nodes.train_logistic_regression_model(
            X_train,
            y_train,
            selected_features,
            _parameters(),
        )

        with self.assertRaisesRegex(ValueError, 'validation features must not be empty'):
            nodes.evaluate_validation_model(
                model,
                X_validation.iloc[0:0],
                y_validation.iloc[0:0],
                selected_features,
            )

    def test_evaluate_test_model_fails_for_mismatched_row_counts(self) -> None:
        X_train, _, X_test, y_train, _, y_test, selected_features = _modeling_artifacts()
        model = nodes.train_logistic_regression_model(
            X_train,
            y_train,
            selected_features,
            _parameters(),
        )

        with self.assertRaisesRegex(ValueError, 'test X/y row counts must match'):
            nodes.evaluate_test_model(
                model,
                X_test,
                y_test.iloc[:-1],
                selected_features,
            )

    def test_evaluate_model_fails_for_missing_selected_features(self) -> None:
        X_train, X_validation, _, y_train, y_validation, _, selected_features = (
            _modeling_artifacts()
        )
        model = nodes.train_logistic_regression_model(
            X_train,
            y_train,
            selected_features,
            _parameters(),
        )

        with self.assertRaisesRegex(ValueError, 'missing selected_features'):
            nodes.evaluate_validation_model(
                model,
                X_validation.drop(columns=['feature_b']),
                y_validation,
                selected_features,
            )

    def test_train_logistic_regression_model_fails_for_missing_selected_features(
        self,
    ) -> None:
        X_train, _, _, y_train, _, _, _ = _modeling_artifacts()

        with self.assertRaisesRegex(ValueError, 'selected_features'):
            nodes.train_logistic_regression_model(
                X_train,
                y_train,
                [],
                _parameters(),
            )


def _parameters() -> dict[str, object]:
    """Return modeling parameters for tests."""
    return {
        'logistic_regression': {
            'max_iter': 1000,
            'solver': 'lbfgs',
            'random_state': 73,
        },
        'mlflow': {
            'tracking_uri': 'mlruns',
            'experiment_name': 'water_potability_modeling',
            'run_name': 'logistic_regression_baseline',
        },
    }


def _modeling_artifacts() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
    list[str],
]:
    """Return linearly separable modeling artifacts."""
    selected_features = ['feature_a', 'feature_b']
    X_train = pd.DataFrame(
        {
            'feature_a': np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]),
            'feature_b': np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]),
        },
        index=[10, 11, 12, 13, 14, 15],
    )
    y_train = pd.Series(np.array([0, 0, 0, 1, 1, 1]), index=X_train.index)
    X_validation = pd.DataFrame(
        {
            'feature_a': np.array([-2.5, -0.75, 0.75, 2.5]),
            'feature_b': np.array([-2.0, -0.25, 0.25, 2.0]),
        },
        index=[20, 21, 22, 23],
    )
    y_validation = pd.Series(np.array([0, 0, 1, 1]), index=X_validation.index)
    X_test = pd.DataFrame(
        {
            'feature_a': np.array([-2.25, -1.25, 1.25, 2.25]),
            'feature_b': np.array([-1.75, -0.75, 0.75, 1.75]),
        },
        index=[30, 31, 32, 33],
    )
    y_test = pd.Series(np.array([0, 0, 1, 1]), index=X_test.index)
    return X_train, X_validation, X_test, y_train, y_validation, y_test, selected_features


if __name__ == '__main__':
    unittest.main()
