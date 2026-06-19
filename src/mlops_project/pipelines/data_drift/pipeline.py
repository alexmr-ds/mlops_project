"""Kedro pipeline definition for data drift detection."""

from kedro.pipeline import Pipeline, node, pipeline

from . import nodes


def create_pipeline(**kwargs: object) -> Pipeline:
    """Create the data drift detection pipeline.

    Compares the training feature distributions against the test split using
    KS tests.  The resulting drift report can be inspected to decide whether
    the model needs retraining before the next production deployment.
    """
    del kwargs
    return pipeline(
        [
            node(
                func=nodes.detect_feature_drift,
                inputs=["X_train", "X_test", "params:data_drift"],
                outputs="drift_report",
                name="detect_feature_drift_node",
            ),
        ]
    )
