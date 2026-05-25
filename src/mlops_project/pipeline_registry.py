"""Pipeline registry for the Kedro project."""

from kedro.pipeline import Pipeline

from .pipelines import modeling, preprocessing


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines."""
    preprocessing_pipeline = preprocessing.create_pipeline()
    logistic_regression_pipeline = modeling.create_logistic_regression_pipeline()
    random_forest_pipeline = modeling.create_random_forest_pipeline()
    modeling_pipeline = logistic_regression_pipeline + random_forest_pipeline
    return {
        "preprocessing": preprocessing_pipeline,
        "modeling_logistic_regression": logistic_regression_pipeline,
        "modeling_random_forest": random_forest_pipeline,
        "modeling": modeling_pipeline,
        "__default__": preprocessing_pipeline + modeling_pipeline,
    }
