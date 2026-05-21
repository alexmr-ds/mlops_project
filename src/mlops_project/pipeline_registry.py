"""Pipeline registry for the Kedro project."""

from kedro.pipeline import Pipeline

from .pipelines import modeling, preprocessing


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines."""
    preprocessing_pipeline = preprocessing.create_pipeline()
    modeling_pipeline = modeling.create_pipeline()
    return {
        "preprocessing": preprocessing_pipeline,
        "modeling": modeling_pipeline,
        "__default__": preprocessing_pipeline + modeling_pipeline,
    }
