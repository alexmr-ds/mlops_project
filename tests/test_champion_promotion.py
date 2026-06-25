"""Guard test for the manual champion-promotion gate.

The champion model (Extra Trees) is hardcoded in three places: the serving
app's model path, the drift pipeline input, and the SHAP pipeline input. That
is a deliberate manual MLOps promotion gate rather than a runtime selection.

This test makes the gate explicit and fail-fast: if a re-tune ever re-ranks the
models so the served champion is no longer the top of the committed
``model_comparison.csv``, this test fails and forces a conscious re-promotion of
the champion across serving, drift, and SHAP. It is the programmatic
"recognition" that keeps the hardcoded choice honest.
"""

import unittest

import pandas as pd

from mlops_project.serving import app as serving_app
from src import project_paths

_COMPARISON_PATH = (
    project_paths.PROJECT_ROOT / "data" / "08_reporting" / "model_comparison.csv"
)


class ChampionPromotionGateTests(unittest.TestCase):
    """Tie the hardcoded champion to the data-driven model ranking."""

    def test_served_champion_is_top_ranked_model(self) -> None:
        comparison = pd.read_csv(_COMPARISON_PATH)
        top_row = comparison.sort_values("development_rank").iloc[0]

        self.assertEqual(
            serving_app.CHAMPION_MODEL_NAME,
            top_row["model_name"],
            msg=(
                "Served champion no longer matches the top of model_comparison.csv. "
                "A re-tune re-ranked the models; re-promote the champion in "
                "serving/app.py, the drift pipeline, and the SHAP pipeline."
            ),
        )

    def test_champion_bundle_path_matches_name_and_exists(self) -> None:
        self.assertEqual(
            serving_app._MODEL_PATH.name,
            f"{serving_app.CHAMPION_MODEL_NAME}_model.pkl",
        )
        bundle_path = project_paths.PROJECT_ROOT / serving_app._MODEL_PATH
        self.assertTrue(
            bundle_path.exists(),
            msg=f"Champion bundle missing at {bundle_path}; run the modeling pipeline.",
        )


if __name__ == "__main__":
    unittest.main()
