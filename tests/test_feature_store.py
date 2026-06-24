"""Unit tests for the file-based feature store and its Kedro dataset."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from mlops_project import feature_store
from mlops_project.datasets import FeatureSetDataset
from mlops_project.pipelines.preprocessing.nodes import _engineer_feature_frame


def _engineered_sample(rows: int = 12, seed: int = 7) -> pd.DataFrame:
    """Build a realistic engineered feature frame for tests."""
    rng = np.random.default_rng(seed)
    raw = pd.DataFrame(
        rng.uniform(1.0, 100.0, size=(rows, len(feature_store.raw_measurement_names()))),
        columns=feature_store.raw_measurement_names(),
    )
    return _engineer_feature_frame(raw)


class FeatureRegistryContractTests(unittest.TestCase):
    """The registry must stay exactly in sync with the engineering code."""

    def test_registry_matches_engineered_columns_in_order(self) -> None:
        engineered = _engineered_sample()

        self.assertEqual(feature_store.feature_names(), list(engineered.columns))

    def test_registry_counts(self) -> None:
        self.assertEqual(len(feature_store.feature_names()), 32)
        self.assertEqual(len(feature_store.raw_measurement_names()), 9)
        self.assertEqual(len(feature_store.engineered_feature_names()), 23)

    def test_registry_dtypes_match_engineered_frame(self) -> None:
        engineered = _engineered_sample()

        mismatches = [
            column
            for column in engineered.columns
            if str(engineered[column].dtype) != feature_store.describe(column).dtype
        ]
        self.assertEqual(mismatches, [])

    def test_describe_rejects_unknown_feature(self) -> None:
        with self.assertRaises(KeyError):
            feature_store.describe("not_a_feature")


class VersioningTests(unittest.TestCase):
    """Versions are deterministic and content-addressable."""

    def test_same_frame_yields_same_versions(self) -> None:
        frame = _engineered_sample()

        self.assertEqual(feature_store.schema_version(frame), feature_store.schema_version(frame.copy()))
        self.assertEqual(feature_store.snapshot_version(frame), feature_store.snapshot_version(frame.copy()))

    def test_changed_data_changes_snapshot_but_not_schema(self) -> None:
        frame = _engineered_sample()
        mutated = frame.copy()
        mutated.iloc[0, 0] = mutated.iloc[0, 0] + 1.0

        self.assertEqual(feature_store.schema_version(frame), feature_store.schema_version(mutated))
        self.assertNotEqual(feature_store.snapshot_version(frame), feature_store.snapshot_version(mutated))

    def test_dropping_a_column_changes_schema_version(self) -> None:
        frame = _engineered_sample()
        fewer = frame.drop(columns=["risk_score"])

        self.assertNotEqual(feature_store.schema_version(frame), feature_store.schema_version(fewer))


class FeatureMetadataTests(unittest.TestCase):
    """Metadata documents describe the materialised feature set faithfully."""

    def test_metadata_structure_and_definitions(self) -> None:
        frame = _engineered_sample(rows=15)

        metadata = feature_store.build_feature_metadata(frame, name="X_train")

        self.assertEqual(metadata["name"], "X_train")
        self.assertEqual(metadata["row_count"], 15)
        self.assertEqual(metadata["feature_count"], 32)
        self.assertEqual(
            metadata["version"],
            f"{metadata['schema_version']}.{metadata['snapshot_version']}",
        )
        self.assertEqual([column["name"] for column in metadata["columns"]], feature_store.feature_names())
        first = metadata["columns"][0]
        self.assertEqual(set(first), {"name", "dtype", "group", "description"})
        self.assertTrue(first["description"])

    def test_metadata_rejects_unregistered_columns(self) -> None:
        frame = _engineered_sample()
        frame["surprise_column"] = 1.0

        with self.assertRaises(ValueError):
            feature_store.build_feature_metadata(frame, name="X_train")


class FeatureSetDatasetTests(unittest.TestCase):
    """The Kedro dataset persists features and a deterministic metadata sidecar."""

    def test_save_writes_pickle_and_metadata_sidecar(self) -> None:
        frame = _engineered_sample()
        with tempfile.TemporaryDirectory() as directory:
            filepath = Path(directory) / "X_train.pkl"
            dataset = FeatureSetDataset(filepath=str(filepath))

            dataset.save(frame)

            metadata_path = filepath.with_name("X_train_metadata.json")
            self.assertTrue(filepath.exists())
            self.assertTrue(metadata_path.exists())
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["name"], "X_train")
            self.assertEqual(metadata["feature_count"], frame.shape[1])

    def test_round_trip_returns_equal_frame(self) -> None:
        frame = _engineered_sample()
        with tempfile.TemporaryDirectory() as directory:
            dataset = FeatureSetDataset(filepath=str(Path(directory) / "X_test.pkl"))

            dataset.save(frame)
            loaded = dataset.load()

            pd.testing.assert_frame_equal(loaded, frame)

    def test_save_is_deterministic(self) -> None:
        frame = _engineered_sample()
        with tempfile.TemporaryDirectory() as directory:
            filepath = Path(directory) / "X_train.pkl"
            dataset = FeatureSetDataset(filepath=str(filepath))
            metadata_path = filepath.with_name("X_train_metadata.json")

            dataset.save(frame)
            first_pickle = filepath.read_bytes()
            first_metadata = metadata_path.read_text(encoding="utf-8")
            dataset.save(frame)

            self.assertEqual(first_pickle, filepath.read_bytes())
            self.assertEqual(first_metadata, metadata_path.read_text(encoding="utf-8"))


class OfflineRetrievalTests(unittest.TestCase):
    """The offline retrieval API returns features together with metadata."""

    def test_load_offline_features_returns_frame_and_metadata(self) -> None:
        frame = _engineered_sample()
        with tempfile.TemporaryDirectory() as directory:
            FeatureSetDataset(filepath=str(Path(directory) / "X_train.pkl")).save(frame)

            loaded, metadata = feature_store.load_offline_features("X_train", feature_dir=Path(directory))

            pd.testing.assert_frame_equal(loaded, frame)
            self.assertEqual(metadata["name"], "X_train")

    def test_missing_feature_set_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                feature_store.load_offline_features("X_train", feature_dir=Path(directory))


if __name__ == "__main__":
    unittest.main()
