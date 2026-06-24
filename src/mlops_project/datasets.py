"""Custom Kedro datasets for CSV, pickle, and figure persistence."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from kedro.io import AbstractDataset
from matplotlib.figure import Figure

from mlops_project import feature_store


class PandasCSVDataset(AbstractDataset[pd.DataFrame, pd.DataFrame]):
    """Load and save pandas dataframes as CSV files."""

    def __init__(
        self,
        filepath: str,
        load_args: dict[str, Any] | None = None,
        save_args: dict[str, Any] | None = None,
    ) -> None:
        self._filepath = Path(filepath)
        self._load_args = load_args or {}
        self._save_args = save_args or {"index": False}

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self._filepath, **self._load_args)

    def save(self, data: pd.DataFrame) -> None:
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(self._filepath, **self._save_args)

    def _exists(self) -> bool:
        return self._filepath.exists()

    def _describe(self) -> dict[str, Any]:
        return {"filepath": str(self._filepath)}


class PickleDataset(AbstractDataset[Any, Any]):
    """Persist arbitrary Python objects with pickle."""

    def __init__(self, filepath: str) -> None:
        self._filepath = Path(filepath)

    def load(self) -> Any:
        with self._filepath.open('rb') as buffer:
            return pickle.load(buffer)

    def save(self, data: Any) -> None:
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        with self._filepath.open('wb') as buffer:
            pickle.dump(data, buffer)

    def _exists(self) -> bool:
        return self._filepath.exists()

    def _describe(self) -> dict[str, Any]:
        return {"filepath": str(self._filepath)}


class FeatureSetDataset(AbstractDataset[pd.DataFrame, pd.DataFrame]):
    """Persist an engineered feature set into the file-based feature store.

    On save this pickles the feature frame and writes a deterministic metadata
    sidecar (`<name>_metadata.json`) describing the schema, the content-addressable
    version, and every feature definition. The pickle payload is identical to a
    plain pickle, so model artifacts trained on these features stay valid; the
    sidecar is what turns the directory into a versioned feature store rather than
    a bag of pickles. The feature set name is taken from the file stem.
    """

    def __init__(self, filepath: str) -> None:
        self._filepath = Path(filepath)

    def load(self) -> pd.DataFrame:
        with self._filepath.open("rb") as buffer:
            return pickle.load(buffer)

    def save(self, data: pd.DataFrame) -> None:
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        with self._filepath.open("wb") as buffer:
            pickle.dump(data, buffer)
        feature_store.write_feature_metadata(data, self._filepath, name=self._filepath.stem)

    def _exists(self) -> bool:
        return self._filepath.exists()

    def _describe(self) -> dict[str, Any]:
        return {
            "filepath": str(self._filepath),
            "metadata_path": str(feature_store.metadata_path_for(self._filepath)),
        }


class MatplotlibFigureDataset(AbstractDataset[Figure, Figure]):
    """Persist matplotlib figures as image files."""

    def __init__(
        self,
        filepath: str,
        save_args: dict[str, Any] | None = None,
    ) -> None:
        self._filepath = Path(filepath)
        self._save_args = save_args or {"bbox_inches": "tight", "dpi": 150}

    def load(self) -> Figure:
        image = plt.imread(self._filepath)
        figure, axis = plt.subplots()
        axis.imshow(image)
        axis.axis("off")
        return figure

    def save(self, data: Figure) -> None:
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        data.savefig(self._filepath, **self._save_args)

    def _exists(self) -> bool:
        return self._filepath.exists()

    def _describe(self) -> dict[str, Any]:
        return {"filepath": str(self._filepath)}
