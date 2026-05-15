"""Custom Kedro datasets for CSV and pickle persistence."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from kedro.io import AbstractDataset


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
