"""A lightweight, file-based feature store for the engineered feature layer.

This is an "own solution" feature store rather than a third-party service. It
gives the project the properties a feature store is meant to provide without the
operational weight of running one:

- A single registry that is the source of truth for every feature name, dtype,
  group, and human-readable description (the feature definitions / schema).
- Content-addressable versioning: each materialised feature set carries a
  deterministic metadata sidecar recording a schema version and a data snapshot
  version, so identical inputs always produce an identical version string and a
  changed feature definition or changed data is immediately visible.
- A small offline-retrieval API used by training, drift detection, and tests to
  read a named feature set back together with its metadata.

The store deliberately covers only the deterministic engineered features. The
fold-local learned preprocessing (outlier filtering, imputation, scaling, RFECV)
is intentionally never stored here, because it must be re-fitted inside each
cross-validation fold to stay leakage-safe. See ``modeling.preprocessing``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

FEATURE_LAYER_DIR = Path("data") / "04_feature"
METADATA_SUFFIX = "_metadata.json"

RAW_MEASUREMENT_GROUP = "raw_measurement"
ENGINEERED_GROUP = "engineered"


@dataclass(frozen=True)
class FeatureDefinition:
    """One entry in the feature registry."""

    name: str
    dtype: str
    group: str
    description: str


# The registry lists features in exactly the column order produced by
# ``preprocessing.nodes._engineer_feature_frame``. ``tests/test_feature_store.py``
# asserts this stays in sync with the engineering code, so the two can never
# silently diverge.
FEATURE_DEFINITIONS: tuple[FeatureDefinition, ...] = (
    FeatureDefinition("ph", "float64", RAW_MEASUREMENT_GROUP, "pH level of the water sample on the 0-14 scale."),
    FeatureDefinition("Hardness", "float64", RAW_MEASUREMENT_GROUP, "Hardness from dissolved calcium and magnesium salts (mg/L)."),
    FeatureDefinition("Solids", "float64", RAW_MEASUREMENT_GROUP, "Total dissolved solids (ppm)."),
    FeatureDefinition("Chloramines", "float64", RAW_MEASUREMENT_GROUP, "Chloramine disinfectant concentration (ppm)."),
    FeatureDefinition("Sulfate", "float64", RAW_MEASUREMENT_GROUP, "Dissolved sulfate concentration (mg/L)."),
    FeatureDefinition("Conductivity", "float64", RAW_MEASUREMENT_GROUP, "Electrical conductivity of the water (uS/cm)."),
    FeatureDefinition("Organic_carbon", "float64", RAW_MEASUREMENT_GROUP, "Total organic carbon content (ppm)."),
    FeatureDefinition("Trihalomethanes", "float64", RAW_MEASUREMENT_GROUP, "Trihalomethane disinfection by-product concentration (ug/L)."),
    FeatureDefinition("Turbidity", "float64", RAW_MEASUREMENT_GROUP, "Water cloudiness from suspended solids (NTU)."),
    FeatureDefinition("conductivity_solids_ratio", "float64", ENGINEERED_GROUP, "Conductivity divided by total dissolved solids."),
    FeatureDefinition("turbidity_solids_ratio", "float64", ENGINEERED_GROUP, "Turbidity divided by total dissolved solids."),
    FeatureDefinition("hardness_conductivity_ratio", "float64", ENGINEERED_GROUP, "Hardness divided by conductivity."),
    FeatureDefinition("hardness_solids_ratio", "float64", ENGINEERED_GROUP, "Hardness divided by total dissolved solids."),
    FeatureDefinition("sulfate_hardness_ratio", "float64", ENGINEERED_GROUP, "Sulfate divided by hardness."),
    FeatureDefinition("tds_conductivity_ratio", "float64", ENGINEERED_GROUP, "Total dissolved solids divided by conductivity."),
    FeatureDefinition("chloramines_ph_interaction", "float64", ENGINEERED_GROUP, "Chloramines multiplied by pH."),
    FeatureDefinition("ph_hardness_interaction", "float64", ENGINEERED_GROUP, "pH multiplied by hardness."),
    FeatureDefinition("organic_trihalo_interaction", "float64", ENGINEERED_GROUP, "Organic carbon multiplied by trihalomethanes."),
    FeatureDefinition("turbidity_organic_interaction", "float64", ENGINEERED_GROUP, "Turbidity multiplied by organic carbon."),
    FeatureDefinition("trihalo_formation_risk", "float64", ENGINEERED_GROUP, "Organic carbon divided by trihalomethanes as a formation-potential proxy."),
    FeatureDefinition("disinfection_stress", "float64", ENGINEERED_GROUP, "Sum of sulfate and chloramines."),
    FeatureDefinition("dbp_precursor_load", "float64", ENGINEERED_GROUP, "Organic carbon plus one tenth of trihalomethanes."),
    FeatureDefinition("total_oxidant_stress", "float64", ENGINEERED_GROUP, "Chloramines plus one tenth of sulfate."),
    FeatureDefinition("solids_sulfate_diff", "float64", ENGINEERED_GROUP, "Total dissolved solids minus sulfate."),
    FeatureDefinition("turbidity_trihalo_risk", "int64", ENGINEERED_GROUP, "Binary flag: turbidity above 5 and trihalomethanes above 80."),
    FeatureDefinition("risk_score", "int64", ENGINEERED_GROUP, "Count of exceeded WHO-style safety thresholds (0-5)."),
    FeatureDefinition("expanded_risk_score", "int64", ENGINEERED_GROUP, "risk_score plus three additional threshold exceedances."),
    FeatureDefinition("ph_safe_range", "int64", ENGINEERED_GROUP, "Binary flag: pH within the safe 6.5-8.5 range."),
    FeatureDefinition("high_turbidity", "int64", ENGINEERED_GROUP, "Binary flag: turbidity above 5."),
    FeatureDefinition("high_sulfate", "int64", ENGINEERED_GROUP, "Binary flag: sulfate above 250."),
    FeatureDefinition("high_chloramines", "int64", ENGINEERED_GROUP, "Binary flag: chloramines above 4."),
    FeatureDefinition("high_hardness", "int64", ENGINEERED_GROUP, "Binary flag: hardness above 200."),
)

_DEFINITION_BY_NAME = {definition.name: definition for definition in FEATURE_DEFINITIONS}


def feature_names() -> list[str]:
    """Return every registered feature name in canonical order."""
    return [definition.name for definition in FEATURE_DEFINITIONS]


def raw_measurement_names() -> list[str]:
    """Return the raw measurement feature names in canonical order."""
    return [d.name for d in FEATURE_DEFINITIONS if d.group == RAW_MEASUREMENT_GROUP]


def engineered_feature_names() -> list[str]:
    """Return the engineered feature names in canonical order."""
    return [d.name for d in FEATURE_DEFINITIONS if d.group == ENGINEERED_GROUP]


def describe(name: str) -> FeatureDefinition:
    """Return the registry entry for one feature."""
    if name not in _DEFINITION_BY_NAME:
        raise KeyError(f"Unknown feature: {name!r}")
    return _DEFINITION_BY_NAME[name]


def validate_columns_against_registry(columns: Sequence[str]) -> None:
    """Raise if any column is not a registered feature."""
    unknown = [column for column in columns if column not in _DEFINITION_BY_NAME]
    if unknown:
        raise ValueError(f"Columns are not registered features: {unknown}")


def schema_version(frame: pd.DataFrame) -> str:
    """Return a deterministic version of the feature schema (names plus dtypes)."""
    schema = [{"name": str(column), "dtype": str(frame[column].dtype)} for column in frame.columns]
    payload = json.dumps(schema, sort_keys=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def snapshot_version(frame: pd.DataFrame) -> str:
    """Return a deterministic content-addressable version of the feature data."""
    row_hashes = pd.util.hash_pandas_object(frame, index=True).to_numpy()
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()[:12]


def build_feature_metadata(frame: pd.DataFrame, *, name: str) -> dict[str, Any]:
    """Build the deterministic metadata document for a materialised feature set.

    The document carries no wall-clock timestamp on purpose: versions are derived
    from the schema and the data only, so re-running the pipeline on the same
    inputs reproduces byte-identical metadata.
    """
    validate_columns_against_registry(frame.columns)
    columns = [
        {
            "name": str(column),
            "dtype": str(frame[column].dtype),
            "group": _DEFINITION_BY_NAME[column].group,
            "description": _DEFINITION_BY_NAME[column].description,
        }
        for column in frame.columns
    ]
    schema = schema_version(frame)
    snapshot = snapshot_version(frame)
    return {
        "name": name,
        "version": f"{schema}.{snapshot}",
        "schema_version": schema,
        "snapshot_version": snapshot,
        "row_count": int(len(frame)),
        "feature_count": int(frame.shape[1]),
        "columns": columns,
    }


def metadata_path_for(filepath: Path) -> Path:
    """Return the metadata sidecar path for a feature set file."""
    filepath = Path(filepath)
    return filepath.with_name(f"{filepath.stem}{METADATA_SUFFIX}")


def write_feature_metadata(frame: pd.DataFrame, filepath: Path, *, name: str) -> Path:
    """Write the metadata sidecar next to a materialised feature set file."""
    metadata = build_feature_metadata(frame, name=name)
    metadata_path = metadata_path_for(filepath)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata_path


def load_feature_metadata(filepath: Path) -> dict[str, Any]:
    """Load the metadata sidecar for a materialised feature set file."""
    metadata_path = metadata_path_for(filepath)
    if not metadata_path.exists():
        raise FileNotFoundError(f"No feature metadata found at {metadata_path}.")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_offline_features(
    name: str,
    feature_dir: Path = FEATURE_LAYER_DIR,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read a named feature set and its metadata from the offline store.

    This is the retrieval entry point used outside the Kedro catalog (tests,
    notebooks, ad-hoc inspection). ``name`` is the feature set name without an
    extension, e.g. ``"X_train"``.
    """
    feature_dir = Path(feature_dir)
    filepath = feature_dir / f"{name}.pkl"
    if not filepath.exists():
        raise FileNotFoundError(
            f"Feature set {name!r} not found at {filepath}. Run the preprocessing "
            "pipeline first to materialise the feature layer."
        )
    frame = pd.read_pickle(filepath)
    return frame, load_feature_metadata(filepath)
