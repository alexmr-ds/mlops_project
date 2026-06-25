"""Project root and dataset path helpers."""

from pathlib import Path


def _find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("Could not find project root")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)
RAW_DATA_DIR = PROJECT_ROOT / "data" / "01_raw"


def data_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)
