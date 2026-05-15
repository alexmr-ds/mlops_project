"""Pipeline tests package."""

from pathlib import Path
import sys


def _ensure_src_on_syspath() -> None:
    """Make the src layout importable during unittest discovery."""
    project_root = Path(__file__).resolve().parents[2]
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_ensure_src_on_syspath()
