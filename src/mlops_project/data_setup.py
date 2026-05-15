"""Interactive local data bootstrap for the Kaggle dataset."""

from __future__ import annotations

from dataclasses import dataclass
from getpass import getpass
import json
from pathlib import Path
import stat

from src import project_paths

KAGGLE_API_URL = "https://www.kaggle.com/settings/api"
KAGGLE_DATASET_SLUG = "adityakadiwal/water-potability"
DATASET_FILENAME = "water_potability.csv"


class DataSetupError(RuntimeError):
    """Raised when local data setup cannot be completed."""


@dataclass(frozen=True)
class KaggleCredentials:
    """Validated Kaggle API credentials."""

    username: str
    key: str


def kaggle_credentials_path(home_dir: Path | None = None) -> Path:
    """Return the standard Kaggle credentials path."""
    base_dir = home_dir if home_dir is not None else Path.home()
    return base_dir / ".kaggle" / "kaggle.json"


def dataset_path() -> Path:
    """Return the expected raw dataset path."""
    return project_paths.RAW_DATA_DIR / DATASET_FILENAME


def prompt_to_create_credentials(
    credentials_path: Path,
    input_func=input,
    secret_input_func=getpass,
) -> KaggleCredentials | None:
    """Offer to create Kaggle credentials locally after explicit confirmation."""
    print("Kaggle API credentials are required to download the dataset.")
    print(f"Generate them from: {KAGGLE_API_URL}")
    response = input_func(
        f"Create {credentials_path} now? [y/N]: "
    ).strip().lower()
    if response not in {"y", "yes"}:
        return None

    username = input_func("Kaggle username: ").strip()
    key = secret_input_func("Kaggle API key: ").strip()
    if not username or not key:
        raise DataSetupError("Kaggle username and key are both required.")
    return KaggleCredentials(username=username, key=key)


def write_kaggle_credentials(credentials_path: Path, credentials: KaggleCredentials) -> None:
    """Write Kaggle credentials and lock down file permissions."""
    credentials_path.parent.mkdir(parents=True, exist_ok=True)
    credentials_path.write_text(
        json.dumps(
            {"username": credentials.username, "key": credentials.key},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    ensure_kaggle_permissions(credentials_path)


def ensure_kaggle_permissions(credentials_path: Path) -> None:
    """Restrict Kaggle credentials to the current user."""
    credentials_path.chmod(0o600)


def load_kaggle_credentials(credentials_path: Path) -> KaggleCredentials:
    """Load and validate Kaggle credentials from disk."""
    try:
        payload = json.loads(credentials_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise DataSetupError(
            f"Missing Kaggle credentials at {credentials_path}. Generate an API key at "
            f"{KAGGLE_API_URL}."
        ) from error
    except json.JSONDecodeError as error:
        raise DataSetupError(
            f"Invalid JSON in {credentials_path}. Recreate it from {KAGGLE_API_URL}."
        ) from error

    username = str(payload.get("username", "")).strip()
    key = str(payload.get("key", "")).strip()
    if not username or not key:
        raise DataSetupError(
            f"{credentials_path} must contain non-empty 'username' and 'key' fields."
        )
    return KaggleCredentials(username=username, key=key)


def credential_permissions(credentials_path: Path) -> int:
    """Return the current credential file mode."""
    return stat.S_IMODE(credentials_path.stat().st_mode)


def ensure_local_kaggle_credentials(
    home_dir: Path | None = None,
    input_func=input,
    secret_input_func=getpass,
) -> Path:
    """Ensure Kaggle credentials exist locally and are valid."""
    credentials_path = kaggle_credentials_path(home_dir)
    if credentials_path.exists():
        load_kaggle_credentials(credentials_path)
        if credential_permissions(credentials_path) != 0o600:
            print(f"Updating permissions for {credentials_path} to 600.")
            ensure_kaggle_permissions(credentials_path)
        return credentials_path

    credentials = prompt_to_create_credentials(
        credentials_path,
        input_func=input_func,
        secret_input_func=secret_input_func,
    )
    if credentials is None:
        raise DataSetupError(
            "Kaggle credentials were not created. Generate an API key at "
            f"{KAGGLE_API_URL} and rerun `uv run python main.py setup-data`."
        )

    write_kaggle_credentials(credentials_path, credentials)
    return credentials_path


def create_kaggle_api(api_factory=None):
    """Create a Kaggle API client."""
    if api_factory is not None:
        return api_factory()

    from kaggle.api.kaggle_api_extended import KaggleApi

    return KaggleApi()


def setup_local_data(
    home_dir: Path | None = None,
    input_func=input,
    secret_input_func=getpass,
    api_factory=None,
) -> Path:
    """Ensure Kaggle credentials and the local raw dataset are available."""
    raw_dataset_path = dataset_path()
    project_paths.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if raw_dataset_path.exists():
        print(f"Dataset already present at {raw_dataset_path}.")
        return raw_dataset_path

    credentials_path = ensure_local_kaggle_credentials(
        home_dir=home_dir,
        input_func=input_func,
        secret_input_func=secret_input_func,
    )
    print(f"Using Kaggle credentials from {credentials_path}.")

    api = create_kaggle_api(api_factory=api_factory)
    try:
        api.authenticate()
        api.dataset_download_files(
            KAGGLE_DATASET_SLUG,
            path=project_paths.RAW_DATA_DIR,
            unzip=True,
        )
    except Exception as error:  # pragma: no cover - exercised via mocks in tests
        raise DataSetupError(
            "Failed to download the Kaggle dataset. Verify your credentials and network "
            "access, then try again."
        ) from error

    if not raw_dataset_path.exists():
        raise DataSetupError(
            f"Download completed but {raw_dataset_path} was not created."
        )

    print(f"Dataset downloaded to {raw_dataset_path}.")
    return raw_dataset_path
