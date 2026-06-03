"""Unit tests for local data bootstrap and CLI behavior."""

from __future__ import annotations

import io
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock

import main
from src.mlops_project import data_setup


class DataSetupTests(unittest.TestCase):
    """Tests for Kaggle credential and dataset bootstrap flows."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.home_dir = Path(self.temp_dir.name) / "home"
        self.project_root = Path(self.temp_dir.name) / "project"
        self.raw_data_dir = self.project_root / "data" / "raw"
        self.home_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_write_kaggle_credentials_creates_file_with_expected_permissions(self) -> None:
        credentials_path = data_setup.kaggle_credentials_path(self.home_dir)
        credentials = data_setup.KaggleCredentials(username="alice", key="secret")

        data_setup.write_kaggle_credentials(credentials_path, credentials)

        self.assertTrue(credentials_path.exists())
        self.assertEqual(
            data_setup.load_kaggle_credentials(credentials_path),
            credentials,
        )
        self.assertEqual(stat.S_IMODE(credentials_path.stat().st_mode), 0o600)

    def test_ensure_local_kaggle_credentials_prompts_and_creates_file(self) -> None:
        prompts = []

        def fake_input(prompt: str) -> str:
            prompts.append(prompt)
            if "Create" in prompt:
                return "y"
            return "alice"

        credentials_path = data_setup.ensure_local_kaggle_credentials(
            home_dir=self.home_dir,
            input_func=fake_input,
            secret_input_func=lambda _: "secret",
        )

        self.assertTrue(credentials_path.exists())
        self.assertGreaterEqual(len(prompts), 2)
        self.assertEqual(
            data_setup.load_kaggle_credentials(credentials_path),
            data_setup.KaggleCredentials(username="alice", key="secret"),
        )

    def test_ensure_local_kaggle_credentials_rejects_missing_confirmation(self) -> None:
        with self.assertRaises(data_setup.DataSetupError):
            data_setup.ensure_local_kaggle_credentials(
                home_dir=self.home_dir,
                input_func=lambda _: "n",
                secret_input_func=lambda _: "secret",
            )

    def test_ensure_local_kaggle_credentials_fixes_permissions_for_existing_file(self) -> None:
        credentials_path = data_setup.kaggle_credentials_path(self.home_dir)
        data_setup.write_kaggle_credentials(
            credentials_path,
            data_setup.KaggleCredentials(username="alice", key="secret"),
        )
        credentials_path.chmod(0o644)

        data_setup.ensure_local_kaggle_credentials(home_dir=self.home_dir)

        self.assertEqual(stat.S_IMODE(credentials_path.stat().st_mode), 0o600)

    def test_setup_local_data_returns_early_when_dataset_exists(self) -> None:
        dataset_file = self.raw_data_dir / data_setup.DATASET_FILENAME
        dataset_file.write_text("existing", encoding="utf-8")

        with mock.patch.object(data_setup.project_paths, "RAW_DATA_DIR", self.raw_data_dir):
            result = data_setup.setup_local_data(home_dir=self.home_dir)

        self.assertEqual(result, dataset_file)

    def test_setup_local_data_creates_credentials_and_downloads_dataset(self) -> None:
        dataset_file = self.raw_data_dir / data_setup.DATASET_FILENAME

        class FakeApi:
            def authenticate(self) -> None:
                return None

            def dataset_download_files(self, slug: str, path: Path, unzip: bool) -> None:
                self.slug = slug
                self.path = path
                self.unzip = unzip
                dataset_file.write_text("downloaded", encoding="utf-8")

        fake_api = FakeApi()

        with mock.patch.object(data_setup.project_paths, "RAW_DATA_DIR", self.raw_data_dir):
            result = data_setup.setup_local_data(
                home_dir=self.home_dir,
                input_func=lambda prompt: "y" if "Create" in prompt else "alice",
                secret_input_func=lambda _: "secret",
                api_factory=lambda: fake_api,
            )

        self.assertEqual(result, dataset_file)
        self.assertEqual(fake_api.slug, data_setup.KAGGLE_DATASET_SLUG)
        self.assertEqual(fake_api.path, self.raw_data_dir)
        self.assertTrue(fake_api.unzip)

    def test_setup_local_data_raises_when_download_does_not_create_file(self) -> None:
        class FakeApi:
            def authenticate(self) -> None:
                return None

            def dataset_download_files(self, slug: str, path: Path, unzip: bool) -> None:
                return None

        with mock.patch.object(data_setup.project_paths, "RAW_DATA_DIR", self.raw_data_dir):
            with self.assertRaises(data_setup.DataSetupError):
                data_setup.setup_local_data(
                    home_dir=self.home_dir,
                    input_func=lambda prompt: "y" if "Create" in prompt else "alice",
                    secret_input_func=lambda _: "secret",
                    api_factory=FakeApi,
                )


class MainCliTests(unittest.TestCase):
    """Tests for the top-level CLI entrypoint."""

    def test_main_runs_setup_data_command(self) -> None:
        with mock.patch.object(main.data_setup, "setup_local_data") as mock_setup:
            exit_code = main.main(["setup-data"])

        mock_setup.assert_called_once_with()
        self.assertEqual(exit_code, 0)

    def test_main_returns_error_code_on_setup_failure(self) -> None:
        stderr = io.StringIO()
        with mock.patch.object(
            main.data_setup,
            "setup_local_data",
            side_effect=data_setup.DataSetupError("boom"),
        ):
            with mock.patch("sys.stderr", stderr):
                exit_code = main.main(["setup-data"])

        self.assertEqual(exit_code, 1)
        self.assertIn("Error: boom", stderr.getvalue())

    def test_main_reports_clean_mlflow_audit(self) -> None:
        stdout = io.StringIO()
        audit_summary = main.mlflow_secret_audit.AuditSummary(findings=[])
        with mock.patch.object(
            main.mlflow_secret_audit,
            "audit_mlflow_stores",
            return_value=audit_summary,
        ) as mock_audit:
            with mock.patch("sys.stdout", stdout):
                exit_code = main.main(["audit-mlflow-secrets"])

        mock_audit.assert_called_once()
        self.assertEqual(exit_code, 0)
        self.assertIn("Suspicious MLflow locations: 0", stdout.getvalue())

    def test_main_returns_nonzero_when_mlflow_audit_finds_matches(self) -> None:
        stdout = io.StringIO()
        audit_summary = main.mlflow_secret_audit.AuditSummary(
            findings=[
                main.mlflow_secret_audit.AuditFinding(
                    location="params.value[rowid=1]",
                    category="sqlite_text_match",
                )
            ]
        )
        with mock.patch.object(
            main.mlflow_secret_audit,
            "audit_mlflow_stores",
            return_value=audit_summary,
        ):
            with mock.patch("sys.stdout", stdout):
                exit_code = main.main(["audit-mlflow-secrets"])

        self.assertEqual(exit_code, 2)
        self.assertIn("sqlite_text_match: params.value[rowid=1]", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
