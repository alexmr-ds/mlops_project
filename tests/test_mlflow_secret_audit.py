"""Unit tests for the MLflow secret audit helper."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.mlops_project import mlflow_secret_audit


class MlflowSecretAuditTests(unittest.TestCase):
    """Tests for MLflow file and SQLite secret scanning."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.tracking_dir = self.root / "mlruns"
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        self.database_path = self.root / "mlflow.db"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_audit_detects_text_file_findings(self) -> None:
        run_dir = self.tracking_dir / "0" / "run-1" / "params"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "api_key").write_text("sk-test-1234567890abcd", encoding="utf-8")

        audit_summary = mlflow_secret_audit.audit_mlflow_stores(
            tracking_dir=self.tracking_dir,
            database_path=self.database_path,
        )

        self.assertEqual(audit_summary.finding_count, 1)
        self.assertEqual(audit_summary.findings[0].category, "mlruns_text_match")

    def test_audit_detects_sqlite_findings(self) -> None:
        with sqlite3.connect(self.database_path) as connection:
            connection.execute("CREATE TABLE params (key TEXT, value TEXT)")
            connection.execute(
                "INSERT INTO params (key, value) VALUES (?, ?)",
                ("openai_api_key", "masked"),
            )
            connection.commit()

        audit_summary = mlflow_secret_audit.audit_mlflow_stores(
            tracking_dir=self.tracking_dir,
            database_path=self.database_path,
        )

        self.assertEqual(audit_summary.finding_count, 1)
        self.assertEqual(audit_summary.findings[0].category, "sqlite_text_match")

    def test_audit_returns_clean_summary_for_safe_state(self) -> None:
        safe_dir = self.tracking_dir / "0" / "run-1" / "metrics"
        safe_dir.mkdir(parents=True, exist_ok=True)
        (safe_dir / "accuracy").write_text("0.95", encoding="utf-8")

        with sqlite3.connect(self.database_path) as connection:
            connection.execute("CREATE TABLE tags (key TEXT, value TEXT)")
            connection.execute(
                "INSERT INTO tags (key, value) VALUES (?, ?)",
                ("mlflow.runName", "baseline"),
            )
            connection.commit()

        audit_summary = mlflow_secret_audit.audit_mlflow_stores(
            tracking_dir=self.tracking_dir,
            database_path=self.database_path,
        )

        self.assertEqual(audit_summary.finding_count, 0)


if __name__ == "__main__":
    unittest.main()
