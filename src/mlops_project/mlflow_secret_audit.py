"""Detect likely secret material in local MLflow stores."""

from __future__ import annotations

from dataclasses import dataclass
import re
import sqlite3
from pathlib import Path


TEXT_FILE_SUFFIXES = {
    "",
    ".csv",
    ".json",
    ".lock",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
}

SUSPICIOUS_PATTERNS = (
    re.compile(r"api[_ -]?key", re.IGNORECASE),
    re.compile(r"access[_ -]?token", re.IGNORECASE),
    re.compile(r"refresh[_ -]?token", re.IGNORECASE),
    re.compile(r"authorization", re.IGNORECASE),
    re.compile(r"bearer\s+[A-Za-z0-9._-]{12,}", re.IGNORECASE),
    re.compile(r"secret", re.IGNORECASE),
    re.compile(r"password", re.IGNORECASE),
    re.compile(r"sk-[A-Za-z0-9_-]{12,}", re.IGNORECASE),
)

SQLITE_TEXT_TABLE_COLUMNS = (
    ("params", "key"),
    ("params", "value"),
    ("tags", "key"),
    ("tags", "value"),
    ("trace_request_metadata", "key"),
    ("trace_request_metadata", "value"),
    ("secrets", "secret_name"),
    ("secrets", "masked_value"),
    ("secrets", "auth_config"),
    ("secrets", "description"),
)


@dataclass(frozen=True)
class AuditFinding:
    """One suspicious location detected by the audit."""

    location: str
    category: str


@dataclass(frozen=True)
class AuditSummary:
    """Compact audit result for CLI and tests."""

    findings: list[AuditFinding]

    @property
    def finding_count(self) -> int:
        """Return the number of suspicious locations."""
        return len(self.findings)


def audit_mlflow_stores(
    *,
    tracking_dir: Path,
    database_path: Path,
) -> AuditSummary:
    """Scan MLflow file and SQLite stores for secret-like content."""
    findings: list[AuditFinding] = []
    findings.extend(audit_mlruns_text_files(tracking_dir))
    findings.extend(audit_mlflow_database(database_path))
    findings.sort(key=lambda finding: (finding.location, finding.category))
    return AuditSummary(findings=findings)


def audit_mlruns_text_files(tracking_dir: Path) -> list[AuditFinding]:
    """Scan text-like MLflow artifact files without echoing contents."""
    if not tracking_dir.exists():
        return []

    findings: list[AuditFinding] = []
    for file_path in sorted(path for path in tracking_dir.rglob("*") if path.is_file()):
        if not is_text_like_path(file_path):
            continue
        try:
            file_contents = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if contains_suspicious_text(file_contents):
            findings.append(
                AuditFinding(
                    location=str(file_path),
                    category="mlruns_text_match",
                )
            )
    return findings


def audit_mlflow_database(database_path: Path) -> list[AuditFinding]:
    """Scan text columns in the MLflow SQLite snapshot."""
    if not database_path.exists():
        return []

    findings: list[AuditFinding] = []
    with sqlite3.connect(database_path) as connection:
        for table_name, column_name in SQLITE_TEXT_TABLE_COLUMNS:
            if column_name not in existing_columns(connection, table_name):
                continue
            query = (
                f"SELECT rowid, {column_name} FROM {table_name} "
                f"WHERE {column_name} IS NOT NULL"
            )
            for row_id, value in connection.execute(query):
                if isinstance(value, str) and contains_suspicious_text(value):
                    findings.append(
                        AuditFinding(
                            location=f"{table_name}.{column_name}[rowid={row_id}]",
                            category="sqlite_text_match",
                        )
                    )
    return findings


def existing_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    """Return known columns for the requested SQLite table."""
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row[1] for row in rows}


def is_text_like_path(file_path: Path) -> bool:
    """Return whether a file should be scanned as UTF-8 text."""
    return file_path.suffix.lower() in TEXT_FILE_SUFFIXES


def contains_suspicious_text(value: str) -> bool:
    """Return whether any configured secret-like pattern matches."""
    return any(pattern.search(value) for pattern in SUSPICIOUS_PATTERNS)
