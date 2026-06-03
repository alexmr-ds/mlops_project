"""CLI entrypoint for data setup and MLflow audits."""

import argparse
import sys
from pathlib import Path

from src.mlops_project import data_setup
from src.mlops_project import mlflow_secret_audit


def build_parser() -> argparse.ArgumentParser:
    """Build the project CLI parser."""
    parser = argparse.ArgumentParser(description="Utilities for the mlops-project repository.")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "setup-data",
        help="Create Kaggle credentials locally if needed and download the raw dataset.",
    )
    audit_parser = subparsers.add_parser(
        "audit-mlflow-secrets",
        help="Scan local MLflow stores for secret-like content without printing values.",
    )
    audit_parser.add_argument(
        "--tracking-dir",
        type=Path,
        default=Path("mlruns"),
        help="Path to the MLflow tracking directory.",
    )
    audit_parser.add_argument(
        "--db",
        type=Path,
        default=Path("mlflow.db"),
        help="Path to the MLflow SQLite snapshot.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the project CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "setup-data":
        try:
            data_setup.setup_local_data()
        except data_setup.DataSetupError as error:
            print(f"Error: {error}", file=sys.stderr)
            return 1
        return 0
    if args.command == "audit-mlflow-secrets":
        audit_summary = mlflow_secret_audit.audit_mlflow_stores(
            tracking_dir=args.tracking_dir,
            database_path=args.db,
        )
        print(f"Suspicious MLflow locations: {audit_summary.finding_count}")
        for finding in audit_summary.findings:
            print(f"{finding.category}: {finding.location}")
        return 0 if audit_summary.finding_count == 0 else 2

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
