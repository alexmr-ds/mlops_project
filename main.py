"""CLI entrypoint for the project."""

import argparse
import sys

from src.mlops_project import data_setup


def build_parser() -> argparse.ArgumentParser:
    """Build the project CLI parser."""
    parser = argparse.ArgumentParser(description="Utilities for the mlops-project repository.")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "setup-data",
        help="Create Kaggle credentials locally if needed and download the raw dataset.",
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

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
