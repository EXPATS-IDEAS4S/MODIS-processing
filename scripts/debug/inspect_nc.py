#!/usr/bin/env python3
"""Inspect a processed NetCDF file and print basic statistics."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from netcdf_inspect import inspect_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a MODIS NetCDF file")
    parser.add_argument("--file", required=True, help="Path to a NetCDF file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inspect_file(Path(args.file))


if __name__ == "__main__":
    main()
