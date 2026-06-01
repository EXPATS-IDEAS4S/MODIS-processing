#!/usr/bin/env python3
"""Plot a single processed file or a NetCDF quicklook image."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from scripts.debug_tools import plot_file, plot_nc_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a single MODIS file")
    parser.add_argument("--file", required=True, help="Path to a file to plot")
    parser.add_argument("--output-dir", required=True, help="Directory for generated PNGs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.file)
    output_dir = Path(args.output_dir)
    if path.suffix.lower() == ".nc":
        plot_nc_file(path, output_dir)
    else:
        plot_file(path, output_dir)


if __name__ == "__main__":
    main()
