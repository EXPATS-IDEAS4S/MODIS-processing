#!/usr/bin/env python3
"""Verify that processed NetCDF files exist in an S3 bucket."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from scripts.debug_tools import verify_s3_uploads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify processed NetCDF files in S3")
    parser.add_argument("--local-base", required=True, help="Local base directory containing NetCDF files")
    parser.add_argument("--credentials", default="s3_credentials.py", help="Path to S3 credentials file")
    parser.add_argument("--bucket-prefix", default="", help="Optional prefix inside the S3 bucket")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_s3_uploads(Path(args.local_base), Path(args.credentials), args.bucket_prefix)


if __name__ == "__main__":
    main()
