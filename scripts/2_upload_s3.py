#!/usr/bin/env python3
"""Upload MODIS processed files to an S3-compatible object store.

The upload window can come from the YAML config (`years`, `months`, and
optional `days`) or from a single `--date YYYY-MM-DD` override.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import logging
from pathlib import Path
from typing import Dict, Iterable

import boto3
import yaml
from botocore.exceptions import ClientError

from scripts.utils.io import iter_days

LOGGER = logging.getLogger("upload_s3")


def load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_s3_credentials(credentials_path: Path) -> Dict[str, str]:
    credentials_path = credentials_path.resolve()
    if not credentials_path.exists():
        raise RuntimeError(f"Credentials file not found: {credentials_path}")

    spec = importlib.util.spec_from_file_location("s3_credentials_local", credentials_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load credentials from {credentials_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    required = ["S3_BUCKET_NAME", "S3_ACCESS_KEY", "S3_SECRET_ACCESS_KEY", "S3_ENDPOINT_URL"]
    missing = [item for item in required if not hasattr(module, item)]
    if missing:
        raise RuntimeError(f"Missing credentials in {credentials_path}: {missing}")
    return {
        "bucket": module.S3_BUCKET_NAME,
        "access_key": module.S3_ACCESS_KEY,
        "secret_key": module.S3_SECRET_ACCESS_KEY,
        "endpoint": module.S3_ENDPOINT_URL,
    }


def iter_files(base_path: Path, years: Iterable[int], months: Iterable[int], days: Iterable[int] | str | None = None, pattern: str = "*.nc") -> Iterable[Path]:
    for day_dir in iter_days(base_path=base_path, years=years, months=months, days=days):
        for file_path in sorted(day_dir.rglob(pattern)):
            if file_path.is_file():
                yield file_path


def upload_file(s3_client, file_path: Path, bucket: str, object_key: str) -> bool:
    try:
        with file_path.open("rb") as stream:
            s3_client.upload_fileobj(stream, bucket, object_key)
    except ClientError as exc:
        LOGGER.error("Upload failed for %s: %s", file_path, exc)
        return False
    return True


def verify_uploaded(s3_client, bucket: str, key: str) -> bool:
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False


def run_upload(config: Dict, credentials_path: Path, verify: bool = False, dry_run: bool = False, date: dt.date | None = None) -> None:
    creds = load_s3_credentials(credentials_path)
    s3_client = boto3.client(
        "s3",
        endpoint_url=creds["endpoint"],
        aws_access_key_id=creds["access_key"],
        aws_secret_access_key=creds["secret_key"],
    )

    local_base = Path(config["processing"]["output_base_path"])
    bucket_prefix = config["upload"].get("bucket_prefix", "").strip("/")
    years = [date.year] if date is not None else config["years"]
    months = [date.month] if date is not None else config["months"]
    days = [date.day] if date is not None else config.get("days", "all")

    uploaded = 0
    for file_path in iter_files(base_path=local_base, years=years, months=months, days=days):
        rel_path = file_path.relative_to(local_base).as_posix()
        key = f"{bucket_prefix}/{rel_path}" if bucket_prefix else rel_path

        if dry_run:
            LOGGER.info("[DRY-RUN] upload %s -> s3://<bucket>/%s", file_path, key)
            continue

        if upload_file(s3_client=s3_client, file_path=file_path, bucket=creds["bucket"], object_key=key):
            uploaded += 1
            if verify and not verify_uploaded(s3_client=s3_client, bucket=creds["bucket"], key=key):
                LOGGER.warning("Uploaded object not found during verification: %s", key)

    LOGGER.info("Uploaded %s files", uploaded)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload processed MODIS NetCDF files to S3")
    parser.add_argument("--config", required=True, help="Path to YAML pipeline config")
    parser.add_argument(
        "--credentials",
        default="s3_credentials.py",
        help="Path to local python credentials file (excluded from git)",
    )
    parser.add_argument("--date", default=None, help="Upload only one day in YYYY-MM-DD format")
    parser.add_argument("--verify", action="store_true", help="Verify each uploaded object with head_object")
    parser.add_argument("--dry-run", action="store_true", help="List files that would be uploaded")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    config = load_config(Path(args.config))
    selected_date = dt.date.fromisoformat(args.date) if args.date else None
    run_upload(config=config, credentials_path=Path(args.credentials), verify=args.verify, dry_run=args.dry_run, date=selected_date)


if __name__ == "__main__":
    main()
