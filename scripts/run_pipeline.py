#!/usr/bin/env python3
"""Orchestrate MODIS download -> processing -> upload per year."""

from __future__ import annotations

import argparse
import logging
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Dict

import yaml

from download_modis import run_download
from process_modis import run_processing
from upload_s3 import run_upload

LOGGER = logging.getLogger("run_pipeline")


def load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _cleanup(paths):
    for path in paths:
        if path.exists():
            shutil.rmtree(path)
            LOGGER.info("Deleted %s", path)


def run_pipeline(config: Dict, credentials_path: Path, dry_run: bool = False) -> None:
    pipeline = config["pipeline"]
    years = config["years"]

    for year in years:
        year_config = deepcopy(config)
        year_config["years"] = [year]
        LOGGER.info("=== Running pipeline for year=%s ===", year)

        if pipeline.get("run_download", True):
            run_download(year_config, dry_run=dry_run)

        if pipeline.get("run_processing", True):
            run_processing(year_config)
            if pipeline.get("delete_raw_after_processing", False):
                _cleanup(
                    [
                        Path(year_config["download"]["radiance_base_path"]) / f"{year:04d}",
                        Path(year_config["download"]["cloud_mask_base_path"]) / f"{year:04d}",
                    ]
                )

        if pipeline.get("run_upload", True):
            run_upload(
                year_config,
                credentials_path=credentials_path,
                verify=pipeline.get("verify_upload", False),
                dry_run=dry_run,
            )
            if pipeline.get("delete_processed_after_upload", False):
                _cleanup([Path(year_config["processing"]["output_base_path"]) / f"{year:04d}"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end MODIS pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML pipeline config")
    parser.add_argument(
        "--credentials",
        default="s3_credentials.py",
        help="Path to local python credentials file (excluded from git)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not write/download/upload files")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    config = load_config(Path(args.config))
    run_pipeline(config=config, credentials_path=Path(args.credentials), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
