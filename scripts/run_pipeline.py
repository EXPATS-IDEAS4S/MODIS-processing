#!/usr/bin/env python3
"""Run the MODIS pipeline one day at a time by calling the stage scripts."""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
import logging
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import yaml

LOGGER = logging.getLogger("run_pipeline")

DOWNLOAD_SCRIPT = "0_download_modis.py"
PROCESS_SCRIPT = "1_process_modis.py"
UPLOAD_SCRIPT = "2_upload_s3.py"

DOWNLOAD_RE = re.compile(
    r"(?P<satellite>terra|aqua) \[(?P<start>\d{4}-\d{2}-\d{2}) - (?P<end>\d{4}-\d{2}-\d{2})\] "
    r"L1 downloaded=(?P<l1_downloaded>\d+) skipped=(?P<l1_skipped>\d+) \| "
    r"L2 downloaded=(?P<l2_downloaded>\d+) skipped=(?P<l2_skipped>\d+)"
)
PROCESS_RE = re.compile(r"Processed (?P<count>\d+) files(?: for (?P<satellite>terra|aqua|combined satellites))?")
UPLOAD_RE = re.compile(r"Uploaded (?P<count>\d+) files")
ERROR_RE = re.compile(r"\bERROR\b|\bTraceback\b")


@dataclass
class StageMetrics:
    downloaded_l1: int = 0
    downloaded_l2: int = 0
    skipped_downloads: int = 0
    processed: int = 0
    uploaded: int = 0
    errors: int = 0


@dataclass
class BatchResult:
    day: dt.date
    download: StageMetrics = field(default_factory=StageMetrics)
    process: StageMetrics = field(default_factory=StageMetrics)
    upload: StageMetrics = field(default_factory=StageMetrics)
    download_seconds: float = 0.0
    process_seconds: float = 0.0
    upload_seconds: float = 0.0
    failed_stage: Optional[str] = None


def load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def iter_day_batches(years: Iterable[int], months: Iterable[int]) -> Iterator[dt.date]:
    for year in years:
        for month in months:
            _, days_in_month = calendar.monthrange(year, month)
            for day in range(1, days_in_month + 1):
                yield dt.date(year, month, day)


def _script_path(name: str) -> Path:
    path = Path(__file__).resolve().parent / name
    if not path.exists():
        raise FileNotFoundError(f"Pipeline script not found: {path}")
    return path


def _default_log_file(config: Dict) -> Path:
    output_base = Path(config["processing"]["output_base_path"])
    return output_base / "pipeline.log"


def _setup_logging(log_file: Path, log_level: str) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode="a", encoding="utf-8")]
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=handlers,
        force=True,
    )


def _increment_metrics(metrics: StageMetrics, line: str) -> None:
    match = DOWNLOAD_RE.search(line)
    if match:
        metrics.downloaded_l1 += int(match.group("l1_downloaded"))
        metrics.downloaded_l2 += int(match.group("l2_downloaded"))
        metrics.skipped_downloads += int(match.group("l1_skipped")) + int(match.group("l2_skipped"))
        return

    match = PROCESS_RE.search(line)
    if match:
        metrics.processed += int(match.group("count"))
        return

    match = UPLOAD_RE.search(line)
    if match:
        metrics.uploaded += int(match.group("count"))


def _run_command(stage: str, command: List[str], log_file: Path, dry_run: bool = False) -> tuple[int, StageMetrics, float]:
    LOGGER.info("Starting %s: %s", stage, " ".join(command))
    started = dt.datetime.now(dt.timezone.utc)
    metrics = StageMetrics()

    with log_file.open("a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n[{started.isoformat()}] START {stage}: {' '.join(command)}\n")
        log_handle.flush()

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip("\n")
            log_handle.write(f"[{stage}] {line}\n")
            log_handle.flush()
            LOGGER.info("[%s] %s", stage, line)
            _increment_metrics(metrics, line)
            if ERROR_RE.search(line):
                metrics.errors += 1

        return_code = process.wait()
        finished = dt.datetime.now(dt.timezone.utc)
        duration = (finished - started).total_seconds()
        log_handle.write(f"[{finished.isoformat()}] END {stage}: returncode={return_code} duration_seconds={duration:.2f}\n")
        log_handle.flush()

    LOGGER.info("Finished %s in %.2fs with return code %s", stage, duration, return_code)
    if metrics.errors:
        LOGGER.warning("%s produced %s error lines", stage, metrics.errors)
    if dry_run:
        LOGGER.info("%s ran in dry-run mode", stage)
    return return_code, metrics, duration


def _build_command(script_name: str, config_path: Path, credentials_path: Optional[Path], extra_flags: Iterable[str]) -> List[str]:
    command = [sys.executable, str(_script_path(script_name)), "--config", str(config_path)]
    if credentials_path is not None:
        command.extend(["--credentials", str(credentials_path)])
    command.extend(list(extra_flags))
    return command


def run_pipeline(config: Dict, config_path: Path, credentials_path: Path, dry_run: bool = False) -> None:
    pipeline = config.get("pipeline", {})
    years = config["years"]
    months = config["months"]
    log_file = Path(config.get("pipeline", {}).get("log_file", _default_log_file(config)))
    _setup_logging(log_file=log_file, log_level=str(config.get("pipeline", {}).get("log_level", "INFO")))

    download_enabled = bool(pipeline.get("run_download", True))
    process_enabled = bool(pipeline.get("run_processing", True))
    upload_enabled = bool(pipeline.get("run_upload", True))
    verify_upload = bool(pipeline.get("verify_upload", False))

    LOGGER.info("Pipeline log file: %s", log_file)
    LOGGER.info("Running daily batches for years=%s months=%s", years, months)

    results: List[BatchResult] = []
    for day in iter_day_batches(years=years, months=months):
        batch = BatchResult(day=day)
        LOGGER.info("=== Batch %s ===", day.isoformat())

        if download_enabled:
            command = _build_command(
                DOWNLOAD_SCRIPT,
                config_path=config_path,
                credentials_path=None,
                extra_flags=["--log-level", "INFO"] + (["--dry-run"] if dry_run else []),
            )
            return_code, metrics, duration = _run_command("download", command, log_file, dry_run=dry_run)
            batch.download = metrics
            batch.download_seconds = duration
            if return_code != 0:
                batch.failed_stage = "download"
                results.append(batch)
                LOGGER.error("Batch %s stopped after download failure", day.isoformat())
                continue

        if process_enabled:
            command = _build_command(
                PROCESS_SCRIPT,
                config_path=config_path,
                credentials_path=None,
                extra_flags=["--log-level", "INFO"] + (["--dry-run"] if dry_run else []),
            )
            return_code, metrics, duration = _run_command("process", command, log_file, dry_run=dry_run)
            batch.process = metrics
            batch.process_seconds = duration
            if return_code != 0:
                batch.failed_stage = "process"
                results.append(batch)
                LOGGER.error("Batch %s stopped after processing failure", day.isoformat())
                continue

        if upload_enabled:
            extra_flags = ["--log-level", "INFO"]
            if verify_upload:
                extra_flags.append("--verify")
            if dry_run:
                extra_flags.append("--dry-run")
            command = _build_command(
                UPLOAD_SCRIPT,
                config_path=config_path,
                credentials_path=credentials_path,
                extra_flags=extra_flags,
            )
            return_code, metrics, duration = _run_command("upload", command, log_file, dry_run=dry_run)
            batch.upload = metrics
            batch.upload_seconds = duration
            if return_code != 0:
                batch.failed_stage = "upload"
                results.append(batch)
                LOGGER.error("Batch %s completed with upload failure", day.isoformat())
                continue

        results.append(batch)
        LOGGER.info(
            "Batch %s finished in download=%.2fs process=%.2fs upload=%.2fs",
            day.isoformat(),
            batch.download_seconds,
            batch.process_seconds,
            batch.upload_seconds,
        )

    LOGGER.info("Pipeline finished for %s daily batches", len(results))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end MODIS pipeline day by day")
    parser.add_argument("--config", required=True, help="Path to YAML pipeline config")
    parser.add_argument(
        "--credentials",
        default="s3_credentials.py",
        help="Path to local python credentials file (excluded from git)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not write/download/upload files")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log-file", default=None, help="Path to unified pipeline log file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    if args.log_file is not None:
        config.setdefault("pipeline", {})["log_file"] = args.log_file
    config.setdefault("pipeline", {})["log_level"] = args.log_level
    run_pipeline(
        config=config,
        config_path=config_path,
        credentials_path=Path(args.credentials).resolve(),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
