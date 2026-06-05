#!/usr/bin/env python3
"""Run the MODIS pipeline one day at a time.

This script reads the YAML pipeline config, expands the configured `years`,
`months`, and optional `days` into daily batches, and for each day calls the stage scripts in order:
`0_download_modis.py`, `1_process_modis.py`, and `2_upload_s3.py`.

Each subprocess receives an explicit `--date YYYY-MM-DD` so standalone stage
invocations can still fall back to config-driven selection when no date is passed.

All stage output is streamed into one unified log file, which defaults to
`processing.output_base_path/pipeline.log` unless overridden with `--log-file`.
The runner also records per-batch timing, download/process/upload counts, and
any failures it encounters.

Usage:
    python scripts/run_pipeline.py --config config/pipeline_config.yaml \
        --credentials s3_credentials.py

Useful options:
    --dry-run    pass dry-run mode to the stages
    --log-file   write the unified pipeline log to a custom path
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import re
import traceback
import subprocess
import sys
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional
from collections import defaultdict

from scripts.utils.io import iter_selected_dates
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


def iter_day_batches(years: Iterable[int], months: Iterable[int], days: Optional[Iterable[int] | str] = None) -> Iterator[dt.date]:
    yield from iter_selected_dates(years=years, months=months, days=days)


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


def _make_file_handler(log_file: Path) -> logging.Handler:
    handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    )
    return handler


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


def _run_command(stage: str, command: List[str], log_file: Path, error_log: Path, dry_run: bool = False) -> tuple[int, StageMetrics, float]:
    LOGGER.info("Starting %s: %s", stage, " ".join(command))
    started = dt.datetime.now(dt.timezone.utc)
    metrics = StageMetrics()
    output_lines: List[str] = []

    try:
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
                output_lines.append(line)
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

    except Exception:
        # Capture internal runner exceptions (e.g., Popen failures)
        tb = traceback.format_exc()
        with error_log.open("a", encoding="utf-8") as errf:
            errf.write(f"\n[{dt.datetime.now(dt.timezone.utc).isoformat()}] EXCEPTION in {stage}: {' '.join(command)}\n")
            errf.write(tb)
            errf.write("\n--- End Exception ---\n")
            errf.flush()
        LOGGER.exception("Exception while running %s", stage)
        return 1, metrics, 0.0

    LOGGER.info("Finished %s in %.2fs with return code %s", stage, duration, return_code)
    if metrics.errors:
        LOGGER.warning("%s produced %s error lines", stage, metrics.errors)

    # If the stage failed or there were error lines, write the full output to the error log
    if return_code != 0 or metrics.errors:
        try:
            with error_log.open("a", encoding="utf-8") as errf:
                errf.write(f"\n[{dt.datetime.now(dt.timezone.utc).isoformat()}] {stage} summary: returncode={return_code} errors={metrics.errors}\n")
                errf.write(f"Command: {' '.join(command)}\n")
                errf.write("--- BEGIN OUTPUT ---\n")
                for ol in output_lines:
                    errf.write(ol + "\n")
                errf.write("--- END OUTPUT ---\n")
                errf.flush()
            LOGGER.info("Wrote full %s output to %s", stage, error_log)
        except Exception:
            LOGGER.exception("Failed to write error log for %s", stage)

    if dry_run:
        LOGGER.info("%s ran in dry-run mode", stage)
    return return_code, metrics, duration


def _build_command(
    script_name: str,
    config_path: Path,
    credentials_path: Optional[Path],
    extra_flags: Iterable[str],
    day: Optional[dt.date] = None,
) -> List[str]:
    command = [sys.executable, str(_script_path(script_name)), "--config", str(config_path)]
    if credentials_path is not None:
        command.extend(["--credentials", str(credentials_path)])
    if day is not None:
        command.extend(["--date", day.isoformat()])
    command.extend(list(extra_flags))
    return command




def _delete_day_folder(base_path: Path, day: dt.date, label: str) -> None:
    """Delete YYYY/MM/DD folder for a given day if it exists."""
    day_dir = (
        base_path
        / f"{day.year:04d}"
        / f"{day.month:02d}"
        / f"{day.day:02d}"
    )

    if day_dir.exists():
        LOGGER.info("Deleting %s data: %s", label, day_dir)
        shutil.rmtree(day_dir)
    else:
        LOGGER.warning("%s folder not found: %s", label, day_dir)


def run_pipeline(config: Dict, config_path: Path, credentials_path: Path, dry_run: bool = False) -> None:
    pipeline = config.get("pipeline", {})
    years = config["years"]
    months = config["months"]
    days = config.get("days", "all")
    log_level = str(pipeline.get("log_level", "INFO"))
    log_file = Path(config.get("pipeline", {}).get("log_file", _default_log_file(config)))
    _setup_logging(log_file=log_file, log_level=log_level)
    log_dir = log_file.parent

    download_enabled = bool(pipeline.get("run_download", True))
    process_enabled = bool(pipeline.get("run_processing", True))
    upload_enabled = bool(pipeline.get("run_upload", True))
    verify_upload = bool(pipeline.get("verify_upload", False))

    delete_raw_after_processing = bool(
    pipeline.get("delete_raw_after_processing", False)
    )

    delete_processed_after_upload = bool(
        pipeline.get("delete_processed_after_upload", False)
    )

    radiance_base = Path(config["download"]["radiance_base_path"])
    cloud_mask_base = Path(config["download"]["cloud_mask_base_path"])
    processed_base = Path(config["processing"]["output_base_path"])

    
    year_stats = defaultdict(
        lambda: {
            "downloaded_l1": 0,
            "downloaded_l2": 0,
            "processed": 0,
            "uploaded": 0,
            "errors": 0,
            "days": 0,
            "duration": 0.0,
        }
    )
    
    
    LOGGER.info("Pipeline log file: %s", log_file)
    LOGGER.info("Running daily batches for years=%s months=%s days=%s", years, months, days)

    results: List[BatchResult] = []
    total_downloaded_l1 = 0
    total_downloaded_l2 = 0
    total_processed = 0
    total_uploaded = 0
    total_errors = 0
    # derive error log path (near the main pipeline log)
    error_log = log_file.parent / f"{log_file.stem}_errors{log_file.suffix}"

    current_year = None

    active_log_file = log_file
    active_error_log = error_log
    year_handler: Optional[logging.Handler] = None

    def emit_year_summary(year: int) -> None:
        ys = year_stats[year]
        LOGGER.info(
            (
                "YEAR %s SUMMARY | days_processed=%s downloaded_l1=%s downloaded_l2=%s "
                "processed=%s uploaded=%s errors=%s duration_hours=%.2f"
            ),
            year,
            ys["days"],
            ys["downloaded_l1"],
            ys["downloaded_l2"],
            ys["processed"],
            ys["uploaded"],
            ys["errors"],
            ys["duration"] / 3600.0,
        )

    def close_year_handler() -> None:
        nonlocal year_handler
        if year_handler is not None:
            root_logger = logging.getLogger()
            root_logger.removeHandler(year_handler)
            year_handler.close()
            year_handler = None

    for day in iter_day_batches(years=years, months=months, days=days):
        if current_year != day.year:
            if current_year is not None:
                emit_year_summary(current_year)
                close_year_handler()

            current_year = day.year
            year_log = log_dir / f"pipeline_{current_year}.log"
            year_handler = _make_file_handler(year_log)
            year_handler.setLevel(getattr(logging, log_level))
            logging.getLogger().addHandler(year_handler)

            LOGGER.info(
                "Starting logging for year %s",
                current_year,
            )

            active_log_file = year_log
            active_error_log = active_log_file.parent / f"{active_log_file.stem}_errors.log"

        batch = BatchResult(day=day)
        LOGGER.info("=== Batch %s ===", day.isoformat())

        if download_enabled:
            command = _build_command(
                DOWNLOAD_SCRIPT,
                config_path=config_path,
                credentials_path=None,
                extra_flags=["--log-level", log_level] + (["--dry-run"] if dry_run else []),
                day=day,
            )
            return_code, metrics, duration = _run_command("download", command, active_log_file, active_error_log, dry_run=dry_run)
            batch.download = metrics
            batch.download_seconds = duration
            total_downloaded_l1 += metrics.downloaded_l1
            total_downloaded_l2 += metrics.downloaded_l2
            total_errors += metrics.errors
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
                extra_flags=["--log-level", log_level] + (["--dry-run"] if dry_run else []),
                day=day,
            )
            return_code, metrics, duration = _run_command("process", command, active_log_file, active_error_log, dry_run=dry_run)
            batch.process = metrics
            batch.process_seconds = duration
            total_processed += metrics.processed
            total_errors += metrics.errors
            if return_code != 0:
                batch.failed_stage = "process"
                results.append(batch)
                LOGGER.error("Batch %s stopped after processing failure", day.isoformat())
                continue

            # Delete raw inputs only after successful processing
            if delete_raw_after_processing and not dry_run:
                try:
                    _delete_day_folder(radiance_base, day, "raw radiance")
                    _delete_day_folder(cloud_mask_base, day, "raw cloud mask")
                except Exception:
                    LOGGER.exception(
                        "Failed deleting raw files for %s",
                        day.isoformat(),
                    )

        if upload_enabled:
            extra_flags = ["--log-level", log_level]
            if verify_upload:
                extra_flags.append("--verify")
            if dry_run:
                extra_flags.append("--dry-run")
            command = _build_command(
                UPLOAD_SCRIPT,
                config_path=config_path,
                credentials_path=credentials_path,
                extra_flags=extra_flags,
                day=day,
            )
            return_code, metrics, duration = _run_command("upload", command, active_log_file, active_error_log, dry_run=dry_run)
            batch.upload = metrics
            batch.upload_seconds = duration
            total_uploaded += metrics.uploaded
            total_errors += metrics.errors
            if return_code != 0:
                batch.failed_stage = "upload"
                results.append(batch)
                LOGGER.error("Batch %s completed with upload failure", day.isoformat())
                continue

            # Delete processed outputs only after successful upload
            if delete_processed_after_upload and not dry_run:
                try:
                    _delete_day_folder(processed_base, day, "processed")
                except Exception:
                    LOGGER.exception(
                        "Failed deleting processed files for %s",
                        day.isoformat(),
                    )

        results.append(batch)
        LOGGER.info(
            "Batch %s finished in download=%.2fs process=%.2fs upload=%.2fs | L1=%s L2=%s processed=%s uploaded=%s errors=%s failed_stage=%s",
            day.isoformat(),
            batch.download_seconds,
            batch.process_seconds,
            batch.upload_seconds,
            batch.download.downloaded_l1,
            batch.download.downloaded_l2,
            batch.process.processed,
            batch.upload.uploaded,
            batch.download.errors + batch.process.errors + batch.upload.errors,
            batch.failed_stage or "none",
        )

        ys = year_stats[day.year]

        ys["downloaded_l1"] += batch.download.downloaded_l1
        ys["downloaded_l2"] += batch.download.downloaded_l2
        ys["processed"] += batch.process.processed
        ys["uploaded"] += batch.upload.uploaded

        ys["errors"] += (
            batch.download.errors
            + batch.process.errors
            + batch.upload.errors
        )

        ys["days"] += 1

        ys["duration"] += (
            batch.download_seconds
            + batch.process_seconds
            + batch.upload_seconds
        )

        if dry_run:
            LOGGER.info("Dry-run requested, stopping after first batch %s", day.isoformat())
            break

    if current_year is not None:
        emit_year_summary(current_year)
        close_year_handler()

    total_duration = sum(
    ys["duration"]
    for ys in year_stats.values()
    )
    
    
    
    
    LOGGER.info(
        "Pipeline finished for %s daily batches | total_downloaded_l1=%s total_downloaded_l2=%s total_processed=%s total_uploaded=%s total_errors=%s",
        len(results),
        total_downloaded_l1,
        total_downloaded_l2,
        total_processed,
        total_uploaded,
        total_errors,
    )

    LOGGER.info(
        (
            "PIPELINE SUMMARY | years=%s downloaded_l1=%s downloaded_l2=%s "
            "processed=%s uploaded=%s errors=%s duration_hours=%.2f"
        ),
        len(year_stats),
        total_downloaded_l1,
        total_downloaded_l2,
        total_processed,
        total_uploaded,
        total_errors,
        total_duration / 3600,
    )

    


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
