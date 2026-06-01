#!/usr/bin/env python3
"""Run debug scripts sequentially or in parallel by group."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

SCRIPT_DIR = Path(__file__).resolve().parent

TASKS = {
    "inspect": SCRIPT_DIR / "inspect_nc.py",
    "plot": SCRIPT_DIR / "plot_nc.py",
    "plot-nc-day": SCRIPT_DIR / "plot_nc_day.py",
    "plot-hdf-day": SCRIPT_DIR / "plot_hdf_day.py",
    "verify-s3": SCRIPT_DIR / "verify_s3.py",
}

GROUPS = {
    "file": ["inspect", "plot"],
    "daily": ["plot-nc-day", "plot-hdf-day"],
    "s3": ["verify-s3"],
    "all": ["inspect", "plot", "plot-nc-day", "plot-hdf-day", "verify-s3"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run debug scripts by group")
    parser.add_argument("--group", action="append", choices=sorted(GROUPS.keys()), required=True, help="Task group to run; can be repeated")
    parser.add_argument("--parallel", action="store_true", help="Run selected tasks simultaneously")

    parser.add_argument("--file", help="Path to a NetCDF or raw file for inspect/plot")
    parser.add_argument("--output-dir", help="Output directory for inspect/plot quicklooks")

    parser.add_argument("--processed-base", help="Processed base path for daily NetCDF quicklooks")
    parser.add_argument("--radiance-base", help="Radiance base path for raw daily quicklooks")
    parser.add_argument("--cloud-base", help="Cloud-mask base path for raw daily quicklooks")
    parser.add_argument("--date", help="Date for daily plotting (YYYY-MM-DD or DD.MM.YYYY)")
    parser.add_argument("--channels", default="ir_105,wv_63", help="Comma-separated channels for raw plotting")
    parser.add_argument("--save-to-raw", action="store_true", help="Save raw quicklooks beside the radiance base")

    parser.add_argument("--local-base", help="Local base directory for S3 verification")
    parser.add_argument("--credentials", default="s3_credentials.py", help="Credentials file for S3 verification")
    parser.add_argument("--bucket-prefix", default="", help="Optional bucket prefix for S3 verification")

    parser.add_argument("--config", help="Optional config file passed to daily plotting scripts")
    return parser.parse_args()


def _build_command(script: Path, args: List[str]) -> List[str]:
    return [sys.executable, str(script), *args]


def _task_args(task: str, args: argparse.Namespace) -> List[str]:
    if task == "inspect":
        if not args.file:
            raise SystemExit("--file is required for the inspect task")
        cmd = ["--file", args.file]
        return cmd
    if task == "plot":
        if not args.file or not args.output_dir:
            raise SystemExit("--file and --output-dir are required for the plot task")
        return ["--file", args.file, "--output-dir", args.output_dir]
    if task == "plot-nc-day":
        cmd: List[str] = []
        if args.config:
            cmd += ["--config", args.config]
        else:
            if not args.processed_base or not args.date:
                raise SystemExit("--processed-base and --date are required for plot-nc-day without --config")
            cmd += ["--processed-base", args.processed_base, "--date", args.date]
            if args.output_dir:
                cmd += ["--output-dir", args.output_dir]
        return cmd
    if task == "plot-hdf-day":
        cmd = []
        if args.config:
            cmd += ["--config", args.config]
        else:
            if not args.radiance_base or not args.cloud_base or not args.date:
                raise SystemExit("--radiance-base, --cloud-base and --date are required for plot-hdf-day without --config")
            cmd += [
                "--radiance-base",
                args.radiance_base,
                "--cloud-base",
                args.cloud_base,
                "--date",
                args.date,
                "--channels",
                args.channels,
            ]
            if args.output_dir:
                cmd += ["--output-dir", args.output_dir]
            if args.save_to_raw:
                cmd += ["--save-to-raw"]
        return cmd
    if task == "verify-s3":
        if not args.local_base:
            raise SystemExit("--local-base is required for verify-s3")
        return ["--local-base", args.local_base, "--credentials", args.credentials, "--bucket-prefix", args.bucket_prefix]
    raise SystemExit(f"Unknown task: {task}")


def _expanded_tasks(groups: Iterable[str]) -> List[str]:
    tasks: List[str] = []
    for group in groups:
        for task in GROUPS[group]:
            if task not in tasks:
                tasks.append(task)
    return tasks


def main() -> None:
    args = parse_args()
    tasks = _expanded_tasks(args.group)

    commands = [_build_command(TASKS[task], _task_args(task, args)) for task in tasks]
    for command in commands:
        print(" ".join(command))

    if args.parallel:
        processes = [subprocess.Popen(command) for command in commands]
        exit_codes = [process.wait() for process in processes]
        failed = [code for code in exit_codes if code != 0]
        if failed:
            raise SystemExit(max(failed))
    else:
        for command in commands:
            completed = subprocess.run(command, check=False)
            if completed.returncode != 0:
                raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
