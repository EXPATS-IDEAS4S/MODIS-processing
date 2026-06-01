#!/usr/bin/env python3
"""Plot all processed NetCDF quicklooks for a single day."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from scripts.debug_tools import plot_nc_day


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot processed NetCDF quicklooks for a day")
    parser.add_argument("--processed-base", required=False, help="Base path where processed day folders are stored")
    parser.add_argument("--date", required=False, help="Date to plot (YYYY-MM-DD or DD.MM.YYYY)")
    parser.add_argument("--output-dir", required=False, help="Output base dir for quicklooks")
    parser.add_argument("--config", required=False, help="YAML config file with parameters")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text())
        if isinstance(cfg, dict) and "processing" in cfg:
            processing_cfg = cfg.get("processing", {})
            quicklooks_cfg = processing_cfg.get("quicklooks", {})
            processed_base = processing_cfg.get("output_base_path")
            date = quicklooks_cfg.get("date") or args.date
            output_dir_cfg = quicklooks_cfg.get("base_path") or args.output_dir
        else:
            processed_base = args.processed_base
            date = args.date
            output_dir_cfg = args.output_dir
    else:
        processed_base = args.processed_base
        date = args.date
        output_dir_cfg = args.output_dir

    if processed_base is None or date is None:
        if processed_base is not None and (date is None or str(date).strip() == ""):
            base_path = Path(processed_base)
            candidates = sorted([p for p in base_path.glob("*/*/*") if p.is_dir()])
            if candidates:
                first = candidates[0]
                date = f"{first.parent.parent.name}-{first.parent.name}-{first.name}"
                print(f"Auto-detected date {date} from {first}")
            else:
                raise SystemExit("processed_base and date must be provided either via CLI or config file")
        else:
            raise SystemExit("processed_base and date must be provided either via CLI or config file")

    output_base = Path(output_dir_cfg) if output_dir_cfg else Path(processed_base) / "quicklooks"
    plot_nc_day(Path(processed_base), str(date), output_base)


if __name__ == "__main__":
    main()
