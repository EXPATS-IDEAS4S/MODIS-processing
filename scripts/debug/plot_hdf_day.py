#!/usr/bin/env python3
"""Plot raw MODIS L1/L2 granules for a single day."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from scripts.debug_tools import plot_hdf_day


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot raw MODIS granules for a day")
    parser.add_argument("--radiance-base", required=False, help="Base path where radiance day folders are stored")
    parser.add_argument("--cloud-base", required=False, help="Base path where cloud mask day folders are stored")
    parser.add_argument("--date", required=False, help="Date to plot (YYYY-MM-DD or DD.MM.YYYY)")
    parser.add_argument("--channels", default="ir_105,wv_63", help="Comma-separated channel keys")
    parser.add_argument("--output-dir", required=False, help="Output base dir for quicklooks")
    parser.add_argument("--save-to-raw", action="store_true", help="Save quicklooks under radiance_base.parent/quicklooks")
    parser.add_argument("--config", required=False, help="YAML config file with parameters")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text())
        if isinstance(cfg, dict) and "download" in cfg:
            radiance_base = cfg.get("download", {}).get("radiance_base_path") or cfg.get("download", {}).get("radiance_base")
            cloud_base = cfg.get("download", {}).get("cloud_mask_base_path") or cfg.get("download", {}).get("cloud_base")
            processing_cfg = cfg.get("processing", {})
            quicklooks_cfg = cfg.get("quicklooks") or processing_cfg.get("quicklooks", {})
            date = quicklooks_cfg.get("date") or cfg.get("date")
            channels = quicklooks_cfg.get("channels") or cfg.get("channels") or [c.strip() for c in args.channels.split(",") if c.strip()]
            save_to_raw = bool(quicklooks_cfg.get("save_to_raw", cfg.get("save_to_raw", False)))
            output_dir_cfg = quicklooks_cfg.get("base_path") or cfg.get("output_dir")
            l1_reader = str(processing_cfg.get("l1_reader", "modis_l1b"))
            l2_reader = str(processing_cfg.get("l2_reader", "modis_l2"))
            cloud_mask_channel = processing_cfg.get("cloud_mask_channel")
            cloud_mask_binary = bool(processing_cfg.get("cloud_mask_binary", True))
        else:
            radiance_base = cfg.get("radiance_base")
            cloud_base = cfg.get("cloud_base")
            date = cfg.get("date")
            channels = cfg.get("channels", []) or [c.strip() for c in args.channels.split(",") if c.strip()]
            save_to_raw = bool(cfg.get("save_to_raw", False))
            output_dir_cfg = cfg.get("output_dir")
            l1_reader = str(cfg.get("l1_reader", "modis_l1b"))
            l2_reader = str(cfg.get("l2_reader", "modis_l2"))
            cloud_mask_channel = cfg.get("cloud_mask_channel")
            cloud_mask_binary = bool(cfg.get("cloud_mask_binary", True))
    else:
        radiance_base = args.radiance_base
        cloud_base = args.cloud_base
        date = args.date
        channels = [c.strip() for c in args.channels.split(",") if c.strip()]
        save_to_raw = bool(args.save_to_raw)
        output_dir_cfg = args.output_dir
        l1_reader = "modis_l1b"
        l2_reader = "modis_l2"
        cloud_mask_channel = None
        cloud_mask_binary = True

    if radiance_base is None or cloud_base is None or date is None:
        if radiance_base is not None and (date is None or str(date).strip() == ""):
            base_path = Path(radiance_base)
            candidates = sorted([p for p in base_path.glob("*/*/*") if p.is_dir()])
            if candidates:
                first = candidates[0]
                date = f"{first.parent.parent.name}-{first.parent.name}-{first.name}"
                print(f"Auto-detected date {date} from {first}")
            else:
                raise SystemExit("radiance_base, cloud_base and date must be provided either via CLI or config file")
        else:
            raise SystemExit("radiance_base, cloud_base and date must be provided either via CLI or config file")

    if save_to_raw:
        output_base = Path(radiance_base).parent / "quicklooks"
    else:
        output_base = Path(output_dir_cfg) if output_dir_cfg else Path(radiance_base).parent / "quicklooks"

    plot_hdf_day(
        Path(radiance_base),
        Path(cloud_base),
        str(date),
        list(channels),
        output_base,
        l1_reader=l1_reader,
        l2_reader=l2_reader,
        cloud_mask_channel=cloud_mask_channel,
        cloud_mask_binary=cloud_mask_binary,
    )


if __name__ == "__main__":
    main()
