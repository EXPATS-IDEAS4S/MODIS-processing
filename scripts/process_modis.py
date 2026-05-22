#!/usr/bin/env python3
"""Process downloaded MODIS files to BT and cloud-mask NetCDF outputs."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import xarray as xr
import yaml
from satpy import DataQuery, Scene

LOGGER = logging.getLogger("process_modis")
SATELLITE_PREFIX = {"terra": "MOD", "aqua": "MYD"}
CHANNEL_TO_BAND = {
    "ir_105": "31",
    "wv_63": "27",
}


def load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def iter_days(base_path: Path, years: Iterable[int], months: Iterable[int]) -> Iterable[Path]:
    for year in years:
        for month in months:
            month_dir = base_path / f"{year:04d}" / f"{month:02d}"
            if not month_dir.exists():
                continue
            for day_dir in sorted([item for item in month_dir.iterdir() if item.is_dir()]):
                yield day_dir


def _is_l1_file(path: Path, satellite: str) -> bool:
    prefix = SATELLITE_PREFIX[satellite]
    return path.name.startswith(f"{prefix}021KM") and path.suffix.lower() in {".hdf", ".h5"}


def _is_l2_file(path: Path, satellite: str) -> bool:
    prefix = SATELLITE_PREFIX[satellite]
    return path.name.startswith(f"{prefix}35_L2") and path.suffix.lower() in {".hdf", ".h5"}


def list_files(radiance_day_dir: Path, cloud_day_dir: Path, satellite: str) -> Tuple[List[Path], List[Path]]:
    l1_files = sorted([item for item in radiance_day_dir.glob("*") if item.is_file() and _is_l1_file(item, satellite)])
    l2_files = sorted([item for item in cloud_day_dir.glob("*") if item.is_file() and _is_l2_file(item, satellite)])
    return l1_files, l2_files


def _extract_granule_key(path: Path) -> str:
    parts = path.stem.split(".")
    return "_".join(parts[1:3]) if len(parts) >= 3 else path.stem


def _index_l2_files(l2_files: Iterable[Path]) -> Dict[str, Path]:
    return {_extract_granule_key(path): path for path in l2_files}


def _load_bt_datasets(l1_file: Path, channels: List[str]) -> Dict[str, xr.DataArray]:
    band_names = [CHANNEL_TO_BAND[channel] for channel in channels]
    queries = [DataQuery(name=band, calibration="brightness_temperature") for band in band_names]

    scene = Scene(reader="modis_l1b", filenames=[str(l1_file)])
    scene.load(queries)

    data = {}
    for channel, band, query in zip(channels, band_names, queries):
        bt = scene[query].astype(np.float32)
        data[f"bt_{channel}"] = bt.rename(f"bt_{channel}")
        data[f"bt_{channel}"].attrs.update(
            {
                "long_name": f"Brightness temperature channel {channel} (MODIS band {band})",
                "units": "K",
            }
        )
    return data


def _load_cloud_mask(l2_file: Path, target_area) -> Optional[xr.DataArray]:
    scene = Scene(reader="modis_l2", filenames=[str(l2_file)])
    available = scene.available_dataset_names()
    preferred_names = ["cloud_mask", "Cloud_Mask", "cloud_mask_byte_segment"]

    dataset_name = None
    for candidate in preferred_names:
        if candidate in available:
            dataset_name = candidate
            break
    if dataset_name is None and available:
        dataset_name = sorted(available)[0]

    if dataset_name is None:
        return None

    scene.load([dataset_name])
    cloud_data = scene[dataset_name]

    if target_area is not None and cloud_data.attrs.get("area") is not None:
        scene = scene.resample(target_area)
        cloud_data = scene[dataset_name]

    cloud_data = cloud_data.astype(np.uint8).rename("cloud_mask")
    cloud_data.attrs.update({"long_name": "MODIS cloud mask"})
    return cloud_data


def _parse_time_from_name(path: Path) -> dt.datetime:
    parts = path.stem.split(".")
    # Example: MOD021KM.A2024123.1050.061.2024123133333
    if len(parts) < 3:
        return dt.datetime.utcnow()
    julian = parts[1][1:]
    hhmm = parts[2]
    year = int(julian[:4])
    doy = int(julian[4:])
    hour = int(hhmm[:2])
    minute = int(hhmm[2:4])
    return dt.datetime(year, 1, 1, hour=hour, minute=minute) + dt.timedelta(days=doy - 1)


def process_day(
    radiance_day_dir: Path,
    cloud_day_dir: Path,
    output_base: Path,
    channels: List[str],
    satellite: str,
    overwrite: bool = False,
) -> int:
    l1_files, l2_files = list_files(radiance_day_dir=radiance_day_dir, cloud_day_dir=cloud_day_dir, satellite=satellite)
    if not l1_files:
        return 0

    l2_index = _index_l2_files(l2_files)
    count = 0

    for l1_file in l1_files:
        granule_key = _extract_granule_key(l1_file)
        l2_file = l2_index.get(granule_key)

        bt_data = _load_bt_datasets(l1_file=l1_file, channels=channels)
        first_bt = next(iter(bt_data.values()))
        target_area = first_bt.attrs.get("area")

        data_vars = dict(bt_data)
        if l2_file is not None:
            cloud_mask = _load_cloud_mask(l2_file=l2_file, target_area=target_area)
            if cloud_mask is not None:
                data_vars["cloud_mask"] = cloud_mask

        dataset = xr.Dataset(data_vars=data_vars)
        timestamp = _parse_time_from_name(l1_file)

        out_dir = output_base / f"{timestamp.year:04d}" / f"{timestamp.month:02d}" / f"{timestamp.day:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{satellite}_modis_bt_cloudmask_{timestamp:%Y%m%dT%H%M}.nc"
        if out_file.exists() and not overwrite:
            LOGGER.debug("Skipping existing processed file %s", out_file)
            continue

        dataset.attrs.update(
            {
                "satellite": satellite,
                "source_l1": l1_file.name,
                "source_l2": l2_file.name if l2_file else "",
                "channels": ",".join(channels),
            }
        )
        dataset.to_netcdf(out_file)
        count += 1
        LOGGER.info("Wrote %s", out_file)

    return count


def run_processing(config: Dict) -> None:
    channels = config["channels"]
    for channel in channels:
        if channel not in CHANNEL_TO_BAND:
            raise ValueError(f"Unsupported channel '{channel}'. Supported: {sorted(CHANNEL_TO_BAND)}")

    years = config["years"]
    months = config["months"]
    overwrite = bool(config.get("overwrite", False))

    radiance_base = Path(config["download"]["radiance_base_path"])
    cloud_base = Path(config["download"]["cloud_mask_base_path"])
    output_base = Path(config["processing"]["output_base_path"])

    for radiance_day_dir in iter_days(base_path=radiance_base, years=years, months=months):
        rel_day = radiance_day_dir.relative_to(radiance_base)
        cloud_day_dir = cloud_base / rel_day
        for satellite in ("terra", "aqua"):
            produced = process_day(
                radiance_day_dir=radiance_day_dir,
                cloud_day_dir=cloud_day_dir,
                output_base=output_base,
                channels=channels,
                satellite=satellite,
                overwrite=overwrite,
            )
            if produced:
                LOGGER.info("Processed %s files for %s in %s", produced, satellite, radiance_day_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process MODIS L1/L2 files to BT+cloud mask NetCDF")
    parser.add_argument("--config", required=True, help="Path to YAML pipeline config")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    config = load_config(Path(args.config))
    run_processing(config=config)


if __name__ == "__main__":
    main()
