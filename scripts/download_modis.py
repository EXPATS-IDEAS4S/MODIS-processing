#!/usr/bin/env python3
"""Download MODIS Terra/Aqua L1B and cloud mask granules for a configured period/ROI."""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import requests
import yaml

LOGGER = logging.getLogger("download_modis")
CMR_GRANULES_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
PRODUCTS = {
    "terra": {"l1": "MOD021KM", "l2": "MOD35_L2"},
    "aqua": {"l1": "MYD021KM", "l2": "MYD35_L2"},
}

VALID_EXTENSIONS = {"hdf", "nc"}


@dataclass(frozen=True)
class DateRange:
    start: dt.datetime
    end: dt.datetime


def load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def iter_day_ranges(years: Iterable[int], months: Iterable[int]) -> Iterable[DateRange]:
    for year in years:
        for month in months:
            _, days_in_month = calendar.monthrange(year, month)
            for day in range(1, days_in_month + 1):
                start = dt.datetime(year, month, day, 0, 0, 0)
                end = start + dt.timedelta(days=1)
                yield DateRange(start=start, end=end)


def cmr_search(short_name: str, roi: Dict[str, float], date_range: DateRange, page_size: int = 2000) -> List[Dict]:
    params = {
        "short_name": short_name,
        "temporal": f"{date_range.start.isoformat()}Z,{date_range.end.isoformat()}Z",
        "bounding_box": f"{roi['lon_min']},{roi['lat_min']},{roi['lon_max']},{roi['lat_max']}",
        "page_size": page_size,
    }
    LOGGER.info("Searching CMR for %s between %s and %s", short_name, date_range.start, date_range.end)
    response = requests.get(CMR_GRANULES_URL, params=params, timeout=60)
    response.raise_for_status()
    return response.json().get("feed", {}).get("entry", [])


def pick_data_link(entry: Dict, extension: str = "hdf") -> str | None:
    ext = extension.lower().lstrip(".")
    if ext not in VALID_EXTENSIONS:
        raise ValueError(f"Unsupported extension '{extension}'. Supported: {sorted(VALID_EXTENSIONS)}")

    for link in entry.get("links", []):
        href = link.get("href")
        if not href:
            continue
        if link.get("inherited"):
            continue
        rel = link.get("rel", "")
        title = (link.get("title") or "").lower()
        if ("data#" in rel or "download" in title) and href.lower().endswith(f".{ext}"):
            return href
    return None


def parse_granule_time(entry: Dict) -> dt.datetime:
    stamp = entry.get("time_start") or entry.get("updated")
    return dt.datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def output_path(base_path: Path, granule_time: dt.datetime, filename: str) -> Path:
    day_dir = base_path / f"{granule_time.year:04d}" / f"{granule_time.month:02d}" / f"{granule_time.day:02d}"
    day_dir.mkdir(parents=True, exist_ok=True)
    return day_dir / filename


def download_file(url: str, token: str, destination: Path, overwrite: bool = False) -> None:
    if destination.exists() and not overwrite:
        LOGGER.debug("Skipping existing file %s", destination)
        return

    headers = {"Authorization": f"Bearer {token}"}
    with requests.get(url, headers=headers, timeout=120, stream=True) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def _download_product(
    short_name: str,
    base_path: Path,
    roi: Dict[str, float],
    date_range: DateRange,
    token: str,
    overwrite: bool,
    dry_run: bool,
    extension: str,
) -> Tuple[int, int]:
    page_size = 1 if dry_run else 2000
    entries = cmr_search(short_name=short_name, roi=roi, date_range=date_range, page_size=page_size)
    downloaded = 0
    skipped = 0

    for entry in entries:
        url = pick_data_link(entry, extension=extension)
        if not url:
            skipped += 1
            continue

        filename = Path(url.split("?")[0]).name
        granule_time = parse_granule_time(entry)
        destination = output_path(base_path=base_path, granule_time=granule_time, filename=filename)

        try:
            download_file(url=url, token=token, destination=destination, overwrite=overwrite)
            downloaded += 1
            if dry_run:
                LOGGER.info("[DRY-RUN] sampled %s -> %s", url, destination)
                break
        except requests.RequestException as exc:
            skipped += 1
            LOGGER.warning("Failed download %s (%s)", url, exc)

        if dry_run:
            break

    return downloaded, skipped


def run_download(config: Dict, dry_run: bool = False) -> None:
    token_env_var = config["auth"]["earthdata_token_env"]
    token = os.getenv(token_env_var)
    if not token:
        raise RuntimeError(f"Missing Earthdata token in env var: {token_env_var}")

    roi = config["roi"]
    years = config["years"]
    months = config["months"]
    overwrite = bool(config.get("overwrite", False))

    radiance_base = Path(config["download"]["radiance_base_path"])
    cloud_base = Path(config["download"]["cloud_mask_base_path"])
    l1_extension = config["download"].get("l1_extension", "hdf")
    l2_extension = config["download"].get("l2_extension", "hdf")

    date_ranges = iter_day_ranges(years=years, months=months)
    if dry_run:
        date_ranges = list(date_ranges)[:1]

    for date_range in date_ranges:
        for satellite, products in PRODUCTS.items():
            l1_product = products["l1"]
            l2_product = products["l2"]
            l1_downloaded, l1_skipped = _download_product(
                short_name=l1_product,
                base_path=radiance_base,
                roi=roi,
                date_range=date_range,
                token=token or "",
                overwrite=overwrite,
                dry_run=dry_run,
                extension=l1_extension,
            )
            l2_downloaded, l2_skipped = _download_product(
                short_name=l2_product,
                base_path=cloud_base,
                roi=roi,
                date_range=date_range,
                token=token or "",
                overwrite=overwrite,
                dry_run=dry_run,
                extension=l2_extension,
            )
            LOGGER.info(
                "%s [%s - %s] L1 downloaded=%s skipped=%s | L2 downloaded=%s skipped=%s",
                satellite,
                date_range.start.date(),
                date_range.end.date(),
                l1_downloaded,
                l1_skipped,
                l2_downloaded,
                l2_skipped,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download MODIS Terra/Aqua radiance + cloud mask data")
    parser.add_argument("--config", required=True, help="Path to YAML pipeline config")
    parser.add_argument("--dry-run", action="store_true", help="List files that would be downloaded")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    config = load_config(Path(args.config))
    run_download(config=config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
