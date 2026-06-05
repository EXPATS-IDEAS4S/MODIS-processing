#!/usr/bin/env python3
"""Download MODIS Terra/Aqua L1B and cloud mask granules for a configured period/ROI.

The selected period can come from the YAML config (`years`, `months`, and
optional `days`) or from a single `--date YYYY-MM-DD` override.
"""

from __future__ import annotations

import argparse
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import requests
import yaml

from scripts.utils.io import iter_selected_dates

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


def iter_day_ranges(years: Iterable[int], months: Iterable[int], days: Iterable[int] | str | None = None) -> Iterable[DateRange]:
    for day in iter_selected_dates(years=years, months=months, days=days):
        start = dt.datetime(day.year, day.month, day.day, 0, 0, 0)
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


def _entry_footprint_bbox(entry: Dict) -> Tuple[float, float, float, float] | None:
    def _to_float_sequence(obj) -> List[float]:
        if isinstance(obj, str):
            return [float(value) for value in obj.replace(",", " ").split()]
        if isinstance(obj, (list, tuple)):
            values: List[float] = []
            for item in obj:
                values.extend(_to_float_sequence(item))
            return values
        return [float(obj)]

    bboxes: List[Tuple[float, float, float, float]] = []

    for polygon in entry.get("polygons", []) or []:
        points = _to_float_sequence(polygon)
        if len(points) < 6 or len(points) % 2 != 0:
            continue
        lons = points[0::2]
        lats = points[1::2]
        bboxes.append((min(lons), min(lats), max(lons), max(lats)))

    for box in entry.get("boxes", []) or []:
        values = _to_float_sequence(box)
        if len(values) != 4:
            continue
        lon_min, lat_min, lon_max, lat_max = values
        bboxes.append((lon_min, lat_min, lon_max, lat_max))

    if not bboxes:
        return None

    lon_min = min(item[0] for item in bboxes)
    lat_min = min(item[1] for item in bboxes)
    lon_max = max(item[2] for item in bboxes)
    lat_max = max(item[3] for item in bboxes)
    return lon_min, lat_min, lon_max, lat_max


def _bbox_overlap_fraction(footprint: Tuple[float, float, float, float], roi: Dict[str, float]) -> float:
    roi_lon_min = float(roi["lon_min"])
    roi_lat_min = float(roi["lat_min"])
    roi_lon_max = float(roi["lon_max"])
    roi_lat_max = float(roi["lat_max"])
    lon_min = max(footprint[0], roi_lon_min)
    lat_min = max(footprint[1], roi_lat_min)
    lon_max = min(footprint[2], roi_lon_max)
    lat_max = min(footprint[3], roi_lat_max)
    if lon_max <= lon_min or lat_max <= lat_min:
        return 0.0

    roi_area = max(roi_lon_max - roi_lon_min, 0.0) * max(roi_lat_max - roi_lat_min, 0.0)
    if roi_area <= 0:
        return 0.0

    inter_area = (lon_max - lon_min) * (lat_max - lat_min)
    return inter_area / roi_area


def _bbox_intersects(footprint: Tuple[float, float, float, float], roi: Dict[str, float]) -> bool:
    lon_min = max(float(footprint[0]), float(roi["lon_min"]))
    lat_min = max(float(footprint[1]), float(roi["lat_min"]))
    lon_max = min(float(footprint[2]), float(roi["lon_max"]))
    lat_max = min(float(footprint[3]), float(roi["lat_max"]))
    return lon_max > lon_min and lat_max > lat_min


def _inner_roi(roi: Dict[str, float], pad_deg: float) -> Dict[str, float] | None:
    pad = max(float(pad_deg), 0.0)
    lon_min = float(roi["lon_min"]) + pad
    lat_min = float(roi["lat_min"]) + pad
    lon_max = float(roi["lon_max"]) - pad
    lat_max = float(roi["lat_max"]) - pad
    if lon_max <= lon_min or lat_max <= lat_min:
        return None
    return {"lon_min": lon_min, "lat_min": lat_min, "lon_max": lon_max, "lat_max": lat_max}


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
    parallel: bool = False,
    workers: int = 4,
    min_overlap_fraction: float = 0.0,
    footprint_filter_mode: str = "overlap_fraction",
    subroi_pad_deg: float = 0.0,
) -> Tuple[int, int]:
    page_size = 1 if dry_run else 2000
    entries = cmr_search(short_name=short_name, roi=roi, date_range=date_range, page_size=page_size)
    candidates: List[Tuple[str, Path]] = []
    skipped = 0

    filter_mode = str(footprint_filter_mode or "overlap_fraction").strip().lower()
    if filter_mode not in {"overlap_fraction", "intersects_roi", "intersects_subroi"}:
        LOGGER.warning("Unknown footprint_filter_mode=%s for %s; falling back to overlap_fraction", filter_mode, short_name)
        filter_mode = "overlap_fraction"

    subroi = _inner_roi(roi, subroi_pad_deg)
    if filter_mode == "intersects_subroi" and subroi is None:
        LOGGER.warning(
            "subroi_pad_deg=%.3f collapses ROI for %s; falling back to intersects_roi",
            float(subroi_pad_deg),
            short_name,
        )
        filter_mode = "intersects_roi"

    for entry in entries:
        footprint_bbox = _entry_footprint_bbox(entry)
        if footprint_bbox is not None:
            if filter_mode == "overlap_fraction" and min_overlap_fraction > 0.0:
                overlap = _bbox_overlap_fraction(footprint_bbox, roi)
                if overlap < min_overlap_fraction:
                    skipped += 1
                    LOGGER.debug(
                        "Skipping granule outside footprint threshold for %s (%s overlap %.3f < %.3f)",
                        short_name,
                        entry.get("title") or entry.get("id") or "unknown",
                        overlap,
                        min_overlap_fraction,
                    )
                    continue
            elif filter_mode == "intersects_roi":
                if not _bbox_intersects(footprint_bbox, roi):
                    skipped += 1
                    continue
            elif filter_mode == "intersects_subroi":
                if subroi is not None and not _bbox_intersects(footprint_bbox, subroi):
                    skipped += 1
                    continue

        url = pick_data_link(entry, extension=extension)
        if not url:
            skipped += 1
            continue

        filename = Path(url.split("?")[0]).name
        granule_time = parse_granule_time(entry)
        destination = output_path(base_path=base_path, granule_time=granule_time, filename=filename)
        candidates.append((url, destination))

    if dry_run:
        if candidates:
            url, destination = candidates[0]
            LOGGER.info("[DRY-RUN] sampled %s -> %s", url, destination)
            return 1, skipped + max(0, len(candidates) - 1)
        return 0, skipped

    downloaded = 0

    def _download_candidate(item: Tuple[str, Path]) -> bool:
        url, destination = item
        try:
            download_file(url=url, token=token, destination=destination, overwrite=overwrite)
            return True
        except requests.RequestException as exc:
            LOGGER.warning("Failed download %s (%s)", url, exc)
            return False

    if parallel and len(candidates) > 1:
        max_workers = max(1, workers)
        LOGGER.info("Downloading %s granules with %s workers for %s", len(candidates), max_workers, short_name)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_download_candidate, item) for item in candidates]
            for future in as_completed(futures):
                if future.result():
                    downloaded += 1
    else:
        for item in candidates:
            if _download_candidate(item):
                downloaded += 1

    return downloaded, skipped


def run_download(config: Dict, dry_run: bool = False, date: dt.date | None = None) -> None:
    token_env_var = config["auth"]["earthdata_token_env"]
    token = os.getenv(token_env_var)
    if not token:
        raise RuntimeError(f"Missing Earthdata token in env var: {token_env_var}")

    roi = config["roi"]
    years = [date.year] if date is not None else config["years"]
    months = [date.month] if date is not None else config["months"]
    days = [date.day] if date is not None else config.get("days", "all")
    overwrite = bool(config.get("overwrite", False))
    download_cfg = config.get("download", {})
    parallel = bool(download_cfg.get("parallel", False))
    workers = int(download_cfg.get("workers", 4))
    min_overlap_fraction = float(download_cfg.get("min_overlap_fraction", 0.0))
    footprint_filter_mode = str(download_cfg.get("footprint_filter_mode", "overlap_fraction"))
    subroi_pad_deg = float(download_cfg.get("subroi_pad_deg", 0.0))

    radiance_base = Path(download_cfg["radiance_base_path"])
    cloud_base = Path(download_cfg["cloud_mask_base_path"])
    l1_extension = download_cfg.get("l1_extension", "hdf")
    l2_extension = download_cfg.get("l2_extension", "hdf")

    date_ranges = iter_day_ranges(years=years, months=months, days=days)
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
                parallel=parallel,
                workers=workers,
                min_overlap_fraction=min_overlap_fraction,
                footprint_filter_mode=footprint_filter_mode,
                subroi_pad_deg=subroi_pad_deg,
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
                parallel=parallel,
                workers=workers,
                min_overlap_fraction=min_overlap_fraction,
                footprint_filter_mode=footprint_filter_mode,
                subroi_pad_deg=subroi_pad_deg,
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
    parser.add_argument("--date", default=None, help="Process only one day in YYYY-MM-DD format")
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
    selected_date = dt.date.fromisoformat(args.date) if args.date else None
    run_download(config=config, dry_run=args.dry_run, date=selected_date)


if __name__ == "__main__":
    main()
