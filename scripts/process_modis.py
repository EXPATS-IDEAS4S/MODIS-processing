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


def _area_lonlats(data_array: xr.DataArray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        lon, lat = data_array.attrs["area"].get_lonlats()
        return np.asarray(lon), np.asarray(lat)
    except Exception:
        return None, None


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


def list_files(radiance_day_dir: Path, cloud_day_dir: Path, satellite: Optional[str]) -> Tuple[List[Path], List[Path]]:
    """List L1 and L2 files for a day. If `satellite` is None, include both terra and aqua."""
    if satellite is None:
        l1_files = sorted([item for item in radiance_day_dir.glob("*") if item.is_file() and any(_is_l1_file(item, s) for s in SATELLITE_PREFIX)])
        l2_files = sorted([item for item in cloud_day_dir.glob("*") if item.is_file() and any(_is_l2_file(item, s) for s in SATELLITE_PREFIX)])
    else:
        l1_files = sorted([item for item in radiance_day_dir.glob("*") if item.is_file() and _is_l1_file(item, satellite)])
        l2_files = sorted([item for item in cloud_day_dir.glob("*") if item.is_file() and _is_l2_file(item, satellite)])
    return l1_files, l2_files


def _extract_granule_key(path: Path) -> str:
    parts = path.stem.split(".")
    return "_".join(parts[1:3]) if len(parts) >= 3 else path.stem


def _index_l2_files(l2_files: Iterable[Path]) -> Dict[str, Path]:
    return {_extract_granule_key(path): path for path in l2_files}


def _satellite_from_name(path: Path) -> Optional[str]:
    name = path.name
    for sat, prefix in SATELLITE_PREFIX.items():
        if name.startswith(prefix):
            return sat
    return None


def _resolve_cloud_mask_name(available: Iterable[str], preferred_name: Optional[str] = None) -> Optional[str]:
    available_list = list(available)
    preferred_names = []
    if preferred_name:
        preferred_names.append(preferred_name)
        preferred_names.append(preferred_name.lower())
        preferred_names.append(preferred_name.upper())
    preferred_names.extend(["cloud_mask", "Cloud_Mask", "cloud_mask_byte_segment", "Integer_Cloud_Mask"])

    for candidate in preferred_names:
        if candidate in available_list:
            return candidate

    return next((name for name in available_list if "cloud" in name.lower()), None)


def _cloud_mask_to_binary_values(values: np.ndarray, binary: bool) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if not binary:
        return arr
    mapped = np.full(arr.shape, np.nan, dtype=np.float32)
    mapped[np.isin(arr, [0])] = 1.0
    mapped[np.isin(arr, [1, 2, 3])] = 0.0
    mapped[arr == -1] = np.nan
    return mapped


def _load_bt_datasets(l1_file: Path, channels: List[str], reader: str = "modis_l1b", area_def=None) -> Dict[str, xr.DataArray]:
    band_names = [CHANNEL_TO_BAND[channel] for channel in channels]
    queries = [DataQuery(name=band, calibration="brightness_temperature") for band in band_names]
    scene = Scene(reader=reader, filenames=[str(l1_file)])
    try:
        scene.load(queries)
    except Exception:
        scene.load(list(queries))

    data: Dict[str, xr.DataArray] = {}
    # retrieve BT arrays
    for channel, band, query in zip(channels, band_names, queries):
        try:
            bt = scene[query].astype(np.float32)
        except Exception:
            # try by band name as string
            try:
                bt = scene[band].astype(np.float32)
            except Exception:
                raise
        data[f"bt_{channel}"] = bt.rename(f"bt_{channel}")
        data[f"bt_{channel}"].attrs.update({"long_name": f"Brightness temperature channel {channel} (MODIS band {band})", "units": "K"})

    # retrieve geolocation directly from the loaded scene data area metadata
    geo_source = data.get(f"bt_{channels[0]}") if channels else None
    if geo_source is not None:
        lon, lat = _area_lonlats(geo_source)
        if lat is not None and lon is not None:
            data["latitude"] = xr.DataArray(lat, dims=geo_source.dims, coords=geo_source.coords).astype(np.float32)
            data["longitude"] = xr.DataArray(lon, dims=geo_source.dims, coords=geo_source.coords).astype(np.float32)

    return data


def _load_cloud_mask(
    l2_file: Path,
    reader: str = "modis_l2",
    cloud_mask_channel: Optional[str] = None,
    cloud_mask_binary: bool = True,
    area_def=None,
) -> Optional[xr.DataArray]:
    scene = Scene(reader=reader, filenames=[str(l2_file)])
    try:
        available = list(scene.available_dataset_names())
    except Exception:
        available = []

    dataset_name = _resolve_cloud_mask_name(available, preferred_name=cloud_mask_channel)

    if dataset_name is None:
        return None

    try:
        # MODIS cloud mask in this pipeline should be taken at 1000 m resolution.
        load_list = [DataQuery(name=dataset_name, resolution=1000)]
        scene.load(load_list)
        cloud_data = scene[DataQuery(name=dataset_name, resolution=1000)]
    except Exception:
        try:
            scene.load([DataQuery(name=dataset_name, resolution=1000)])
            cloud_data = scene[DataQuery(name=dataset_name, resolution=1000)]
        except Exception:
            return None

    cloud_lon, cloud_lat = _area_lonlats(cloud_data)
    #cloud_data_latlon = (cloud_lat, cloud_lon) if cloud_lat is not None and cloud_lon is not None else (None, None)

    cloud_values = _cloud_mask_to_binary_values(cloud_data.values, cloud_mask_binary)
    cloud_data = xr.DataArray(cloud_values, dims=cloud_data.dims, coords=cloud_data.coords, attrs=dict(cloud_data.attrs)).rename("cloud_mask")
    cloud_data.attrs.update({"long_name": "MODIS cloud mask"})
    # attach lat/lon pair as attribute for caller
    #cloud_data.attrs["_latlon_pair"] = cloud_data_latlon
    return cloud_data, cloud_lon, cloud_lat


def _load_l2_with_geolocation(
    l2_file: Path,
    reader: str = "modis_l2",
    cloud_mask_channel: Optional[str] = None,
    cloud_mask_binary: bool = True,
) -> Tuple[Optional[xr.DataArray], Optional[xr.DataArray], Optional[xr.DataArray]]:
    """Load L2 cloud mask and return (cloud_mask, latitude, longitude) DataArrays when available.

    Returns a tuple of (cloud_mask, lat, lon) where missing items are None.
    """
    scene = Scene(reader=reader, filenames=[str(l2_file)])
    try:
        available = list(scene.available_dataset_names())
    except Exception:
        available = []

    # pick cloud mask dataset name
    cloud_name = _resolve_cloud_mask_name(available, preferred_name=cloud_mask_channel)
    if cloud_name is None:
        return None, None, None

    load_list = [DataQuery(name=cloud_name, resolution=1000)]

    try:
        scene.load(load_list)
    except Exception:
        scene.load([DataQuery(name=cloud_name)])

    try:
        cloud = scene[DataQuery(name=cloud_name, resolution=1000)]
    except Exception:
        try:
            cloud = scene[DataQuery(name=cloud_name, resolution=1000)]
        except Exception:
            cloud = None

    if cloud is not None:
        cloud = xr.DataArray(
            _cloud_mask_to_binary_values(cloud.values, cloud_mask_binary),
            dims=cloud.dims,
            coords=cloud.coords,
            attrs=dict(cloud.attrs),
        ).rename("cloud_mask")

    lon_values, lat_values = _area_lonlats(cloud)
    lat = xr.DataArray(lat_values, dims=cloud.dims, coords=cloud.coords) if lat_values is not None else None
    lon = xr.DataArray(lon_values, dims=cloud.dims, coords=cloud.coords) if lon_values is not None else None

    return cloud, lat, lon


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


def _regrid_to_target(
    src_da: xr.DataArray,
    src_lon: np.ndarray,
    src_lat: np.ndarray,
    target_lon: np.ndarray,
    target_lat: np.ndarray,
    method: str = "linear",
    chunk_rows: Optional[int] = None,
) -> xr.DataArray:
    """Regrid a 2D DataArray given source lon/lat and target lon/lat (2D).

    Uses scipy.interpolate.griddata. Returns DataArray with dims ('y','x')
    and coords 'latitude' and 'longitude'.
    """
    try:
        from scipy.interpolate import griddata
    except Exception as e:
        raise RuntimeError("scipy is required for 2D geolocation regridding: install scipy")

    # ensure numpy arrays
    src_lon_a = np.asarray(src_lon)
    src_lat_a = np.asarray(src_lat)
    vals = src_da.values

    # flatten source points
    points = np.column_stack((src_lon_a.ravel(), src_lat_a.ravel()))
    values = vals.ravel()

    # target grids
    tgt_lon = np.asarray(target_lon)
    tgt_lat = np.asarray(target_lat)

    # If chunk_rows is provided and target is 2D, process the target in row-chunks
    if chunk_rows and tgt_lat.ndim == 2:
        rows = tgt_lat.shape[0]
        pieces = []
        for start in range(0, rows, chunk_rows):
            stop = min(start + chunk_rows, rows)
            sub_lon = tgt_lon[start:stop, :]
            sub_lat = tgt_lat[start:stop, :]
            grid_z_sub = griddata(points, values, (sub_lon, sub_lat), method=method, fill_value=np.nan)
            pieces.append(grid_z_sub)
        grid_z = np.vstack(pieces)
    else:
        # griddata expects (nx, ny) matching meshgrid order; provide (lon, lat)
        grid_z = griddata(points, values, (tgt_lon, tgt_lat), method=method, fill_value=np.nan)

    da = xr.DataArray(grid_z.astype(values.dtype), dims=("y", "x"))
    da = da.assign_coords({"latitude": (("y", "x"), tgt_lat), "longitude": (("y", "x"), tgt_lon)})
    return da


def _resample_with_outside_nan(
    src_da: xr.DataArray,
    src_lon: np.ndarray,
    src_lat: np.ndarray,
    target_lon: np.ndarray,
    target_lat: np.ndarray,
    method: str = "linear",
    chunk_rows: Optional[int] = None,
) -> xr.DataArray:
    regridded = _regrid_to_target(src_da, src_lon, src_lat, target_lon, target_lat, method=method, chunk_rows=chunk_rows)
    if method == "nearest":
        coverage = _regrid_to_target(
            xr.DataArray(np.ones_like(np.asarray(src_da.values), dtype=np.float32), dims=src_da.dims),
            src_lon,
            src_lat,
            target_lon,
            target_lat,
            method="linear",
            chunk_rows=chunk_rows,
        )
        regridded = regridded.where(np.isfinite(coverage))
    return regridded


def _coverage_extent_regular(valid_mask: np.ndarray, lon_1d: np.ndarray, lat_1d: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.where(valid_mask)
    if ys.size == 0 or xs.size == 0:
        return 0.0, 0.0
    lon_vals = np.asarray(lon_1d)
    lat_vals = np.asarray(lat_1d)
    lon_res = abs(float(lon_vals[1] - lon_vals[0])) if lon_vals.size > 1 else 0.0
    lat_res = abs(float(lat_vals[1] - lat_vals[0])) if lat_vals.size > 1 else 0.0
    lon_extent = float(np.nanmax(lon_vals[xs]) - np.nanmin(lon_vals[xs]) + lon_res)
    lat_extent = float(np.nanmax(lat_vals[ys]) - np.nanmin(lat_vals[ys]) + lat_res)
    return lon_extent, lat_extent


def _coverage_extent_from_points(lon: np.ndarray, lat: np.ndarray, roi: Dict[str, float]) -> Tuple[float, float]:
    lon_arr = np.asarray(lon)
    lat_arr = np.asarray(lat)
    valid = np.isfinite(lon_arr) & np.isfinite(lat_arr)
    if roi:
        valid &= (
            (lon_arr >= float(roi.get("lon_min", -180.0)))
            & (lon_arr <= float(roi.get("lon_max", 180.0)))
            & (lat_arr >= float(roi.get("lat_min", -90.0)))
            & (lat_arr <= float(roi.get("lat_max", 90.0)))
        )
    if not np.any(valid):
        return 0.0, 0.0
    lon_vals = lon_arr[valid]
    lat_vals = lat_arr[valid]
    return float(np.nanmax(lon_vals) - np.nanmin(lon_vals)), float(np.nanmax(lat_vals) - np.nanmin(lat_vals))


def _largest_rectangle_from_mask(valid_mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return the largest all-True rectangle as (row_start, row_stop, col_start, col_stop)."""
    if valid_mask.ndim != 2 or not np.any(valid_mask):
        return None

    heights = np.zeros(valid_mask.shape[1], dtype=np.int64)
    best_area = 0
    best_bounds: Optional[Tuple[int, int, int, int]] = None

    for row_idx in range(valid_mask.shape[0]):
        row = valid_mask[row_idx]
        heights = np.where(row, heights + 1, 0)

        stack: List[int] = []
        extended = np.append(heights, 0)
        for col_idx, height in enumerate(extended):
            while stack and extended[stack[-1]] > height:
                top = stack.pop()
                rect_height = int(extended[top])
                rect_left = stack[-1] + 1 if stack else 0
                rect_right = col_idx
                rect_width = rect_right - rect_left
                area = rect_height * rect_width
                if area > best_area:
                    best_area = area
                    best_bounds = (row_idx - rect_height + 1, row_idx + 1, rect_left, rect_right)
            stack.append(col_idx)

    return best_bounds


def _crop_regular_dataset_to_valid_rectangle(dataset: xr.Dataset) -> xr.Dataset:
    if "latitude" not in dataset.dims or "longitude" not in dataset.dims:
        return dataset

    valid_mask = None
    for name in dataset.data_vars:
        arr = np.asarray(dataset[name].values)
        finite = np.isfinite(arr)
        finite = np.squeeze(finite)
        if finite.ndim != 2:
            continue
        valid_mask = finite if valid_mask is None else (valid_mask & finite)

    if valid_mask is None or not np.any(valid_mask):
        return dataset

    bounds = _largest_rectangle_from_mask(valid_mask)
    if bounds is None:
        return dataset

    row_start, row_stop, col_start, col_stop = bounds

    cropped = dataset.isel(latitude=slice(row_start, row_stop), longitude=slice(col_start, col_stop))
    LOGGER.info(
        "Cropped regular-grid dataset to largest valid rectangle: latitude[%d:%d], longitude[%d:%d]",
        row_start,
        row_stop,
        col_start,
        col_stop,
    )
    # Verify cropped area contains no NaNs across all 2D variables. If any remain
    # (unexpected), fall back to iterative edge-trimming until the rectangle is clean.
    try:
        def _is_clean(ds: xr.Dataset) -> bool:
            for name in ds.data_vars:
                arr = np.asarray(ds[name].values)
                arr = np.squeeze(arr)
                if arr.ndim != 2:
                    continue
                if not np.all(np.isfinite(arr)):
                    return False
            return True

        if not _is_clean(cropped):
            LOGGER.warning("Cropped rectangle still contains NaNs; trimming edges as fallback")
            lat_len = cropped.dims.get("latitude", 0)
            lon_len = cropped.dims.get("longitude", 0)
            r0, r1 = 0, lat_len
            c0, c1 = 0, lon_len
            changed = True
            while changed and r0 < r1 and c0 < c1:
                changed = False
                # build combined finite mask for current window
                combined = None
                for name in cropped.data_vars:
                    arr = np.asarray(cropped[name].values)
                    arr = np.squeeze(arr)
                    if arr.ndim != 2:
                        continue
                    finite = np.isfinite(arr)
                    combined = finite if combined is None else (combined & finite)
                if combined is None:
                    break
                # trim top
                if not np.all(combined[0, :]):
                    r0 += 1
                    combined = combined[1:, :]
                    changed = True
                # trim bottom
                if r0 < r1 and not np.all(combined[-1, :]):
                    r1 -= 1
                    combined = combined[:-1, :]
                    changed = True
                # trim left
                if c0 < c1 and not np.all(combined[:, 0]):
                    c0 += 1
                    combined = combined[:, 1:]
                    changed = True
                # trim right
                if c0 < c1 and not np.all(combined[:, -1]):
                    c1 -= 1
                    combined = combined[:, :-1]
                    changed = True
                # update cropped
                if changed and r0 < r1 and c0 < c1:
                    cropped = cropped.isel(latitude=slice(r0, r1), longitude=slice(c0, c1))

        # final check
        if not _is_clean(cropped):
            LOGGER.error("Unable to produce a NaN-free rectangle after trimming; returning original crop")
        else:
            LOGGER.info("Cropped rectangle verified NaN-free")
    except Exception:
        LOGGER.exception("Exception during cropped-rectangle verification")

    return cropped


def process_day(
    radiance_day_dir: Path,
    cloud_day_dir: Path,
    output_base: Path,
    channels: List[str],
    satellite: Optional[str],
    roi: Dict[str, float],
    l1_reader: str = "modis_l1b",
    l2_reader: str = "modis_l2",
    cloud_mask_channel: Optional[str] = None,
    cloud_mask_binary: bool = True,
    resolution_deg: float = 0.01,
    resample: bool = True,
    dry_run: bool = False,
    overwrite: bool = False,
    resample_chunk_rows: Optional[int] = None,
    coord_decimals: Optional[int] = None,
) -> int:
    l1_files, l2_files = list_files(radiance_day_dir=radiance_day_dir, cloud_day_dir=cloud_day_dir, satellite=satellite)
    # report per-satellite counts and total files in the directories for clarity
    try:
        total_radiance = sum(1 for _ in radiance_day_dir.iterdir() if _.is_file())
    except Exception:
        total_radiance = -1
    try:
        total_cloud = sum(1 for _ in cloud_day_dir.iterdir() if _.is_file())
    except Exception:
        total_cloud = -1
    sat_label = satellite if satellite is not None else "combined"
    LOGGER.info("Day %s %s: found %d L1 files and %d L2 files (dir totals: %d radiance, %d cloud)", sat_label, radiance_day_dir, len(l1_files), len(l2_files), total_radiance, total_cloud)
    if not l1_files:
        return 0

    l2_index = _index_l2_files(l2_files)
    count = 0

    # decide target grid: regular lat/lon (if resample True) or L1 native grid
    lon_min = roi.get("lon_min")
    lon_max = roi.get("lon_max")
    lat_min = roi.get("lat_min")
    lat_max = roi.get("lat_max")
    res = resolution_deg
    if resample:
        regular_lons = np.arange(lon_min, lon_max + res, res)
        regular_lats = np.arange(lat_max, lat_min - res, -res)
        tgt_lon2d, tgt_lat2d = np.meshgrid(regular_lons, regular_lats)
        target_is_regular = True
    else:
        regular_lons = None
        regular_lats = None
        tgt_lon2d = None
        tgt_lat2d = None
        target_is_regular = False

    for l1_file in l1_files:
        granule_key = _extract_granule_key(l1_file)
        l2_file = l2_index.get(granule_key)
        LOGGER.debug("L1 %s -> granule %s -> matched L2 %s", l1_file.name, granule_key, l2_file.name if l2_file is not None else None)
        try:
            ts = _parse_time_from_name(l1_file)
            LOGGER.info("Processing granule %s timestamp=%s matched_L2=%s", l1_file.name, ts.strftime("%Y%m%dT%H%M"), l2_file.name if l2_file is not None else "none")
        except Exception:
            LOGGER.info("Processing granule %s timestamp=unknown matched_L2=%s", l1_file.name, l2_file.name if l2_file is not None else "none")
        # If there's no matching L2 cloud-mask file, skip this L1 granule
        if l2_file is None:
            LOGGER.warning("Skipping %s: no matching L2 cloud file", l1_file.name)
            continue
        try:
            LOGGER.info("Step: loading BT datasets for %s", l1_file.name)
            bt_data = _load_bt_datasets(l1_file=l1_file, channels=channels, reader=l1_reader)
            LOGGER.info("Step: loaded BT datasets for %s: %s", l1_file.name, ",".join(k for k in bt_data.keys() if k.startswith("bt_")))
        except Exception as e:
            LOGGER.exception("Failed to load BT datasets for %s: %s", l1_file.name, e)
            continue

        # build a reference L1 grid from the first BT channel
        reference_key = f"bt_{channels[0]}"
        reference_da = bt_data.get(reference_key)
        if reference_da is None:
            LOGGER.warning("Skipping %s: missing reference BT channel %s", l1_file.name, reference_key)
            continue
        ref_lon, ref_lat = _area_lonlats(reference_da)
        if ref_lon is None or ref_lat is None:
            LOGGER.warning("Skipping %s: missing geolocation for reference channel %s", l1_file.name, reference_key)
            continue

        if target_is_regular:
            source_lon_extent, source_lat_extent = _coverage_extent_from_points(ref_lon, ref_lat, roi)
            LOGGER.info(
                "Native swath extent inside ROI for %s -> lon: %.2f deg, lat: %.2f deg",
                l1_file.name,
                source_lon_extent,
                source_lat_extent,
            )
            if source_lon_extent < 4.0 or source_lat_extent < 4.0:
                LOGGER.info(
                    "Skipping %s before resampling: native swath inside ROI is too small (lon %.2f deg, lat %.2f deg)",
                    l1_file.name,
                    source_lon_extent,
                    source_lat_extent,
                )
                continue

        if not target_is_regular:
            tgt_lon2d = ref_lon
            tgt_lat2d = ref_lat

        # resample or preserve each BT channel against the target grid
        data_vars_final: Dict[str, xr.DataArray] = {}
        for channel in channels:
            key = f"bt_{channel}"
            src_da = bt_data.get(key)
            if src_da is None:
                continue
            src_lon, src_lat = _area_lonlats(src_da)
            if src_lon is None or src_lat is None:
                LOGGER.warning("Dropping %s for %s: missing source geolocation", key, l1_file.name)
                continue

            if target_is_regular:
                LOGGER.info("Step: resampling BT %s to regular grid for %s", channel, l1_file.name)
                try:
                    re_da = _regrid_to_target(src_da.astype(np.float32), src_lon, src_lat, tgt_lon2d, tgt_lat2d, method="linear", chunk_rows=resample_chunk_rows)
                    data_vars_final[key] = re_da.rename({"y": "latitude", "x": "longitude"}).reset_coords(drop=True).rename(key).astype(np.float32)
                    LOGGER.debug("Resampled %s -> shape %s", key, data_vars_final[key].shape)
                except Exception:
                    LOGGER.warning("Failed regridding BT %s for %s", channel, l1_file.name)
            else:
                if src_da.shape == ref_lat.shape and src_da.shape == ref_lon.shape:
                    LOGGER.info("Step: keeping native L1 grid for %s", channel)
                    data_vars_final[key] = src_da.astype(np.float32).rename(key)
                else:
                    LOGGER.info("Step: regridding BT %s onto L1 grid for %s", channel, l1_file.name)
                    try:
                        re_da = _regrid_to_target(src_da.astype(np.float32), src_lon, src_lat, tgt_lon2d, tgt_lat2d, method="linear", chunk_rows=resample_chunk_rows)
                        data_vars_final[key] = re_da.rename(key).astype(np.float32)
                        LOGGER.debug("Regridded %s -> shape %s", key, data_vars_final[key].shape)
                    except Exception:
                        LOGGER.warning("Failed regridding BT %s for %s onto L1 grid", channel, l1_file.name)

        # cloud mask: load L2 and place it on the same target grid
        cloud_da = None
        if l2_file is not None:
            cloud_src, cloud_lon, cloud_lat = _load_cloud_mask(
                l2_file=l2_file,
                reader=l2_reader,
                cloud_mask_channel=cloud_mask_channel,
                cloud_mask_binary=cloud_mask_binary,
            )

            if cloud_src is not None and cloud_lon is not None and cloud_lat is not None:
                LOGGER.info("Step: loading/resampling cloud mask from %s", l2_file.name)
                if target_is_regular:
                    try:
                        cloud_da = _resample_with_outside_nan(
                            cloud_src.astype(np.float32),
                            cloud_lon,
                            cloud_lat,
                            tgt_lon2d,
                            tgt_lat2d,
                            method="nearest",
                            chunk_rows=resample_chunk_rows,
                        ).rename({"y": "latitude", "x": "longitude"}).reset_coords(drop=True).rename("cloud_mask").astype(np.float32)
                        LOGGER.debug("Cloud mask resampled -> shape %s", cloud_da.shape)
                    except Exception:
                        LOGGER.warning("Failed regridding cloud mask for %s", l2_file.name)
                else:
                    if cloud_src.shape == ref_lat.shape and cloud_src.shape == ref_lon.shape:
                        LOGGER.info("Step: keeping native L1 grid for cloud mask")
                        cloud_da = cloud_src.astype(np.float32).rename("cloud_mask")
                    else:
                        try:
                            cloud_da = _resample_with_outside_nan(
                                        cloud_src.astype(np.float32),
                                        cloud_lon,
                                        cloud_lat,
                                        tgt_lon2d,
                                        tgt_lat2d,
                                        method="nearest",
                                        chunk_rows=resample_chunk_rows,
                                    ).rename("cloud_mask").astype(np.float32)
                            LOGGER.debug("Cloud mask regridded onto L1 grid -> shape %s", cloud_da.shape)
                        except Exception:
                            LOGGER.warning("Failed regridding cloud mask for %s onto L1 grid", l2_file.name)

        if cloud_da is not None:
            data_vars_final["cloud_mask"] = cloud_da

        if not data_vars_final:
            LOGGER.warning("Skipping %s: no variables available after loading/resampling", l1_file.name)
            continue

        # reject very small swaths in regular-grid mode: need at least a 4x4 degree footprint
        if target_is_regular:
            valid_mask = None
            for v in data_vars_final.values():
                arr = np.asarray(v.values)
                finite = np.isfinite(arr)
                if valid_mask is None:
                    valid_mask = finite.copy()
                else:
                    valid_mask |= finite
            if valid_mask is None or not np.any(valid_mask):
                LOGGER.warning("Skipping %s: no valid values after resampling", l1_file.name)
                continue
            lon_extent, lat_extent = _coverage_extent_regular(valid_mask, regular_lons, regular_lats)
            LOGGER.info("Resampled coverage extent for %s -> lon: %.2f deg, lat: %.2f deg", l1_file.name, lon_extent, lat_extent)
            if lon_extent < 4.0 or lat_extent < 4.0:
                LOGGER.info(
                    "Skipping %s: resampled footprint too small (lon %.2f deg, lat %.2f deg)",
                    l1_file.name,
                    lon_extent,
                    lat_extent,
                )
                continue

        timestamp = _parse_time_from_name(l1_file)
        out_dir = output_base / f"{timestamp.year:04d}" / f"{timestamp.month:02d}" / f"{timestamp.day:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_sat = satellite if satellite is not None else (_satellite_from_name(l1_file) or "unknown")
        var_tag = "_".join(data_vars_final.keys())
        out_file = out_dir / f"{timestamp:%Y%m%dT%H%M}_modis_{out_sat}_{var_tag}.nc"
        if out_file.exists() and not overwrite:
            LOGGER.debug("Skipping existing processed file %s", out_file)
            continue

        coords: Dict[str, Tuple[Tuple[str, ...], np.ndarray] | np.ndarray] = {}
        if target_is_regular:
            coords["latitude"] = ("latitude", np.asarray(regular_lats, dtype=np.float32))
            coords["longitude"] = ("longitude", np.asarray(regular_lons, dtype=np.float32))
            data_vars_final = {
                name: da.rename({"y": "latitude", "x": "longitude"}).reset_coords(drop=True) if set(da.dims) == {"y", "x"} else da
                for name, da in data_vars_final.items()
            }
        else:
            coords["latitude"] = (("y", "x"), np.asarray(tgt_lat2d, dtype=np.float32))
            coords["longitude"] = (("y", "x"), np.asarray(tgt_lon2d, dtype=np.float32))

        dataset = xr.Dataset(data_vars=data_vars_final, coords=coords).expand_dims(time=[np.datetime64(timestamp)])
        if target_is_regular:
            # Ensure latitude/longitude are exact 1D regular axes (remove small numerical noise
            # introduced during regridding). Drop any existing 2D coord variables named
            # 'latitude'/'longitude' and assign rounded 1D arrays so downstream tools
            # can treat the grid as separable. `coord_decimals` may be provided by config
            try:
                # remove any 2D coord variables
                drop_coords = [n for n in list(dataset.coords) if n in ("latitude", "longitude") and getattr(dataset.coords[n], "ndim", 1) > 1]
                if drop_coords:
                    dataset = dataset.drop_vars(drop_coords)

                # build exact 1D axes rounded to stable values
                lon_axis = np.asarray(regular_lons, dtype=np.float64)
                lat_axis = np.asarray(regular_lats, dtype=np.float64)
                # compute decimals if not provided
                if coord_decimals is None:
                    if resolution_deg > 0:
                        coord_decimals = max(0, int(-np.floor(np.log10(resolution_deg))) + 2)
                    else:
                        coord_decimals = 6
                lon_axis = np.round(lon_axis, decimals=coord_decimals).astype(np.float32)
                lat_axis = np.round(lat_axis, decimals=coord_decimals).astype(np.float32)

                dataset = dataset.assign_coords({"latitude": ("latitude", lat_axis), "longitude": ("longitude", lon_axis)})
            except Exception:
                pass

            dataset = _crop_regular_dataset_to_valid_rectangle(dataset)
        LOGGER.info("Step: prepared Dataset for %s with vars=%s coords=%s", l1_file.name, ",".join(dataset.data_vars), ",".join(dataset.coords))
        timestamp = _parse_time_from_name(l1_file)
        start_time = timestamp.isoformat()
        end_time = (timestamp + dt.timedelta(minutes=5)).isoformat()

        dataset.attrs.update(
            {
                "satellite": out_sat,
                "source_l1": l1_file.name,
                "source_l2": l2_file.name if l2_file else "",
                "channels": ",".join(channels),
                "start_time": start_time,
                "end_time": end_time,
            }
        )
        if resample:
            dataset.attrs["grid_type"] = "regular_latlon"
            dataset.attrs["target_resolution_deg"] = float(resolution_deg)
        else:
            dataset.attrs["grid_type"] = "l1_native"
        dataset.attrs.pop("channels", None)
        dataset.attrs.pop("roi", None)
        # Robustly sanitize all attributes: convert datetimes and coerce anything
        # else to strings so NetCDF serialization will not fail.
        def _safe_attr(val):
            if isinstance(val, dt.datetime):
                return val.isoformat()
            if isinstance(val, (bool, np.bool_)):
                return int(val)
            simple_types = (str, bytes, int, float)
            if isinstance(val, simple_types):
                return val
            if isinstance(val, (list, tuple)) and all(isinstance(i, simple_types) for i in val):
                return val
            try:
                # try numpy array -> list
                if isinstance(val, np.ndarray):
                    return val.tolist()
            except Exception:
                pass
            # fallback: coerce to string
            try:
                return str(val)
            except Exception:
                return ""

        for k in list(dataset.attrs.keys()):
            dataset.attrs[k] = _safe_attr(dataset.attrs[k])

        for name in list(dataset.data_vars) + list(dataset.coords):
            obj = dataset[name]
            for k in list(obj.attrs.keys()):
                obj.attrs[k] = _safe_attr(obj.attrs[k])

        # Prepare NetCDF encodings with compression
        enc: Dict[str, Dict] = {}
        for name in dataset.data_vars:
            enc[name] = {"zlib": True, "complevel": 9, "dtype": "float32"}
        if "cloud_mask" in dataset.data_vars:
            enc["cloud_mask"] = {"zlib": True, "complevel": 9, "dtype": "float32"}
        # coords
        if "latitude" in dataset.coords:
            enc["latitude"] = {"zlib": True, "complevel": 9, "dtype": "float32"}
        if "longitude" in dataset.coords:
            enc["longitude"] = {"zlib": True, "complevel": 9, "dtype": "float32"}

        if dry_run:
            LOGGER.info("Dry-run: would write %s (vars: %s)", out_file, ",".join(dataset.data_vars))
            count += 1
        else:
            dataset.to_netcdf(out_file, encoding=enc)
            count += 1
            LOGGER.info("Wrote %s", out_file)

    return count


def run_processing(config: Dict, dry_run: bool = False) -> None:
    channels = config["channels"]
    for channel in channels:
        if channel not in CHANNEL_TO_BAND:
            raise ValueError(f"Unsupported channel '{channel}'. Supported: {sorted(CHANNEL_TO_BAND)}")

    years = config["years"]
    months = config["months"]
    overwrite = bool(config.get("overwrite", False))
    roi = config.get("roi", {})
    resolution_deg = float(config.get("processing", {}).get("target_resolution_deg", 0.01))
    resample = bool(config.get("processing", {}).get("resample", True))
    l1_reader = str(config.get("processing", {}).get("l1_reader", "modis_l1b"))
    l2_reader = str(config.get("processing", {}).get("l2_reader", "modis_l2"))
    cloud_mask_channel = config.get("processing", {}).get("cloud_mask_channel")
    cloud_mask_binary = bool(config.get("processing", {}).get("cloud_mask_binary", True))
    resample_chunk_rows = config.get("processing", {}).get("resample_chunk_rows")
    coord_decimals = config.get("processing", {}).get("coord_decimals")

    radiance_base = Path(config["download"]["radiance_base_path"])
    cloud_base = Path(config["download"]["cloud_mask_base_path"])
    output_base = Path(config["processing"]["output_base_path"])
    combine_satellites = bool(config.get("processing", {}).get("combine_satellites", False))

    for radiance_day_dir in iter_days(base_path=radiance_base, years=years, months=months):
        rel_day = radiance_day_dir.relative_to(radiance_base)
        cloud_day_dir = cloud_base / rel_day
        if combine_satellites:
            produced = process_day(
                radiance_day_dir=radiance_day_dir,
                cloud_day_dir=cloud_day_dir,
                output_base=output_base,
                channels=channels,
                satellite=None,
                roi=roi,
                l1_reader=l1_reader,
                l2_reader=l2_reader,
                cloud_mask_channel=cloud_mask_channel,
                cloud_mask_binary=cloud_mask_binary,
                resolution_deg=resolution_deg,
                resample=resample,
                dry_run=dry_run,
                overwrite=overwrite,
                resample_chunk_rows=resample_chunk_rows,
                coord_decimals=coord_decimals,
            )
            if produced:
                LOGGER.info("Processed %s files (combined satellites) in %s", produced, radiance_day_dir)
        else:
            for satellite in ("terra", "aqua"):
                produced = process_day(
                    radiance_day_dir=radiance_day_dir,
                    cloud_day_dir=cloud_day_dir,
                    output_base=output_base,
                    channels=channels,
                    satellite=satellite,
                    roi=roi,
                    l1_reader=l1_reader,
                    l2_reader=l2_reader,
                    cloud_mask_channel=cloud_mask_channel,
                    cloud_mask_binary=cloud_mask_binary,
                    resolution_deg=resolution_deg,
                    resample=resample,
                    dry_run=dry_run,
                    overwrite=overwrite,
                    resample_chunk_rows=resample_chunk_rows,
                    coord_decimals=coord_decimals,
                )
                if produced:
                    LOGGER.info("Processed %s files for %s in %s", produced, satellite, radiance_day_dir)
        if dry_run:
            LOGGER.info("Dry-run: processed only first day %s, exiting", rel_day)
            return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process MODIS L1/L2 files to BT+cloud mask NetCDF")
    parser.add_argument("--config", required=True, help="Path to YAML pipeline config")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--dry-run", action="store_true", help="Process only the first available day and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))

    # Determine logging level: CLI provides a default, but config.processing.verbose
    # can force verbose mode (DEBUG). This allows silencing via config as needed.
    cfg_proc = config.get("processing", {}) if isinstance(config, dict) else {}
    if bool(cfg_proc.get("verbose", False)):
        log_level = logging.DEBUG
    else:
        log_level = getattr(logging, args.log_level)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    if bool(getattr(args, "dry_run", False)):
        LOGGER.info("Running in dry-run mode")

    run_processing(config=config, dry_run=bool(getattr(args, "dry_run", False)))


if __name__ == "__main__":
    main()
