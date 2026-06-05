#!/usr/bin/env python3
"""
Process downloaded MODIS L1/L2 files into brightness-temperature (BT)
and cloud-mask NetCDF outputs.

This script scans directories of downloaded MODIS L1 radiance files and
corresponding L2 cloud-mask files, loads required bands, optionally
regrids/resamples data to a regular lat/lon grid, applies cloud masks,
and writes per-granule NetCDF files containing BT channels and a cloud
mask variable.

How to run
-----------
- Prepare a YAML pipeline config (see `config/pipeline_config.example.yaml`) and
    set the following keys at minimum: `download.radiance_base_path`,
    `download.cloud_mask_base_path`, `processing.output_base_path`, `channels`,
    `years`, `months`.
- Run from the repository root:

    python scripts/1_process_modis.py --config config/pipeline_config.yaml

- Useful flags:
    - `--dry-run`: run without writing output (processes only the first day)
    - `--log-level`: set logging verbosity (DEBUG/INFO/WARNING/ERROR)

Config notes
------------
The script reads processing options from the `processing` section of the
config, e.g. `target_resolution_deg`, `resample` (True/False),
`combine_satellites`, `l1_reader`, `l2_reader`, `cloud_mask_channel`,
`cloud_mask_binary`, `overwrite`, and `verbose`. Day selection comes from
`years`, `months`, and optional `days` in the YAML config, or from a single
`--date YYYY-MM-DD` override. See `config/pipeline_config.example.yaml` for
defaults and examples.

Output
------
Per-granule NetCDF files are written under the configured
`processing.output_base_path` organized by YYYY/MM/DD and named like
`YYYYMMDDTHHMM_modis_<sat>_<vars>.nc`.

Notes for developers
--------------------
- The main entry points are `process_day()` (processes one day's
    directories) and `run_processing()` (iterates configured days).
- Many helpers are implemented in `scripts/utils/` (loaders, regrid,
    cloudmask, netcdf utilities). Keep CLI/IO logic here and heavy
    data-processing in the utils modules.
"""

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

# Import utility helpers from scripts.utils
from scripts.utils.io import (
    load_config,
    iter_days,
    list_files,
    _is_l1_file,
    _is_l2_file,
    _extract_granule_key,
    _index_l2_files,
    _satellite_from_name,
)

from scripts.utils.regrid import (
    build_regrid_cache,
    _same_grid,
    _regrid_to_target,
    _resample_with_outside_nan,
    _coverage_extent_from_points,
    _coverage_extent_regular,
)

from scripts.utils.loaders import (
    _area_lonlats,
    _load_bt_datasets,
    _load_cloud_mask,
    _load_l2_with_geolocation,
)

from scripts.utils.cloudmask import (_resolve_cloud_mask_name, _cloud_mask_to_binary_values)

from scripts.utils.crop import (_largest_rectangle_from_mask, _crop_regular_dataset_to_valid_rectangle)

from scripts.utils.time_utils import _parse_time_from_name

from scripts.utils.netcdf_utils import safe_attr, prepare_netcdf_encoding


# Helper functions moved to scripts/utils/* modules
# Cloud mask helpers moved to scripts/utils/cloudmask.py
# BT loader moved to scripts/utils/loaders.py
# Cloud mask loader moved to scripts/utils/loaders.py
# L2 loader with geolocation moved to scripts/utils/loaders.py
# Time parsing moved to scripts/utils/time_utils.py
# `_regrid_to_target` moved to scripts/utils/regrid.py
# `_resample_with_outside_nan` moved to scripts/utils/regrid.py
# `_coverage_extent_regular` moved to scripts/utils/regrid.py
# `_coverage_extent_from_points` moved to scripts/utils/regrid.py


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
    compression_level: int = 4,
    parallel: bool = False,
    workers: int = 1,
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

    # Define single-granule processing function so we can optionally run in parallel
    def _process_single(l1_file: Path) -> int:
        produced = 0
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
            return 0
        try:
            LOGGER.info("Step: loading BT datasets for %s", l1_file.name)
            bt_data = _load_bt_datasets(l1_file=l1_file, channels=channels, reader=l1_reader)
            LOGGER.info("Step: loaded BT datasets for %s: %s", l1_file.name, ",".join(k for k in bt_data.keys() if k.startswith("bt_")))
        except Exception as e:
            LOGGER.exception("Failed to load BT datasets for %s: %s", l1_file.name, e)
            return 0

        # build a reference L1 grid from the first BT channel
        reference_key = f"bt_{channels[0]}"
        reference_da = bt_data.get(reference_key)
        if reference_da is None:
            LOGGER.warning("Skipping %s: missing reference BT channel %s", l1_file.name, reference_key)
            return 0
        ref_lon, ref_lat = _area_lonlats(reference_da)
        if ref_lon is None or ref_lat is None:
            LOGGER.warning("Skipping %s: missing geolocation for reference channel %s", l1_file.name, reference_key)
            return 0
        
        nan_lon = np.isnan(ref_lon).sum()
        nan_lat = np.isnan(ref_lat).sum()


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
                return 0

        # use local copies of the target grid inside the per-granule function
        # to avoid creating inner-scope assignments that shadow outer variables
        local_tgt_lon2d = tgt_lon2d
        local_tgt_lat2d = tgt_lat2d
        if not target_is_regular:
            local_tgt_lon2d = ref_lon
            local_tgt_lat2d = ref_lat

        bt_cache = None
        native_cache = None

        if target_is_regular:

            if nan_lon or nan_lat:
                LOGGER.warning(
                    "Skipping %s: geolocation contains NaNs "
                    "(lon=%s lat=%s)",
                    l1_file.name,
                    nan_lon,
                    nan_lat,
                )
                return 0

            bt_cache = build_regrid_cache(
                ref_lon,
                ref_lat,
                local_tgt_lon2d,
                local_tgt_lat2d,
            )

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
                    cache = bt_cache if bt_cache is not None and _same_grid(src_lon, src_lat, ref_lon, ref_lat) else None
                    re_da = _regrid_to_target(
                        src_da.astype(np.float32),
                        src_lon,
                        src_lat,
                        local_tgt_lon2d,
                        local_tgt_lat2d,
                        method="linear",
                        chunk_rows=resample_chunk_rows,
                        cache=cache,
                    )
                    data_vars_final[key] = re_da.rename({"y": "latitude", "x": "longitude"}).reset_coords(drop=True).rename(key).astype(np.float32)
                    LOGGER.debug("Resampled %s -> shape %s", key, data_vars_final[key].shape)
                except Exception:
                    LOGGER.exception("Failed regridding BT %s for %s", channel, l1_file.name)
            else:
                if src_da.shape == ref_lat.shape and src_da.shape == ref_lon.shape:
                    LOGGER.info(
                        "Step: keeping native L1 grid for %s",
                        channel,
                    )

                    data_vars_final[key] = (
                        src_da.astype(np.float32)
                        .rename(key)
                    )

                else:
                    LOGGER.info(
                        "Step: regridding BT %s onto L1 grid for %s",
                        channel,
                        l1_file.name,
                    )

                    if native_cache is None:

                        if nan_lon or nan_lat:
                            LOGGER.warning(
                                "Cannot build native regrid cache for %s "
                                "because geolocation contains NaNs",
                                l1_file.name,
                            )
                            return 0

   
                        native_cache = build_regrid_cache(
                            ref_lon,
                            ref_lat,
                            local_tgt_lon2d,
                            local_tgt_lat2d,
                        )

                    try:
                        re_da = _regrid_to_target(
                            src_da.astype(np.float32),
                            src_lon,
                            src_lat,
                            local_tgt_lon2d,
                            local_tgt_lat2d,
                            method="linear",
                            chunk_rows=resample_chunk_rows,
                            cache=native_cache,
                        )
                        data_vars_final[key] = re_da.rename(key).astype(np.float32)
                        LOGGER.debug("Regridded %s -> shape %s", key, data_vars_final[key].shape)
                    except Exception:
                        LOGGER.exception("Failed regridding BT %s for %s onto L1 grid", channel, l1_file.name)

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
                        cloud_cache = bt_cache if bt_cache is not None and _same_grid(cloud_lon, cloud_lat, ref_lon, ref_lat) else None
                        cloud_da = _resample_with_outside_nan(
                            cloud_src.astype(np.float32),
                            cloud_lon,
                            cloud_lat,
                            local_tgt_lon2d,
                            local_tgt_lat2d,
                            method="nearest",
                            chunk_rows=resample_chunk_rows,
                            cache=cloud_cache,
                        ).rename({"y": "latitude", "x": "longitude"}).reset_coords(drop=True).rename("cloud_mask").astype(np.float32)
                        LOGGER.debug("Cloud mask resampled -> shape %s", cloud_da.shape)
                    except Exception:
                        LOGGER.exception("Failed regridding cloud mask for %s", l2_file.name)
                else:
                    if cloud_src.shape == ref_lat.shape and cloud_src.shape == ref_lon.shape:
                        LOGGER.info("Step: keeping native L1 grid for cloud mask")
                        cloud_da = cloud_src.astype(np.float32).rename("cloud_mask")
                    else:
                        try:
                            cloud_cache = native_cache if native_cache is not None and _same_grid(cloud_lon, cloud_lat, ref_lon, ref_lat) else None
                            cloud_da = _resample_with_outside_nan(
                                cloud_src.astype(np.float32),
                                cloud_lon,
                                cloud_lat,
                                local_tgt_lon2d,
                                local_tgt_lat2d,
                                method="nearest",
                                chunk_rows=resample_chunk_rows,
                                cache=cloud_cache,
                            ).rename("cloud_mask").astype(np.float32)
                            LOGGER.debug("Cloud mask regridded onto L1 grid -> shape %s", cloud_da.shape)
                        except Exception:
                            LOGGER.exception("Failed regridding cloud mask for %s onto L1 grid", l2_file.name)

        if cloud_da is not None:
            data_vars_final["cloud_mask"] = cloud_da

        if not data_vars_final:
            LOGGER.warning("Skipping %s: no variables available after loading/resampling", l1_file.name)
            return 0

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
                return 0
            lon_extent, lat_extent = _coverage_extent_regular(valid_mask, regular_lons, regular_lats)
            LOGGER.info("Resampled coverage extent for %s -> lon: %.2f deg, lat: %.2f deg", l1_file.name, lon_extent, lat_extent)
            if lon_extent < 4.0 or lat_extent < 4.0:
                LOGGER.info(
                    "Skipping %s: resampled footprint too small (lon %.2f deg, lat %.2f deg)",
                    l1_file.name,
                    lon_extent,
                    lat_extent,
                )
                return 0

        timestamp = _parse_time_from_name(l1_file)
        out_dir = output_base / f"{timestamp.year:04d}" / f"{timestamp.month:02d}" / f"{timestamp.day:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_sat = satellite if satellite is not None else (_satellite_from_name(l1_file) or "unknown")
        var_tag = "_".join(data_vars_final.keys())
        out_file = out_dir / f"{timestamp:%Y%m%dT%H%M}_modis_{out_sat}_{var_tag}.nc"
        if out_file.exists() and not overwrite:
            LOGGER.debug("Skipping existing processed file %s", out_file)
            return 0

        coords: Dict[str, Tuple[Tuple[str, ...], np.ndarray] | np.ndarray] = {}
        if target_is_regular:
            coords["latitude"] = ("latitude", np.asarray(regular_lats, dtype=np.float32))
            coords["longitude"] = ("longitude", np.asarray(regular_lons, dtype=np.float32))
            data_vars_final = {
                name: da.rename({"y": "latitude", "x": "longitude"}).reset_coords(drop=True) if set(da.dims) == {"y", "x"} else da
                for name, da in data_vars_final.items()
            }
        else:
            coords["latitude"] = (("y", "x"), np.asarray(local_tgt_lat2d, dtype=np.float32))
            coords["longitude"] = (("y", "x"), np.asarray(local_tgt_lon2d, dtype=np.float32))

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
        # Sanitize attributes using shared helper and prepare NetCDF encodings
        for k in list(dataset.attrs.keys()):
            dataset.attrs[k] = safe_attr(dataset.attrs[k])

        for name in list(dataset.data_vars) + list(dataset.coords):
            obj = dataset[name]
            for k in list(obj.attrs.keys()):
                val = obj.attrs[k]
                if k == "coordinates" and isinstance(val, (list, tuple)):
                    obj.attrs[k] = " ".join(str(item) for item in val)
                else:
                    obj.attrs[k] = safe_attr(val)

        # Convert regular-grid datasets to the requested y/x DataArray layout
        # with 1D coords named `lat` (y) and `lon` (x) before writing. Ensure
        # `lat`/`lon` are coordinates (not data variables) and final dataset
        # dims are (time, y, x).
        dataset_to_write = dataset
        if target_is_regular:
            try:
                # prefer existing 1D axes if available, otherwise use computed regular arrays
                lat_coord = dataset.coords.get("latitude")
                lon_coord = dataset.coords.get("longitude")
                lat_vals = (
                    np.asarray(lat_coord.values, dtype=np.float32)
                    if lat_coord is not None and getattr(lat_coord, "ndim", 1) == 1
                    else np.asarray(regular_lats, dtype=np.float32)
                )
                lon_vals = (
                    np.asarray(lon_coord.values, dtype=np.float32)
                    if lon_coord is not None and getattr(lon_coord, "ndim", 1) == 1
                    else np.asarray(regular_lons, dtype=np.float32)
                )

                ds_out = xr.Dataset()
                for name in dataset.data_vars:
                    var = dataset[name]
                    vals = np.asarray(var.values)
                    # collapse leading time singleton if present
                    if vals.ndim == 3 and vals.shape[0] == 1:
                        vals2 = vals[0]
                    elif vals.ndim == 2:
                        vals2 = vals
                    else:
                        vals2 = np.full((len(lat_vals), len(lon_vals)), np.nan, dtype=np.float32)

                    # ensure shape matches (len(lat), len(lon))
                    if vals2.shape != (len(lat_vals), len(lon_vals)):
                        tmp = np.full((len(lat_vals), len(lon_vals)), np.nan, dtype=np.float32)
                        try:
                            r0 = min(tmp.shape[0], vals2.shape[0])
                            c0 = min(tmp.shape[1], vals2.shape[1])
                            tmp[:r0, :c0] = vals2[:r0, :c0]
                            vals2 = tmp
                        except Exception:
                            vals2 = tmp

                    da = xr.DataArray(
                        vals2.astype(np.float32),
                        dims=("y", "x"),
                        coords={"lat": ("y", lat_vals), "lon": ("x", lon_vals)},
                        name=name,
                    )
                    # copy attributes
                    for k, v in var.attrs.items():
                        if k == "coordinates" and isinstance(v, (list, tuple)):
                            da.attrs[k] = " ".join(str(item) for item in v)
                        else:
                            da.attrs[k] = safe_attr(v)
                    ds_out[name] = da

                # remove any accidental datavars named 'lat'/'lon' or old 'latitude'/'longitude'
                for bad in ("lat", "lon", "latitude", "longitude"):
                    if bad in ds_out.data_vars:
                        ds_out = ds_out.drop_vars(bad)

                # expand time dimension as the leading axis
                ds_out = ds_out.expand_dims(time=[np.datetime64(timestamp)])

                # attach 1D coordinate arrays (not variables) and copy global attrs
                ds_out = ds_out.assign_coords({"lat": ("y", lat_vals), "lon": ("x", lon_vals)})
                for k, v in dataset.attrs.items():
                    ds_out.attrs[k] = safe_attr(v)

                dataset_to_write = ds_out
            except Exception:
                LOGGER.exception("Failed to convert dataset to requested regular layout; falling back to original dataset")

        if not target_is_regular:
            try:
                dataset_to_write = dataset_to_write.rename({"latitude": "lat", "longitude": "lon"})
            except Exception:
                pass

            # Keep native-grid outputs readable by stripping Satpy-specific
            # metadata and retaining only a small, useful attribute set.
            for name in list(dataset_to_write.variables):
                obj = dataset_to_write[name]
                source_attrs = dict(obj.attrs)
                cleaned_attrs: Dict[str, object] = {}
                if name in {"lat", "lon"}:
                    for key in ("standard_name", "units", "long_name"):
                        if key in source_attrs:
                            cleaned_attrs[key] = safe_attr(source_attrs[key])
                elif name == "cloud_mask":
                    cleaned_attrs["long_name"] = "Binary MODIS cloud mask" if cloud_mask_binary else safe_attr(source_attrs.get("long_name", "MODIS cloud mask"))
                    cleaned_attrs["units"] = "1" if cloud_mask_binary else safe_attr(source_attrs.get("units", "none"))
                    cleaned_attrs["coordinates"] = "lat lon"
                    if cloud_mask_binary:
                        cleaned_attrs["flag_values"] = [0, 1]
                        cleaned_attrs["flag_meanings"] = "clear cloudy"
                elif name.startswith("bt_"):
                    for key in ("long_name", "units", "standard_name"):
                        if key in source_attrs:
                            cleaned_attrs[key] = safe_attr(source_attrs[key])
                    cleaned_attrs["coordinates"] = "lat lon"
                elif name not in {"time"}:
                    for key in ("long_name", "units", "standard_name", "coordinates"):
                        if key in source_attrs:
                            cleaned_attrs[key] = safe_attr(source_attrs[key])

                obj.attrs.clear()
                obj.attrs.update(cleaned_attrs)

            dataset_to_write.attrs.pop("crs", None)
            dataset_to_write.attrs["grid_type"] = "l1_native"

        # NetCDF backends can't serialize arbitrary Python objects (for example
        # a CRS object stored in a scalar variable). Preserve a string version
        # in attrs and drop object-typed variables before writing.
        object_vars: list[str] = []
        for name in list(dataset_to_write.variables):
            try:
                if dataset_to_write[name].dtype != object:
                    continue
            except Exception:
                continue

            object_vars.append(name)
            value_str = ""
            try:
                scalar = dataset_to_write[name].values.item()
                if hasattr(scalar, "to_wkt"):
                    value_str = str(scalar.to_wkt())
                elif hasattr(scalar, "to_string"):
                    value_str = str(scalar.to_string())
                else:
                    value_str = str(scalar)
            except Exception:
                try:
                    value_str = str(dataset_to_write[name].values)
                except Exception:
                    value_str = ""

            if name == "crs":
                dataset_to_write.attrs["crs"] = safe_attr(value_str)
            else:
                dataset_to_write.attrs[f"{name}_value"] = safe_attr(value_str)

        drop_vars = [name for name in object_vars if name in dataset_to_write.data_vars or name in dataset_to_write.coords]
        if drop_vars:
            dataset_to_write = dataset_to_write.drop_vars(drop_vars)

        enc_out = prepare_netcdf_encoding(dataset_to_write, compression_level=compression_level)

        if dry_run:
            LOGGER.info("Dry-run: would write %s (vars: %s)", out_file, ",".join(dataset_to_write.data_vars))
            produced += 1
        else:
            dataset_to_write.to_netcdf(out_file, encoding=enc_out)
            produced += 1
            LOGGER.info("Wrote %s", out_file)

        return produced

    # Execute per-granule processing either in parallel (threads) or serially
    if parallel and workers and workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        LOGGER.info("Processing %d granules with %d workers", len(l1_files), workers)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_process_single, f): f for f in l1_files}
            for fut in as_completed(futures):
                try:
                    produced = fut.result()
                    count += produced
                except Exception:
                    LOGGER.exception("Exception in worker while processing %s", futures.get(fut))
    else:
        for l1_file in l1_files:
            count += _process_single(l1_file)

    return count


def run_processing(config: Dict, dry_run: bool = False, date: dt.date | None = None) -> None:
    channels = config["channels"]
    for channel in channels:
        if channel not in CHANNEL_TO_BAND:
            raise ValueError(f"Unsupported channel '{channel}'. Supported: {sorted(CHANNEL_TO_BAND)}")

    years = [date.year] if date is not None else config["years"]
    months = [date.month] if date is not None else config["months"]
    days = [date.day] if date is not None else config.get("days", "all")
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
    compression_level = int(config.get("processing", {}).get("compression_level", 4))
    parallel = bool(config.get("processing", {}).get("parallel", False))
    workers = int(config.get("processing", {}).get("workers", 1))

    radiance_base = Path(config["download"]["radiance_base_path"])
    cloud_base = Path(config["download"]["cloud_mask_base_path"])
    output_base = Path(config["processing"]["output_base_path"])
    combine_satellites = bool(config.get("processing", {}).get("combine_satellites", False))

    for radiance_day_dir in iter_days(base_path=radiance_base, years=years, months=months, days=days):
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
                compression_level=compression_level,
                parallel=parallel,
                workers=workers,
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
                    compression_level=compression_level,
                    parallel=parallel,
                    workers=workers,
                )
                if produced:
                    LOGGER.info("Processed %s files for %s in %s", produced, satellite, radiance_day_dir)
        if dry_run:
            LOGGER.info("Dry-run: processed only first day %s, exiting", rel_day)
            return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process MODIS L1/L2 files to BT+cloud mask NetCDF")
    parser.add_argument("--config", required=True, help="Path to YAML pipeline config")
    parser.add_argument("--date", default=None, help="Process only one day in YYYY-MM-DD format")
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

    selected_date = dt.date.fromisoformat(args.date) if args.date else None
    run_processing(config=config, dry_run=bool(getattr(args, "dry_run", False)), date=selected_date)


if __name__ == "__main__":
    main()
