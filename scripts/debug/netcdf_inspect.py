#!/usr/bin/env python3
"""NetCDF inspection helpers for MODIS debug scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import xarray as xr


def _nc_geolocation_for_var(ds: xr.Dataset, data: xr.DataArray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    lon_name = None
    lat_name = None
    for candidate in ("longitude", "lon"):
        if candidate in data.coords:
            lon_name = candidate
            break
    for candidate in ("latitude", "lat"):
        if candidate in data.coords:
            lat_name = candidate
            break

    if lon_name is not None and lat_name is not None:
        lon = np.asarray(data.coords[lon_name].values)
        lat = np.asarray(data.coords[lat_name].values)
        if lon.ndim == 1 and lat.ndim == 1:
            lon2d, lat2d = np.meshgrid(lon, lat)
            return lon2d, lat2d
        return lon, lat

    lon_name = None
    lat_name = None
    for candidate in ("longitude", "lon"):
        if candidate in ds.coords:
            lon_name = candidate
            break
    for candidate in ("latitude", "lat"):
        if candidate in ds.coords:
            lat_name = candidate
            break

    if lon_name is not None and lat_name is not None:
        lon = np.asarray(ds.coords[lon_name].values)
        lat = np.asarray(ds.coords[lat_name].values)
        if lon.ndim == 1 and lat.ndim == 1:
            lon2d, lat2d = np.meshgrid(lon, lat)
            return lon2d, lat2d
        return lon, lat

    return None, None


def inspect_file(path: Path) -> None:
    ds = xr.open_dataset(path)
    print("=== File ===")
    print(path)
    print("=== Dataset ===")
    print(ds)
    print("=== Variables statistics ===")
    for name, data in ds.data_vars.items():
        values = data.values
        values = values[np.isfinite(values)] if np.issubdtype(values.dtype, np.floating) else values
        if values.size == 0:
            stats = {"min": None, "max": None, "mean": None}
        else:
            stats = {"min": float(np.nanmin(values)), "max": float(np.nanmax(values)), "mean": float(np.nanmean(values))}
        print(name, stats)

    def _report_grid_info(dataset: xr.Dataset):
        data_vars_2d = [n for n, d in dataset.data_vars.items() if d.squeeze().ndim >= 2]
        lon = lat = None
        if data_vars_2d:
            lon, lat = _nc_geolocation_for_var(dataset, dataset[data_vars_2d[0]])
        if lon is None or lat is None:
            if "longitude" in dataset.coords and "latitude" in dataset.coords:
                lon = np.asarray(dataset.coords["longitude"].values)
                lat = np.asarray(dataset.coords["latitude"].values)
            elif "lon" in dataset.coords and "lat" in dataset.coords:
                lon = np.asarray(dataset.coords["lon"].values)
                lat = np.asarray(dataset.coords["lat"].values)

        if lon is None or lat is None:
            print("No geolocation (latitude/longitude) coordinates found in this dataset.")
            return

        print(f"longitude ndim={lon.ndim}, latitude ndim={lat.ndim}")

        def _corners_from_arrays(lon_arr, lat_arr):
            if lon_arr.ndim == 1 and lat_arr.ndim == 1:
                lon_min = float(np.nanmin(lon_arr))
                lon_max = float(np.nanmax(lon_arr))
                lat_min = float(np.nanmin(lat_arr))
                lat_max = float(np.nanmax(lat_arr))
                return (lon_min, lat_max), (lon_max, lat_max), (lon_min, lat_min), (lon_max, lat_min)
            return (float(lon_arr[0, 0]), float(lat_arr[0, 0])), (float(lon_arr[0, -1]), float(lat_arr[0, -1])), (float(lon_arr[-1, 0]), float(lat_arr[-1, 0])), (float(lon_arr[-1, -1]), float(lat_arr[-1, -1]))

        corners = _corners_from_arrays(lon, lat)
        print("Corners (TL, TR, BL, BR) as (lon, lat):")
        for corner in corners:
            print(f"  {corner}")

        def _is_regular_1d(arr):
            if arr.ndim != 1 or arr.size < 2:
                return True, None
            diffs = np.diff(arr)
            ok = np.allclose(diffs, diffs[0], rtol=1e-6, atol=1e-9)
            return bool(ok), float(diffs[0])

        def _is_regular_2d(lon2d, lat2d):
            lon_var_x = np.allclose(lon2d - lon2d[:, 0:1], np.tile(lon2d[0, :], (lon2d.shape[0], 1)), atol=1e-6)
            lat_var_y = np.allclose(lat2d - lat2d[0:1, :], np.tile(lat2d[:, 0], (lat2d.shape[1], 1)).T, atol=1e-6)
            if lon_var_x and lat_var_y:
                lon1d = lon2d[0, :]
                lat1d = lat2d[:, 0]
                lon_reg, lon_res = _is_regular_1d(lon1d)
                lat_reg, lat_res = _is_regular_1d(lat1d)
                return lon_reg and lat_reg, lon_res, lat_res
            return False, None, None

        if lon.ndim == 1 and lat.ndim == 1:
            lon_reg, lon_res = _is_regular_1d(lon)
            lat_reg, lat_res = _is_regular_1d(lat)
            print(f"1D coordinate grids. longitude regular={lon_reg}, resolution={lon_res}; latitude regular={lat_reg}, resolution={lat_res}")
            print(f"Overall grid regular: {bool(lon_reg and lat_reg)}")
        else:
            reg, lon_res, lat_res = _is_regular_2d(lon, lat)
            print(f"2D coordinate grids separable regular={reg}, lon_res={lon_res}, lat_res={lat_res}")

    print("=== Grid inspection ===")
    try:
        _report_grid_info(ds)
    except Exception as exc:
        print(f"Grid inspection failed: {exc}")
