from __future__ import annotations

from typing import Optional, Tuple, Dict

import numpy as np
import xarray as xr


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
    except Exception:
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
