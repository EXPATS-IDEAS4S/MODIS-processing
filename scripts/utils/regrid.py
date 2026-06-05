from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import xarray as xr


@dataclass
class RegridCache:
    source_lon: np.ndarray
    source_lat: np.ndarray
    target_lon: np.ndarray
    target_lat: np.ndarray

    def __post_init__(self) -> None:
        self.source_lon = np.asarray(self.source_lon, dtype=np.float64)
        self.source_lat = np.asarray(self.source_lat, dtype=np.float64)
        self.target_lon = np.asarray(self.target_lon, dtype=np.float64)
        self.target_lat = np.asarray(self.target_lat, dtype=np.float64)
        self.source_points = np.column_stack((self.source_lon.ravel(), self.source_lat.ravel()))
        self.target_points = np.column_stack((self.target_lon.ravel(), self.target_lat.ravel()))
        self.target_shape = self.target_lon.shape

        from scipy.spatial import Delaunay, cKDTree

        self.triangulation = Delaunay(self.source_points)
        self.target_simplex = self.triangulation.find_simplex(self.target_points)
        self.nearest_tree = cKDTree(self.source_points)
        self.nearest_indices = self.nearest_tree.query(self.target_points, k=1)[1]

    def _as_dataarray(self, grid_z: np.ndarray, dtype) -> xr.DataArray:
        da = xr.DataArray(np.asarray(grid_z).reshape(self.target_shape).astype(dtype), dims=("y", "x"))
        da = da.assign_coords({"latitude": (("y", "x"), self.target_lat), "longitude": (("y", "x"), self.target_lon)})
        return da

    def linear(self, src_values: np.ndarray, fill_value: float = np.nan, chunk_rows: Optional[int] = None) -> xr.DataArray:
        from scipy.interpolate import LinearNDInterpolator

        values = np.asarray(src_values)
        interpolator = LinearNDInterpolator(self.triangulation, values.ravel(), fill_value=fill_value)

        if chunk_rows and self.target_lat.ndim == 2:
            rows = self.target_lat.shape[0]
            pieces = []
            for start in range(0, rows, chunk_rows):
                stop = min(start + chunk_rows, rows)
                sub_points = np.column_stack((self.target_lon[start:stop, :].ravel(), self.target_lat[start:stop, :].ravel()))
                grid_piece = interpolator(sub_points).reshape(stop - start, self.target_lon.shape[1])
                pieces.append(grid_piece)
            grid_z = np.vstack(pieces)
        else:
            grid_z = interpolator(self.target_points).reshape(self.target_shape)

        return self._as_dataarray(grid_z, values.dtype)

    def nearest(self, src_values: np.ndarray, outside_nan: bool = True) -> xr.DataArray:
        values = np.asarray(src_values)
        grid_z = values.ravel()[self.nearest_indices].reshape(self.target_shape)
        if outside_nan:
            grid_z = np.asarray(grid_z, dtype=np.float64)
            grid_z[self.target_simplex.reshape(self.target_shape) < 0] = np.nan
        return self._as_dataarray(grid_z, values.dtype)


def build_regrid_cache(
    src_lon: np.ndarray,
    src_lat: np.ndarray,
    target_lon: np.ndarray,
    target_lat: np.ndarray,
) -> RegridCache:
    return RegridCache(source_lon=src_lon, source_lat=src_lat, target_lon=target_lon, target_lat=target_lat)


def _same_grid(a_lon: np.ndarray, a_lat: np.ndarray, b_lon: np.ndarray, b_lat: np.ndarray, atol: float = 1e-6) -> bool:
    return (
        np.asarray(a_lon).shape == np.asarray(b_lon).shape
        and np.asarray(a_lat).shape == np.asarray(b_lat).shape
        and np.allclose(np.asarray(a_lon), np.asarray(b_lon), equal_nan=True, atol=atol)
        and np.allclose(np.asarray(a_lat), np.asarray(b_lat), equal_nan=True, atol=atol)
    )


def _regrid_to_target(
    src_da: xr.DataArray,
    src_lon: np.ndarray,
    src_lat: np.ndarray,
    target_lon: np.ndarray,
    target_lat: np.ndarray,
    method: str = "linear",
    chunk_rows: Optional[int] = None,
    cache: Optional[RegridCache] = None,
) -> xr.DataArray:
    """Regrid a 2D DataArray given source lon/lat and target lon/lat (2D).

    Uses scipy.interpolate.griddata. Returns DataArray with dims ('y','x')
    and coords 'latitude' and 'longitude'.
    """
    cache = cache or build_regrid_cache(src_lon=src_lon, src_lat=src_lat, target_lon=target_lon, target_lat=target_lat)
    if method == "nearest":
        return cache.nearest(src_da.values, outside_nan=True)
    return cache.linear(src_da.values, chunk_rows=chunk_rows)


def _resample_with_outside_nan(
    src_da: xr.DataArray,
    src_lon: np.ndarray,
    src_lat: np.ndarray,
    target_lon: np.ndarray,
    target_lat: np.ndarray,
    method: str = "linear",
    chunk_rows: Optional[int] = None,
    cache: Optional[RegridCache] = None,
) -> xr.DataArray:
    cache = cache or build_regrid_cache(src_lon=src_lon, src_lat=src_lat, target_lon=target_lon, target_lat=target_lat)
    if method == "nearest":
        return cache.nearest(src_da.values, outside_nan=True)
    return cache.linear(src_da.values, chunk_rows=chunk_rows)


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
