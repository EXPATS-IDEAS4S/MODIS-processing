from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
from satpy import DataQuery, Scene

from scripts.utils.cloudmask import _cloud_mask_to_binary_values

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


def _resolve_cloud_mask_name(available: List[str], preferred_name: Optional[str] = None) -> Optional[str]:
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


def _load_cloud_mask(
    l2_file: Path,
    reader: str = "modis_l2",
    cloud_mask_channel: Optional[str] = None,
    cloud_mask_binary: bool = True,
    area_def=None,
) -> Optional[Tuple[xr.DataArray, Optional[np.ndarray], Optional[np.ndarray]]]:
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

    cloud_values = _cloud_mask_to_binary_values(cloud_data.values, cloud_mask_binary)
    cloud_data = xr.DataArray(cloud_values, dims=cloud_data.dims, coords=cloud_data.coords, attrs=dict(cloud_data.attrs)).rename("cloud_mask")
    cloud_data.attrs.update({"long_name": "MODIS cloud mask"})
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
