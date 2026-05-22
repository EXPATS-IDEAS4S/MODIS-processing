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
try:
    from pyresample.geometry import AreaDefinition
except Exception:  # pyresample optional
    AreaDefinition = None

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


def _load_bt_datasets(l1_file: Path, channels: List[str], area_def=None) -> Dict[str, xr.DataArray]:
    band_names = [CHANNEL_TO_BAND[channel] for channel in channels]
    queries = [DataQuery(name=band, calibration="brightness_temperature") for band in band_names]
    scene = Scene(reader="modis_l1b", filenames=[str(l1_file)])

    # inspect available dataset names and pick lat/lon names case-insensitively
    try:
        available = list(scene.available_dataset_names())
    except Exception:
        available = []

    # prefer explicit 'latitude'/'longitude' but fall back to any name containing lat/lon
    lat_name = next((n for n in available if n.lower() == "latitude"), None)
    lon_name = next((n for n in available if n.lower() == "longitude"), None)
    if lat_name is None:
        lat_name = next((n for n in available if "lat" in n.lower()), None)
    if lon_name is None:
        lon_name = next((n for n in available if "lon" in n.lower()), None)

    # try to load BT bands and (if available) the detected lat/lon names
    load_items = list(queries)
    if lat_name:
        load_items.append(lat_name)
    if lon_name:
        load_items.append(lon_name)

    try:
        scene.load(load_items)
    except Exception:
        # fallback: try with DataQuery wrappers for lat/lon names
        try:
            load_items = list(queries)
            if lat_name:
                load_items.append(DataQuery(name=lat_name))
            if lon_name:
                load_items.append(DataQuery(name=lon_name))
            scene.load(load_items)
        except Exception:
            # last resort: load only BT bands
            scene.load(queries)

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

    # retrieve lat/lon DataArrays if available (try string-key then DataQuery)
    if lat_name and lon_name:
        try:
            lat = scene[lat_name]
        except Exception:
            try:
                lat = scene[DataQuery(name=lat_name)]
            except Exception:
                lat = None
        try:
            lon = scene[lon_name]
        except Exception:
            try:
                lon = scene[DataQuery(name=lon_name)]
            except Exception:
                lon = None
        if lat is not None and lon is not None:
            data["latitude"] = lat
            data["longitude"] = lon

    return data


def _load_cloud_mask(l2_file: Path, area_def=None) -> Optional[xr.DataArray]:
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

    # detect lat/lon dataset names
    available = []
    try:
        available = list(scene.available_dataset_names())
    except Exception:
        available = []
    lat_name = next((n for n in available if "lat" in n.lower()), None)
    lon_name = next((n for n in available if "lon" in n.lower()), None)

    try:
        # MODIS cloud mask in this pipeline should be taken at 1000 m resolution.
        load_list = [DataQuery(name=dataset_name, resolution=1000)]
        if lat_name:
            load_list.append(DataQuery(name=lat_name, resolution=1000))
        if lon_name:
            load_list.append(DataQuery(name=lon_name, resolution=1000))
        scene.load(load_list)
        cloud_data = scene[DataQuery(name=dataset_name, resolution=1000)]
    except Exception:
        try:
            scene.load([DataQuery(name=dataset_name, resolution=1000)])
            cloud_data = scene[DataQuery(name=dataset_name, resolution=1000)]
        except Exception:
            return None

    # try to attach lat/lon if available
    try:
        if lat_name and lon_name:
            try:
                lat = scene[DataQuery(name=lat_name, resolution=1000)]
            except Exception:
                lat = scene[DataQuery(name=lat_name)]
            try:
                lon = scene[DataQuery(name=lon_name, resolution=1000)]
            except Exception:
                lon = scene[DataQuery(name=lon_name)]
            cloud_data.attrs.update({"has_geolocation": True})
            cloud_data_latlon = (lat, lon)
        else:
            cloud_data_latlon = (None, None)
    except Exception:
        cloud_data_latlon = (None, None)

    cloud_data = cloud_data.astype(np.uint8).rename("cloud_mask")
    cloud_data.attrs.update({"long_name": "MODIS cloud mask"})
    # attach lat/lon pair as attribute for caller
    cloud_data.attrs["_latlon_pair"] = cloud_data_latlon
    return cloud_data


def _load_l2_with_geolocation(l2_file: Path) -> Tuple[Optional[xr.DataArray], Optional[xr.DataArray], Optional[xr.DataArray]]:
    """Load L2 cloud mask and return (cloud_mask, latitude, longitude) DataArrays when available.

    Returns a tuple of (cloud_mask, lat, lon) where missing items are None.
    """
    scene = Scene(reader="modis_l2", filenames=[str(l2_file)])
    try:
        available = list(scene.available_dataset_names())
    except Exception:
        available = []

    # pick cloud mask dataset name
    preferred = ["cloud_mask", "Cloud_Mask", "cloud_mask_byte_segment"]
    cloud_name = next((n for n in preferred if n in available), None)
    if cloud_name is None:
        # fallback: any name containing 'cloud'
        cloud_name = next((n for n in available if "cloud" in n.lower()), None)
    if cloud_name is None:
        return None, None, None

    # detect lat/lon names
    lat_name = next((n for n in available if n.lower() == "latitude"), None)
    lon_name = next((n for n in available if n.lower() == "longitude"), None)
    if lat_name is None:
        lat_name = next((n for n in available if "lat" in n.lower()), None)
    if lon_name is None:
        lon_name = next((n for n in available if "lon" in n.lower()), None)

    load_list = [DataQuery(name=cloud_name, resolution=1000)]
    if lat_name:
        load_list.append(DataQuery(name=lat_name, resolution=1000))
    if lon_name:
        load_list.append(DataQuery(name=lon_name, resolution=1000))

    try:
        scene.load(load_list)
    except Exception:
        # try with DataQuery wrappers
        try:
            dq = [DataQuery(name=cloud_name)]
            if lat_name:
                dq.append(DataQuery(name=lat_name))
            if lon_name:
                dq.append(DataQuery(name=lon_name))
            scene.load(dq)
        except Exception:
            # try loading cloud alone
            scene.load([cloud_name])

    try:
        cloud = scene[DataQuery(name=cloud_name, resolution=1000)].astype(np.uint8).rename("cloud_mask")
    except Exception:
        try:
            cloud = scene[DataQuery(name=cloud_name, resolution=1000)].astype(np.uint8).rename("cloud_mask")
        except Exception:
            cloud = None

    lat = None
    lon = None
    if lat_name and lon_name:
        try:
            lat = scene[DataQuery(name=lat_name, resolution=1000)]
        except Exception:
            try:
                lat = scene[DataQuery(name=lat_name)]
            except Exception:
                lat = None
        try:
            lon = scene[DataQuery(name=lon_name, resolution=1000)]
        except Exception:
            try:
                lon = scene[DataQuery(name=lon_name)]
            except Exception:
                lon = None

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


def make_latlon_area(roi: Dict[str, float], resolution_deg: float = 0.01):
    if AreaDefinition is None:
        return None
    lon_min = roi["lon_min"]
    lon_max = roi["lon_max"]
    lat_min = roi["lat_min"]
    lat_max = roi["lat_max"]
    width = max(1, int((lon_max - lon_min) / resolution_deg))
    height = max(1, int((lat_max - lat_min) / resolution_deg))
    area_extent = (lon_min, lat_min, lon_max, lat_max)
    proj_dict = {"proj": "latlong"}
    return AreaDefinition("target", "latlon", proj_dict, width, height, area_extent)


def _regrid_to_target(src_da: xr.DataArray, src_lon: np.ndarray, src_lat: np.ndarray, target_lon: np.ndarray, target_lat: np.ndarray, method: str = "linear") -> xr.DataArray:
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

    # griddata expects (nx, ny) matching meshgrid order; provide (lon, lat)
    grid_z = griddata(points, values, (tgt_lon, tgt_lat), method=method)

    da = xr.DataArray(grid_z.astype(values.dtype), dims=("y", "x"))
    da = da.assign_coords({"latitude": (("y", "x"), tgt_lat), "longitude": (("y", "x"), tgt_lon)})
    return da


def process_day(
    radiance_day_dir: Path,
    cloud_day_dir: Path,
    output_base: Path,
    channels: List[str],
    satellite: str,
    roi: Dict[str, float],
    resolution_deg: float = 0.01,
    resample: bool = True,
    overwrite: bool = False,
) -> int:
    l1_files, l2_files = list_files(radiance_day_dir=radiance_day_dir, cloud_day_dir=cloud_day_dir, satellite=satellite)
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
        lons = np.arange(lon_min, lon_max + res, res)
        lats = np.arange(lat_max, lat_min - res, -res)
        tgt_lon2d, tgt_lat2d = np.meshgrid(lons, lats)
        target_is_regular = True
    else:
        tgt_lon2d = None
        tgt_lat2d = None
        target_is_regular = False

    for l1_file in l1_files:
        granule_key = _extract_granule_key(l1_file)
        l2_file = l2_index.get(granule_key)
        bt_data = _load_bt_datasets(l1_file=l1_file, channels=channels)

        # extract source geolocation (L1 native)
        src_lat = None
        src_lon = None
        if "latitude" in bt_data and "longitude" in bt_data:
            src_lat = bt_data.pop("latitude").values
            src_lon = bt_data.pop("longitude").values

        if src_lat is None or src_lon is None:
            LOGGER.warning("Skipping %s: missing geolocation (lat/lon)", l1_file.name)
            continue

        # determine actual target grid for this granule
        if not target_is_regular:
            # target is L1 native grid
            tgt_lon2d = src_lon
            tgt_lat2d = src_lat

        # regrid BT channels if target is regular; otherwise keep L1 native
        regridded = {}
        for channel in channels:
            key = f"bt_{channel}"
            if key not in bt_data:
                continue
            src_da = bt_data[key]
            if target_is_regular:
                try:
                    re_da = _regrid_to_target(src_da, src_lon, src_lat, tgt_lon2d, tgt_lat2d, method="linear")
                    regridded[key] = re_da.rename(key)
                except Exception:
                    LOGGER.warning("Failed regridding BT %s for %s; using native", channel, l1_file.name)
                    regridded[key] = src_da
            else:
                # preserve native L1 array
                regridded[key] = src_da

        # cloud mask: load L2 and regrid to target grid (regular or L1 native)
        cloud_da = None
        if l2_file is not None:
            cloud_src = _load_cloud_mask(l2_file=l2_file)
            if cloud_src is not None:
                # try to get L2 geolocation; fallback to L1 geolocation
                latlon = cloud_src.attrs.get("_latlon_pair")
                if latlon and latlon[0] is not None and latlon[1] is not None:
                    c_src_lat = latlon[0].values
                    c_src_lon = latlon[1].values
                else:
                    c_src_lat = src_lat
                    c_src_lon = src_lon

                if c_src_lat is not None and c_src_lon is not None:
                    try:
                        cloud_re = _regrid_to_target(cloud_src.astype(np.float32), c_src_lon, c_src_lat, tgt_lon2d, tgt_lat2d, method="nearest")
                        # keep cloud mask as float32 so missing points can be NaN
                        cloud_da = cloud_re.rename("cloud_mask").astype(np.float32)
                    except Exception:
                        LOGGER.warning("Failed regridding cloud mask for %s; skipping cloud_mask", l2_file.name)
                        cloud_da = None

        # assemble final dataset using only the requested channels + cloud_mask
        data_vars_final = {}
        target_shape = tgt_lon2d.shape
        for k, v in regridded.items():
            # ensure variable matches target grid shape
            try:
                if getattr(v, "shape", None) != target_shape:
                    LOGGER.warning("Dropping %s: shape %s doesn't match target %s", k, getattr(v, "shape", None), target_shape)
                    continue
            except Exception:
                LOGGER.warning("Dropping %s: unable to determine shape", k)
                continue
            data_vars_final[k] = v
        if cloud_da is not None:
            if getattr(cloud_da, "shape", None) == target_shape:
                data_vars_final["cloud_mask"] = cloud_da
            else:
                LOGGER.warning("Dropping cloud_mask: shape %s doesn't match target %s", getattr(cloud_da, "shape", None), target_shape)

        # set coords latitude/longitude from target grid
        lat_coord = xr.DataArray(tgt_lat2d, dims=("y", "x"))
        lon_coord = xr.DataArray(tgt_lon2d, dims=("y", "x"))
        dataset = xr.Dataset(data_vars=data_vars_final, coords={"latitude": (("y", "x"), tgt_lat2d), "longitude": (("y", "x"), tgt_lon2d)})
        timestamp = _parse_time_from_name(l1_file)
        out_dir = output_base / f"{timestamp.year:04d}" / f"{timestamp.month:02d}" / f"{timestamp.day:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{satellite}_modis_bt_cloudmask_{timestamp:%Y%m%dT%H%M}.nc"
        if out_file.exists() and not overwrite:
            LOGGER.debug("Skipping existing processed file %s", out_file)
            continue
        # start/end as ISO strings
        start_time = timestamp.isoformat()
        end_time = (timestamp + dt.timedelta(minutes=5)).isoformat()

        dataset.attrs.update(
            {
                "satellite": satellite,
                "source_l1": l1_file.name,
                "source_l2": l2_file.name if l2_file else "",
                "channels": ",".join(channels),
                "start_time": start_time,
                "end_time": end_time,
            }
        )
        # record grid/resampling metadata
        dataset.attrs["resampled"] = bool(resample)
        if resample:
            dataset.attrs["grid_type"] = "regular_latlon"
            dataset.attrs["target_resolution_deg"] = float(resolution_deg)
            dataset.attrs["roi"] = f"{lon_min},{lat_min},{lon_max},{lat_max}"
        else:
            dataset.attrs["grid_type"] = "l1_native"
        # Robustly sanitize all attributes: convert datetimes and coerce anything
        # else to strings so NetCDF serialization will not fail.
        def _safe_attr(val):
            if isinstance(val, dt.datetime):
                return val.isoformat()
            simple_types = (str, bytes, int, float, bool)
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
        for ch in channels:
            key = f"bt_{ch}"
            if key in dataset.data_vars:
                enc[key] = {"zlib": True, "complevel": 9, "dtype": "float32"}
        if "cloud_mask" in dataset.data_vars:
            enc["cloud_mask"] = {"zlib": True, "complevel": 9, "dtype": "uint8"}
        # coords
        if "latitude" in dataset.coords:
            enc["latitude"] = {"zlib": True, "complevel": 9, "dtype": "float32"}
        if "longitude" in dataset.coords:
            enc["longitude"] = {"zlib": True, "complevel": 9, "dtype": "float32"}

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
                roi=roi,
                resolution_deg=resolution_deg,
                resample=resample,
                overwrite=overwrite,
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
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    config = load_config(Path(args.config))
    run_processing(config=config, dry_run=bool(getattr(args, "dry_run", False)))


if __name__ == "__main__":
    main()
