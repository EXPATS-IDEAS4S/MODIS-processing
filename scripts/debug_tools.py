#!/usr/bin/env python3
"""Debug utilities for MODIS processing outputs.

This script provides a small command-line toolbox for inspecting and plotting
both raw MODIS HDF/HDF5 granules and processed NetCDF outputs.

Available commands:
    - inspect: print a NetCDF dataset summary and basic variable statistics
    - plot: create quicklook plots for a single NetCDF file
    - plot-hdf-day: plot raw L1/L2 granules for one day
    - plot-nc-day: plot processed NetCDF files for one day
    - verify-s3: check whether local files exist in an S3 bucket

Typical usage:
    python scripts/debug_tools.py inspect --file /path/to/file.nc
    python scripts/debug_tools.py plot --file /path/to/file.nc --output-dir /tmp/quicklooks
    python scripts/debug_tools.py plot-nc-day --config config/pipeline_config.yaml
    python scripts/debug_tools.py plot-hdf-day --config config/pipeline_config.yaml

For the day-based plotting commands, the YAML config can provide the base
paths, date, channel selection, and output directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional
import yaml

#import boto3
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from satpy import DataQuery, Scene
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as mpl
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm, ListedColormap

#from upload_s3 import load_s3_credentials


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
    # Grid inspection: determine lon/lat arrays and check regularity
    def _report_grid_info(ds: xr.Dataset):
        # pick a representative data variable with 2D spatial dims
        data_vars_2d = [n for n, d in ds.data_vars.items() if d.squeeze().ndim >= 2]
        lon = lat = None
        if data_vars_2d:
            lon, lat = _nc_geolocation_for_var(ds, ds[data_vars_2d[0]])
        # fallback to coords
        if lon is None or lat is None:
            if "longitude" in ds.coords and "latitude" in ds.coords:
                lon = np.asarray(ds.coords["longitude"].values)
                lat = np.asarray(ds.coords["latitude"].values)

        if lon is None or lat is None:
            print("No geolocation (latitude/longitude) coordinates found in this dataset.")
            return

        # determine if 1D or 2D
        lon_ndim = lon.ndim
        lat_ndim = lat.ndim
        print(f"longitude ndim={lon_ndim}, latitude ndim={lat_ndim}")

        def _corners_from_arrays(lon_arr, lat_arr):
            # lon_arr and lat_arr can be 1D or 2D; extract four corners as (lon, lat)
            if lon_arr.ndim == 1 and lat_arr.ndim == 1:
                lon_min = float(np.nanmin(lon_arr))
                lon_max = float(np.nanmax(lon_arr))
                lat_min = float(np.nanmin(lat_arr))
                lat_max = float(np.nanmax(lat_arr))
                return (lon_min, lat_max), (lon_max, lat_max), (lon_min, lat_min), (lon_max, lat_min)
            else:
                # assume matching 2D shapes
                return (float(lon_arr[0, 0]), float(lat_arr[0, 0])), (float(lon_arr[0, -1]), float(lat_arr[0, -1])), (float(lon_arr[-1, 0]), float(lat_arr[-1, 0])), (float(lon_arr[-1, -1]), float(lat_arr[-1, -1]))

        corners = _corners_from_arrays(lon, lat)
        print("Corners (TL, TR, BL, BR) as (lon, lat):")
        for c in corners:
            print(f"  {c}")

        # check regularity
        def _is_regular_1d(arr):
            if arr.ndim != 1 or arr.size < 2:
                return True, None
            diffs = np.diff(arr)
            # allow small numerical tolerance
            ok = np.allclose(diffs, diffs[0], rtol=1e-6, atol=1e-9)
            return bool(ok), float(diffs[0])

        def _is_regular_2d(lon2d, lat2d):
            # check if lon varies only along axis 1 and lat only along axis 0
            lon_var_x = np.allclose(lon2d - lon2d[:, 0:1], np.tile(lon2d[0, :], (lon2d.shape[0], 1)), atol=1e-6)
            lat_var_y = np.allclose(lat2d - lat2d[0:1, :], np.tile(lat2d[:, 0], (lat2d.shape[1], 1)).T, atol=1e-6)
            if lon_var_x and lat_var_y:
                # derive 1D axes
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
    except Exception as e:
        print(f"Grid inspection failed: {e}")


def plot_file(path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ds = xr.open_dataset(path)

    for name in ds.data_vars:
        data = ds[name]
        if data.ndim < 2:
            continue
        plt.figure(figsize=(8, 6))
        plt.title(name)
        plt.imshow(data.squeeze().values, cmap="viridis")
        plt.colorbar(label=str(data.attrs.get("units", "")))
        out_file = output_dir / f"{path.stem}_{name}.png"
        plt.tight_layout()
        plt.savefig(out_file, dpi=120)
        plt.close()
        print(f"Saved {out_file}")


def _nc_geolocation_for_var(ds: xr.Dataset, data: xr.DataArray) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if "longitude" in data.coords and "latitude" in data.coords:
        lon = np.asarray(data.coords["longitude"].values)
        lat = np.asarray(data.coords["latitude"].values)
        if lon.ndim == 1 and lat.ndim == 1:
            lon2d, lat2d = np.meshgrid(lon, lat)
            return lon2d, lat2d
        return lon, lat

    if "longitude" in ds.coords and "latitude" in ds.coords:
        lon = np.asarray(ds.coords["longitude"].values)
        lat = np.asarray(ds.coords["latitude"].values)
        if lon.ndim == 1 and lat.ndim == 1:
            lon2d, lat2d = np.meshgrid(lon, lat)
            return lon2d, lat2d
        return lon, lat

    return None, None


def _plot_nc_panel(ax, data: xr.DataArray, name: str):
    values = np.asarray(data.squeeze().values)
    lon, lat = _nc_geolocation_for_var(data.to_dataset(name="tmp"), data)

    if name == "cloud_mask":
        cmap = "gray"
        vmin = 0.0
        vmax = 1.0
    elif name.startswith("bt_") or str(data.attrs.get("units", "")).upper() == "K":
        cmap = "gray_r"
        vmin = 200.0
        vmax = 300.0
    else:
        cmap = "viridis"
        vmin = None
        vmax = None

    if lon is not None and lat is not None:
        mesh = ax.pcolormesh(lon, lat, values, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        mesh = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.coastlines(resolution="110m", color="yellow", linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, edgecolor="yellow", linewidth=0.5)
    ax.set_title(name, fontsize=10)
    return mesh


def plot_nc_file(path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ds = xr.open_dataset(path)

    data_vars = [name for name, data in ds.data_vars.items() if data.squeeze().ndim >= 2]
    if not data_vars:
        print(f"No 2D variables found in {path}")
        return

    fig, axes = plt.subplots(
        1,
        len(data_vars),
        figsize=(6.5 * len(data_vars), 5.5),
        subplot_kw={"projection": ccrs.PlateCarree()},
        squeeze=False,
    )
    axes = list(axes[0])

    for index, (ax, name) in enumerate(zip(axes, data_vars)):
        mesh = _plot_nc_panel(ax, ds[name], name)
        gridlines = ax.gridlines(draw_labels=True, linewidth=0.35, color="white", alpha=0.35, linestyle="--")
        gridlines.top_labels = False
        gridlines.right_labels = False
        gridlines.bottom_labels = True
        gridlines.left_labels = index == 0
        gridlines.xformatter = LONGITUDE_FORMATTER
        gridlines.yformatter = LATITUDE_FORMATTER
        gridlines.xlabel_style = {"size": 8}
        gridlines.ylabel_style = {"size": 8}
        try:
            fig.colorbar(mesh, ax=ax, orientation="vertical", fraction=0.035, pad=0.02)
        except Exception:
            pass

    fig.suptitle(path.name, fontsize=12, y=0.98)
    fig.subplots_adjust(left=0.03, right=0.985, bottom=0.08, top=0.90, wspace=0.08, hspace=0.02)
    out_file = output_dir / f"{path.stem}_multiplot.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    mpl.close(fig)
    print(f"Saved {out_file}")


def plot_nc_day(processed_base: Path, date_str: str, output_base: Path) -> None:
    import datetime as dt

    try:
        if "." in date_str:
            day = dt.datetime.strptime(date_str, "%d.%m.%Y")
        else:
            day = dt.datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        raise ValueError("date must be YYYY-MM-DD or DD.MM.YYYY")

    day_dir = processed_base / f"{day.year:04d}" / f"{day.month:02d}" / f"{day.day:02d}"
    if not day_dir.exists():
        print(f"Processed day directory not found: {day_dir}")
        return

    nc_files = sorted([p for p in day_dir.glob("*.nc") if p.is_file()])
    if not nc_files:
        print(f"No NetCDF files found in {day_dir}")
        return

    out_day_dir = output_base / f"{day.year:04d}" / f"{day.month:02d}" / f"{day.day:02d}"
    out_day_dir.mkdir(parents=True, exist_ok=True)

    for nc_file in nc_files:
        try:
            plot_nc_file(nc_file, out_day_dir)
        except Exception as e:
            print(f"Failed to plot {nc_file}: {e}")


def _extract_granule_key(path: Path) -> str:
    parts = path.stem.split(".")
    return "_".join(parts[1:3]) if len(parts) >= 3 else path.stem


def _resolve_cloud_mask_name(available: List[str], preferred_name: Optional[str] = None) -> Optional[str]:
    preferred_names = []
    if preferred_name:
        preferred_names.extend([preferred_name, preferred_name.lower(), preferred_name.upper()])
    preferred_names.extend(["cloud_mask", "Cloud_Mask", "cloud_mask_byte_segment", "Integer_Cloud_Mask"])

    for candidate in preferred_names:
        if candidate in available:
            return candidate
    return next((name for name in available if "cloud" in name.lower()), None)


def _resolve_geo_name(available: List[str], axis: str) -> Optional[str]:
    preferred = axis.lower()
    exact_matches = [preferred, preferred.upper(), preferred.capitalize()]
    for candidate in exact_matches:
        if candidate in available:
            return candidate

    for name in available:
        if name.lower() == preferred:
            return name

    return next((name for name in available if axis in name.lower()), None)


def _cloud_mask_values(values: np.ndarray, binary: bool) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if binary:
        mapped = np.full(arr.shape, np.nan, dtype=np.float32)
        mapped[np.isin(arr, [0, 1])] = 1
        mapped[np.isin(arr, [2, 3])] = 0
        mapped[arr == -1] = np.nan
        return mapped
    return arr


def _cloud_mask_plot_style(binary: bool):
    if binary:
        return {
            "cmap": "gray",
            "vmin": 0.0,
            "vmax": 1.0,
            "ticks": [0.0, 1.0],
            "ticklabels": ["clear", "cloud"],
            "label": "cloud mask binary",
        }

    cmap = ListedColormap(["#6e6e6e", "#2b6cb0", "#f59e0b", "#22c55e", "#ef4444"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    return {
        "cmap": cmap,
        "norm": norm,
        "ticks": [-1, 0, 1, 2, 3],
        "ticklabels": ["no result", "cloudy", "prob. cloudy", "prob. clear", "clear"],
        "label": "cloud mask classes",
    }


def plot_hdf_day(
    radiance_base: Path,
    cloud_base: Path,
    date_str: str,
    channels: List[str],
    output_base: Path,
    l1_reader: str = "modis_l1b",
    l2_reader: str = "modis_l2",
    cloud_mask_channel: Optional[str] = None,
    cloud_mask_binary: bool = True,
) -> None:
    """Plot MODIS HDF files for a single day (YYYY-MM-DD or DD.MM.YYYY).

    Saves radiance channel quicklooks and cloud masks as PNGs under `output_base/YYYY/MM/DD/...`.
    """
    # parse date
    import datetime as dt

    try:
        if "." in date_str:
            day = dt.datetime.strptime(date_str, "%d.%m.%Y")
        else:
            day = dt.datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        raise ValueError("date must be YYYY-MM-DD or DD.MM.YYYY")

    day_dir = radiance_base / f"{day.year:04d}" / f"{day.month:02d}" / f"{day.day:02d}"
    cloud_dir = cloud_base / f"{day.year:04d}" / f"{day.month:02d}" / f"{day.day:02d}"
    if not day_dir.exists():
        print(f"Radiance day directory not found: {day_dir}")
        return

    l1_files = sorted([p for p in day_dir.glob("*") if p.is_file() and (p.suffix.lower() in {".hdf", ".h5"})])
    l2_files = sorted([p for p in cloud_dir.glob("*") if p.is_file() and (p.suffix.lower() in {".hdf", ".h5"})])
    l2_index = {_extract_granule_key(p): p for p in l2_files}

    out_day_dir = output_base / f"{day.year:04d}" / f"{day.month:02d}" / f"{day.day:02d}"
    out_day_dir.mkdir(parents=True, exist_ok=True)

    for l1 in l1_files:
        granule = _extract_granule_key(l1)
        matched_l2 = l2_index.get(granule)
        try:
            scene = Scene(reader=l1_reader, filenames=[str(l1)])
            # prefer the two channels closest to 10.8 and 6.2 (default mapping ir_105, wv_63)
            band_map = {"ir_105": "31", "wv_63": "27"}
            ch1_name = channels[0] if channels else "ir_105"
            ch2_name = channels[1] if len(channels) > 1 else "wv_63"
            ch1_key = band_map.get(ch1_name, "31")
            ch2_key = band_map.get(ch2_name, "27")

            queries = [DataQuery(name=ch1_key, calibration="brightness_temperature"), DataQuery(name=ch2_key, calibration="brightness_temperature")]
            scene.load(queries)
        except Exception as e:
            print(f"Failed to load L1 {l1}: {e}")
            continue

        # load BT arrays
        #da1 = scene[queries[0]]
        da1 = scene[ch1_key].values
        
        #da2 = scene[queries[1]]
        da2 = scene[ch2_key].values    

        # extract L1 lat/lon directly from area metadata (same approach as cloud mask)
        ch1_lon, ch1_lat = scene[ch1_key].attrs["area"].get_lonlats()
        ch2_lon, ch2_lat = scene[ch2_key].attrs["area"].get_lonlats()
            

        # load cloud mask if matched
        cmask = None
        cloud_lat = None
        cloud_lon = None
        if matched_l2 is not None:
            try:
                #print(f"Matched L2 file for {l1.name}: {matched_l2.name}")
                scene2 = Scene(reader=l2_reader, filenames=[str(matched_l2)])
                avail2 = list(scene2.available_dataset_names())
                #print(avail2)
                cloud_name = _resolve_cloud_mask_name(avail2, preferred_name=cloud_mask_channel)
                if cloud_name:
                    #try:
                    scene2.load([DataQuery(name=cloud_name, resolution=1000)])
                    #cloud_da = scene2[DataQuery(name=cloud_name, resolution=1000)].astype(float)
                    cmask = scene2[cloud_name].values
                    #print(cmask) #plot unique values
                    #print(np.unique(cmask))
                    cloud_lon, cloud_lat = scene2[cloud_name].attrs["area"].get_lonlats()
                    cloud_lon = np.asarray(cloud_lon)
                    cloud_lat = np.asarray(cloud_lat)
                    #print(cmask.shape)
                    #print(cloud_lat.shape, cloud_lon.shape)
                    
            except Exception as e:
                print(f"Failed to load L2 {matched_l2}: {e}")

        cloud_style = _cloud_mask_plot_style(cloud_mask_binary)
        cloud_values = None if cmask is None else _cloud_mask_values(cmask, cloud_mask_binary)
        bt_vmin = 200.0
        bt_vmax = 300.0
        # create multiplot: 1x3 (BT1 with overlay, BT2 with overlay, cloud-only)
        fig = mpl.figure(figsize=(18, 6))
        axes = [fig.add_subplot(1, 3, i + 1, projection=ccrs.PlateCarree()) for i in range(3)]
        for ax in axes:
            ax.coastlines(resolution="110m", color="yellow", linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, edgecolor="yellow", linewidth=0.5)

        def _plot_on_ax(ax, lon_arr, lat_arr, data_arr, cmap="gray_r", vmin=None, vmax=None):
            try:
                pcm = ax.pcolormesh(lon_arr, lat_arr, data_arr, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax)
                return pcm
            except Exception:
                im = ax.imshow(data_arr, cmap=cmap)
                return im

        # BT1
        if da1 is not None:
            pcm1 = None
            pcm1 = _plot_on_ax(axes[0], ch1_lon, ch2_lat, da1, cmap="gray_r", vmin=bt_vmin, vmax=bt_vmax)
            axes[0].set_title(f"BT {ch1_name}", fontsize=10)
            try:
                fig.colorbar(pcm1, ax=axes[0], orientation="vertical", fraction=0.035, pad=0.02)
            except Exception:
                pass
            # overlay cloud mask
            if cloud_values is not None and np.isfinite(cloud_values).any():
                overlay = cloud_values
                if cloud_lat is not None and cloud_lon is not None:
                    overlay_mappable = axes[0].pcolormesh(
                        cloud_lon,
                        cloud_lat,
                        overlay,
                        transform=ccrs.PlateCarree(),
                        cmap=cloud_style["cmap"],
                        norm=cloud_style.get("norm"),
                        vmin=cloud_style.get("vmin"),
                        vmax=cloud_style.get("vmax"),
                        alpha=0.45,
                    )
                else:
                    overlay_mappable = axes[0].imshow(overlay, cmap=cloud_style["cmap"], norm=cloud_style.get("norm"), vmin=cloud_style.get("vmin"), vmax=cloud_style.get("vmax"), alpha=0.45)

        # BT2
        if da2 is not None:
            pcm2 = None
            pcm2 = _plot_on_ax(axes[1], ch2_lon, ch2_lat, da2, cmap="gray_r", vmin=bt_vmin, vmax=bt_vmax)
            axes[1].set_title(f"BT {ch2_name}", fontsize=10)
            try:
                fig.colorbar(pcm2, ax=axes[1], orientation="vertical", fraction=0.035, pad=0.02)
            except Exception:
                pass
            if cloud_values is not None and np.isfinite(cloud_values).any():
                if cloud_lat is not None and cloud_lon is not None:
                    axes[1].pcolormesh(
                        cloud_lon,
                        cloud_lat,
                        cloud_values,
                        transform=ccrs.PlateCarree(),
                        cmap=cloud_style["cmap"],
                        norm=cloud_style.get("norm"),
                        vmin=cloud_style.get("vmin"),
                        vmax=cloud_style.get("vmax"),
                        alpha=0.45,
                    )
                else:
                    axes[1].imshow(cloud_values, cmap=cloud_style["cmap"], norm=cloud_style.get("norm"), vmin=cloud_style.get("vmin"), vmax=cloud_style.get("vmax"), alpha=0.45)

        # cloud-only
        if cloud_values is not None:
            if cloud_lat is not None and cloud_lon is not None:
                pcm3 = axes[2].pcolormesh(
                    cloud_lon,
                    cloud_lat,
                    cloud_values,
                    transform=ccrs.PlateCarree(),
                    cmap=cloud_style["cmap"],
                    norm=cloud_style.get("norm"),
                    vmin=cloud_style.get("vmin"),
                    vmax=cloud_style.get("vmax"),
                )
            else:
                pcm3 = axes[2].imshow(cloud_values, cmap=cloud_style["cmap"], norm=cloud_style.get("norm"), vmin=cloud_style.get("vmin"), vmax=cloud_style.get("vmax"))
            try:
                fig.colorbar(pcm3, ax=axes[2], orientation="vertical", fraction=0.035, pad=0.02, ticks=cloud_style["ticks"])
                pcm3.colorbar.ax.set_yticklabels(cloud_style["ticklabels"])
            except Exception:
                pass
        axes[2].set_title("Cloud mask", fontsize=10)

        out_file = out_day_dir / f"{l1.stem}_multiplot.png"
        fig.suptitle(l1.name, fontsize=12, y=0.965)
        fig.subplots_adjust(left=0.03, right=0.98, bottom=0.05, top=0.93, wspace=0.10)
        fig.savefig(out_file, dpi=150, bbox_inches="tight")
        mpl.close(fig)
        print(f"Saved {out_file}")


def verify_s3_uploads(base_path: Path, credentials_path: Path, bucket_prefix: str = "") -> None:
    creds = load_s3_credentials(credentials_path)
    s3 = boto3.client(
        "s3",
        endpoint_url=creds["endpoint"],
        aws_access_key_id=creds["access_key"],
        aws_secret_access_key=creds["secret_key"],
    )

    missing = []
    prefix = bucket_prefix.strip("/")
    for file_path in sorted(base_path.rglob("*.nc")):
        rel = file_path.relative_to(base_path).as_posix()
        key = f"{prefix}/{rel}" if prefix else rel
        try:
            s3.head_object(Bucket=creds["bucket"], Key=key)
        except Exception:
            missing.append(key)

    print(json.dumps({"checked": len(list(base_path.rglob('*.nc'))), "missing": missing}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug helpers for MODIS processing")
    sub = parser.add_subparsers(dest="cmd", required=True)

    inspect_cmd = sub.add_parser("inspect", help="Print content/stats for NetCDF file")
    inspect_cmd.add_argument("--file", required=True)

    plot_cmd = sub.add_parser("plot", help="Generate quicklook maps for NetCDF variables")
    plot_cmd.add_argument("--file", required=True)
    plot_cmd.add_argument("--output-dir", required=True)

    plot_nc_cmd = sub.add_parser("plot-nc-day", help="Plot processed NetCDF quicklooks for a single day")
    plot_nc_cmd.add_argument("--processed-base", required=False, help="Base path where processed day folders are stored")
    plot_nc_cmd.add_argument("--date", required=False, help="Date to plot (YYYY-MM-DD or DD.MM.YYYY)")
    plot_nc_cmd.add_argument("--output-dir", required=False, help="Output base dir for quicklooks (default: processed_base/quicklooks)")
    plot_nc_cmd.add_argument("--config", required=False, help="YAML config file with parameters (overrides CLI args)")

    plot_hdf_cmd = sub.add_parser("plot-hdf-day", help="Plot MODIS HDF radiances and cloud masks for a single day")
    plot_hdf_cmd.add_argument("--radiance-base", required=False, help="Base path where radiance day folders are stored")
    plot_hdf_cmd.add_argument("--cloud-base", required=False, help="Base path where cloud mask day folders are stored")
    plot_hdf_cmd.add_argument("--date", required=False, help="Date to plot (YYYY-MM-DD or DD.MM.YYYY)")
    plot_hdf_cmd.add_argument("--channels", default="ir_105,wv_63", help="Comma-separated channel keys (e.g. ir_105,wv_63)")
    plot_hdf_cmd.add_argument("--output-dir", required=False, help="Output base dir for quicklooks (default: radiance_base.parent/quicklooks)")
    plot_hdf_cmd.add_argument("--save-to-raw", action="store_true", help="Save quicklooks under the radiance base parent path (radiance_base.parent/quicklooks)")
    plot_hdf_cmd.add_argument("--config", required=False, help="YAML config file with parameters (overrides CLI args)")

    verify_cmd = sub.add_parser("verify-s3", help="Verify local files exist in S3 bucket")
    verify_cmd.add_argument("--local-base", required=True)
    verify_cmd.add_argument("--credentials", default="s3_credentials.py")
    verify_cmd.add_argument("--bucket-prefix", default="")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "inspect":
        inspect_file(Path(args.file))
    elif args.cmd == "plot":
        path = Path(args.file)
        if path.suffix.lower() == ".nc":
            plot_nc_file(path, Path(args.output_dir))
        else:
            plot_file(path, Path(args.output_dir))
    elif args.cmd == "verify-s3":
        verify_s3_uploads(Path(args.local_base), Path(args.credentials), args.bucket_prefix)
    elif args.cmd == "plot-nc-day":
        if args.config:
            cfg = yaml.safe_load(Path(args.config).read_text())
            if isinstance(cfg, dict) and "processing" in cfg:
                processing_cfg = cfg.get("processing", {})
                q = processing_cfg.get("quicklooks", {})
                processed_base = processing_cfg.get("output_base_path")
                date = q.get("date") or args.date
                out_dir_cfg = q.get("base_path") or args.output_dir
            else:
                processed_base = args.processed_base
                date = args.date
                out_dir_cfg = args.output_dir
        else:
            processed_base = args.processed_base
            date = args.date
            out_dir_cfg = args.output_dir

        if processed_base is None or date is None:
            if processed_base is not None and (date is None or str(date).strip() == ""):
                pb = Path(processed_base)
                candidates = sorted([p for p in pb.glob("*/*/*") if p.is_dir()])
                if candidates:
                    first = candidates[0]
                    day = first.name
                    month = first.parent.name
                    year = first.parent.parent.name
                    date = f"{year}-{month}-{day}"
                    print(f"Auto-detected date {date} from {first}")
                else:
                    raise SystemExit("processed_base and date must be provided either via CLI or config file")
            else:
                raise SystemExit("processed_base and date must be provided either via CLI or config file")

        out_base = Path(out_dir_cfg) if out_dir_cfg else Path(processed_base) / "quicklooks"
        plot_nc_day(Path(processed_base), date, out_base)
    elif args.cmd == "plot-hdf-day":
        # prefer config file if provided
        if args.config:
            cfg = yaml.safe_load(Path(args.config).read_text())
            # support either a simple quicklook config or the full pipeline_config.yaml
            if isinstance(cfg, dict) and "download" in cfg:
                # pipeline_config.yaml structure
                radiance_base = cfg.get("download", {}).get("radiance_base_path") or cfg.get("download", {}).get("radiance_base")
                cloud_base = cfg.get("download", {}).get("cloud_mask_base_path") or cfg.get("download", {}).get("cloud_base")
                # prefer a top-level `quicklooks` section; fallback to `processing.quicklooks`
                q = cfg.get("quicklooks") or cfg.get("processing", {}).get("quicklooks", {})
                processing_cfg = cfg.get("processing", {})
                date = q.get("date") or cfg.get("date")
                channels = q.get("channels") or cfg.get("channels") or [c.strip() for c in args.channels.split(",") if c.strip()]
                save_to_raw = bool(q.get("save_to_raw", cfg.get("save_to_raw", False)))
                out_dir_cfg = q.get("base_path") or cfg.get("output_dir")
                l1_reader = str(processing_cfg.get("l1_reader", "modis_l1b"))
                l2_reader = str(processing_cfg.get("l2_reader", "modis_l2"))
                cloud_mask_channel = processing_cfg.get("cloud_mask_channel")
                cloud_mask_binary = bool(processing_cfg.get("cloud_mask_binary", True))
            else:
                radiance_base = cfg.get("radiance_base")
                cloud_base = cfg.get("cloud_base")
                date = cfg.get("date")
                channels = cfg.get("channels", []) or [c.strip() for c in args.channels.split(",") if c.strip()]
                save_to_raw = bool(cfg.get("save_to_raw", False))
                out_dir_cfg = cfg.get("output_dir")
                l1_reader = str(cfg.get("l1_reader", "modis_l1b"))
                l2_reader = str(cfg.get("l2_reader", "modis_l2"))
                cloud_mask_channel = cfg.get("cloud_mask_channel")
                cloud_mask_binary = bool(cfg.get("cloud_mask_binary", True))
        else:
            radiance_base = args.radiance_base
            cloud_base = args.cloud_base
            date = args.date
            channels = [c.strip() for c in args.channels.split(",") if c.strip()]
            save_to_raw = bool(args.save_to_raw)
            out_dir_cfg = args.output_dir
            l1_reader = "modis_l1b"
            l2_reader = "modis_l2"
            cloud_mask_channel = None
            cloud_mask_binary = True

        if radiance_base is None or cloud_base is None or date is None:
                # try to auto-detect a date from radiance_base if date is missing
                if radiance_base is not None and (date is None or str(date).strip() == ""):
                    rb = Path(radiance_base)
                    # look for first day folder YYYY/MM/DD
                    candidates = sorted([p for p in rb.glob("*/*/*") if p.is_dir()])
                    if candidates:
                        first = candidates[0]
                        day = first.name
                        month = first.parent.name
                        year = first.parent.parent.name
                        date = f"{year}-{month}-{day}"
                        print(f"Auto-detected date {date} from {first}")
                    else:
                        raise SystemExit("radiance_base, cloud_base and date must be provided either via CLI or config file")
                else:
                    raise SystemExit("radiance_base, cloud_base and date must be provided either via CLI or config file")

        if save_to_raw:
            out_base = Path(radiance_base).parent / "quicklooks"
        else:
            out_base = Path(out_dir_cfg) if out_dir_cfg else Path(radiance_base).parent / "quicklooks"

        plot_hdf_day(
            Path(radiance_base),
            Path(cloud_base),
            date,
            channels,
            out_base,
            l1_reader=l1_reader,
            l2_reader=l2_reader,
            cloud_mask_channel=cloud_mask_channel,
            cloud_mask_binary=cloud_mask_binary,
        )


if __name__ == "__main__":
    main()
