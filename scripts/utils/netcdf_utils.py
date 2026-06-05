from __future__ import annotations

import datetime as dt
from typing import Dict

import numpy as np
import xarray as xr


def safe_attr(val):
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


def prepare_netcdf_encoding(dataset: xr.Dataset, compression_level: int = 4) -> Dict[str, Dict]:
    compression_level = int(compression_level)
    if compression_level < 0 or compression_level > 9:
        raise ValueError("compression_level must be between 0 and 9")

    enc = {}
    for name in dataset.data_vars:
        enc[name] = {"zlib": True, "complevel": compression_level, "dtype": "float32"}
    if "cloud_mask" in dataset.data_vars:
        enc["cloud_mask"] = {"zlib": True, "complevel": compression_level, "dtype": "float32"}
    # coords
    for coord_name in ("latitude", "longitude", "lat", "lon"):
        if coord_name in dataset.coords:
            enc[coord_name] = {"zlib": True, "complevel": compression_level, "dtype": "float32"}
    return enc
