#!/usr/bin/env python3
"""Profile _regrid_to_target using synthetic data."""
from time import time
import cProfile
import pstats
import numpy as np
import xarray as xr
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("regrid", Path(__file__).parents[2] / "scripts" / "utils" / "regrid.py")
regrid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(regrid)
_regrid_to_target = regrid._regrid_to_target

# Build synthetic source (swath-like) and target regular grid
n_src_y, n_src_x = 500, 500
src_lon = np.linspace(-10, 30, n_src_x)
src_lat = np.linspace(56, 35, n_src_y)
src_lon2d, src_lat2d = np.meshgrid(src_lon, src_lat)

# create a synthetic DataArray with some pattern
vals = np.sin(np.deg2rad(src_lat2d)) + np.cos(np.deg2rad(src_lon2d))
da = xr.DataArray(vals.astype(np.float32), dims=("y","x"))

# target grid
n_tgt_y, n_tgt_x = 600, 600
tgt_lon = np.linspace(-10, 30, n_tgt_x)
tgt_lat = np.linspace(56, 35, n_tgt_y)
tgt_lon2d, tgt_lat2d = np.meshgrid(tgt_lon, tgt_lat)

print("Starting profile: regridding from", vals.shape, "to", tgt_lon2d.shape)

prof_file = "profile_regrid.prof"

cProfile.runctx(
    'out = _regrid_to_target(da, src_lon2d, src_lat2d, tgt_lon2d, tgt_lat2d, method="linear")',
    globals(), locals(), filename=prof_file,
)

print("Wrote", prof_file)
with open(prof_file + ".txt", "w") as fh:
    ps = pstats.Stats(prof_file, stream=fh)
    ps.sort_stats("cumulative").print_stats(30)

print("Profile summary written to", prof_file + ".txt")
