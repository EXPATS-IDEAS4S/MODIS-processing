# MODIS-processing

Scripts for downloading and processing MODIS Terra/Aqua L1 radiances and L2 cloud mask data for a configured period and ROI.

## What is included

- `scripts/download_modis.py`
  - Downloads Terra/Aqua L1 (`MOD021KM`, `MYD021KM`) and L2 cloud mask (`MOD35_L2`, `MYD35_L2`) from NASA CMR search results.
  - Filters by ROI and time range (example config: April–September 2024, Europe).
  - Saves to `year/month/day` subfolders.
  - L1 Terra + Aqua go to the same radiance base path, cloud mask goes to a separate base path.

- `scripts/process_modis.py`
  - Reads L1 data with Satpy reader `modis_l1b`.
  - Converts selected channels to brightness temperature (BT):
    - `ir_105` -> MODIS band 31
    - `wv_63` -> MODIS band 27
  - Reads L2 cloud mask with Satpy reader `modis_l2` and writes BT + cloud mask into one NetCDF per overpass.
  - Processing options (config-driven):
    - `processing.resample` (bool): when true, resample both L1 BT and L2 cloud mask to a regular latitude/longitude grid covering the configured ROI. When false, use the L1 native grid as the common target and regrid L2 to L1.
    - `processing.target_resolution_deg` (float): grid spacing in degrees when `resample=true` (default `0.01`).
    - `processing.output_base_path`: output folder base for NetCDF files.
  - Output contents and format:
    - The produced NetCDF contains only the two BT variables (one per configured channel), `cloud_mask`, and 2D coordinates `latitude` and `longitude` (dims `y,x`).
    - Missing/outside-swath points are represented as `NaN` on the target grid (cloud_mask stored as float to allow NaN). If you prefer integer mask + fill value, adjust the script.
    - Files are written with NetCDF compression (`zlib=True`, `complevel=9`).
  - Output folder structure is `year/month/day`.

- `scripts/upload_s3.py`
  - Uploads processed NetCDF files to an S3-compatible bucket.
  - Credentials are read from a separate local file (`s3_credentials.py`) that is gitignored.
  - Optional upload verification with `head_object`.

- `scripts/run_pipeline.py`
  - Orchestrates per-year cycles of download -> processing -> upload.
  - Each step can be enabled/disabled.
  - Optional cleanup of raw data after processing and/or processed data after upload.

- `scripts/debug/`
  - `inspect_nc.py`: print NetCDF content and variable statistics.
  - `plot_nc.py`: generate quicklook PNGs for a single file.
  - `plot_nc_day.py`: generate quicklooks for all processed NetCDF files in a day folder.
  - `plot_hdf_day.py`: generate quicklooks for raw L1/L2 granules for a day.
  - `verify_s3.py`: check local NetCDF objects exist in the bucket.
  - `run_debug.py`: launch one or more debug tasks sequentially or in parallel.

The legacy `scripts/debug_tools.py` entry point is still present for backward compatibility, but the new scripts above are the preferred interface.

## Configuration

1. Copy example config:

```bash
cp config/pipeline_config.example.yaml config/pipeline_config.yaml
```

2. Edit paths/period/settings in `config/pipeline_config.yaml`.

Example includes:
- channels: `ir_105`, `wv_63`
- ROI over Europe
- years: `[2024]`
- months: `[4, 5, 6, 7, 8, 9]`

- `processing.resample`: true|false — whether to resample to a regular lat/lon grid (default: true)
- `processing.target_resolution_deg`: grid spacing in degrees when resampling (default: 0.01)
- `processing.resample_chunk_rows`: integer or null — when set, resampling will be performed in row-chunks of this many rows to reduce peak memory usage during griddata regridding. Default: null (disabled).
- `processing.coord_decimals`: integer or null — number of decimal places to round output 1D `latitude`/`longitude` axes so the grid is exactly regular. If null the pipeline derives a sensible value from `target_resolution_deg`.

3. Set Earthdata token:

```bash
export EARTHDATA_TOKEN="<your-token>"
```

4. Configure S3 secrets locally (not in git):

```bash
cp s3_credentials.example.py s3_credentials.py
# then edit s3_credentials.py
```

## Usage

### 1) Download only

```bash
python scripts/download_modis.py --config config/pipeline_config.yaml
```

Dry run:

```bash
python scripts/download_modis.py --config config/pipeline_config.yaml --dry-run
```

### 2) Process only

```bash
python scripts/process_modis.py --config config/pipeline_config.yaml
```

Dry run (process only first available day):

```bash
python scripts/process_modis.py --config config/pipeline_config.yaml --dry-run
```

### 3) Upload only

```bash
python scripts/upload_s3.py --config config/pipeline_config.yaml --credentials s3_credentials.py --verify
```

### 4) Full orchestrated pipeline

```bash
python scripts/run_pipeline.py --config config/pipeline_config.yaml --credentials s3_credentials.py
```

This runs yearly cycles using the `pipeline` toggles in config:
- `run_download`
- `run_processing`
- `run_upload`
- `verify_upload`
- `delete_raw_after_processing`
- `delete_processed_after_upload`

## Debugging helpers

Inspect file content and stats:

```bash
python scripts/debug/inspect_nc.py --file /path/to/file.nc
```

Generate quicklook plots:

```bash
python scripts/debug/plot_nc.py --file /path/to/file.nc --output-dir /path/to/plots
```

Run a group of debug tasks together:

```bash
python scripts/debug/run_debug.py --group file --file /path/to/file.nc --output-dir /path/to/plots --parallel
```

Verify local NetCDF files are in bucket:

```bash
python scripts/debug/verify_s3.py --local-base /data/modis/processed --credentials s3_credentials.py --bucket-prefix modis/processed
```

## Notes

- Satpy readers expected by processing script:
  - `modis_l1b` for Terra/Aqua L1 EOS-hdf4 files
  - `modis_l2` for MOD35 cloud-mask products
- Download script uses NASA CMR search API and then downloads returned data links.
- Because product providers can change link metadata, test with `--dry-run` first.
 
Additional notes on processing behavior and dependencies:
- The processor prefers to use the native geolocation (latitude/longitude) returned by Satpy for each granule. If L1 geolocation is missing for a granule it will currently be skipped and a warning logged; we can enable fallback to L2 geolocation if desired.
- Regridding is implemented using `scipy.interpolate.griddata` as a generic fallback for 2D geolocation. For best performance on swath-to-grid resampling, installing `pyresample` is recommended — the code will attempt to use pyresample where available in the future.
  - When processing large granules the resampling step can be memory intensive. The pipeline supports two mitigations:
    - Chunked regridding: set `processing.resample_chunk_rows` to an integer to process the target grid in row-blocks and reduce peak memory usage.
    - Deterministic coordinates: set `processing.coord_decimals` to force rounding of output 1D coordinates so tools treating the grid as regular do not fail on tiny floating differences.
  For distributed/very large workloads, a `dask`-enabled xarray backend or `pyresample` are recommended for better performance and lower memory pressure.
- The Satpy readers require HDF4 support (`pyhdf`) in the Python environment. Ensure your `satpy` conda env includes `pyhdf` (and `scipy` for resampling).
- NetCDF files are written with compression level 9 to reduce disk and upload bandwidth.
