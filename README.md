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
  - Output folder structure is `year/month/day`.

- `scripts/upload_s3.py`
  - Uploads processed NetCDF files to an S3-compatible bucket.
  - Credentials are read from a separate local file (`s3_credentials.py`) that is gitignored.
  - Optional upload verification with `head_object`.

- `scripts/run_pipeline.py`
  - Orchestrates per-year cycles of download -> processing -> upload.
  - Each step can be enabled/disabled.
  - Optional cleanup of raw data after processing and/or processed data after upload.

- `scripts/debug_tools.py`
  - `inspect`: print NetCDF content and variable statistics.
  - `plot`: generate quicklook PNG maps (BT/cloud mask).
  - `verify-s3`: check local NetCDF objects exist in the bucket.

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
python scripts/debug_tools.py inspect --file /path/to/file.nc
```

Generate quicklook plots:

```bash
python scripts/debug_tools.py plot --file /path/to/file.nc --output-dir /path/to/plots
```

Verify local NetCDF files are in bucket:

```bash
python scripts/debug_tools.py verify-s3 --local-base /data/modis/processed --credentials s3_credentials.py --bucket-prefix modis/processed
```

## Notes

- Satpy readers expected by processing script:
  - `modis_l1b` for Terra/Aqua L1 EOS-hdf4 files
  - `modis_l2` for MOD35 cloud-mask products
- Download script uses NASA CMR search API and then downloads returned data links.
- Because product providers can change link metadata, test with `--dry-run` first.
