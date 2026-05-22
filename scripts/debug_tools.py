#!/usr/bin/env python3
"""Debug utilities: inspect NetCDF files, plot quicklooks, and verify S3 uploads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import boto3
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from upload_s3 import load_s3_credentials


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
        plot_file(Path(args.file), Path(args.output_dir))
    elif args.cmd == "verify-s3":
        verify_s3_uploads(Path(args.local_base), Path(args.credentials), args.bucket_prefix)


if __name__ == "__main__":
    main()
