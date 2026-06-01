from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml

SATELLITE_PREFIX = {"terra": "MOD", "aqua": "MYD"}


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


def list_files(radiance_day_dir: Path, cloud_day_dir: Path, satellite: Optional[str]) -> Tuple[List[Path], List[Path]]:
    """List L1 and L2 files for a day. If `satellite` is None, include both terra and aqua."""
    if satellite is None:
        l1_files = sorted([item for item in radiance_day_dir.glob("*") if item.is_file() and any(_is_l1_file(item, s) for s in SATELLITE_PREFIX)])
        l2_files = sorted([item for item in cloud_day_dir.glob("*") if item.is_file() and any(_is_l2_file(item, s) for s in SATELLITE_PREFIX)])
    else:
        l1_files = sorted([item for item in radiance_day_dir.glob("*") if item.is_file() and _is_l1_file(item, satellite)])
        l2_files = sorted([item for item in cloud_day_dir.glob("*") if item.is_file() and _is_l2_file(item, satellite)])
    return l1_files, l2_files


def _extract_granule_key(path: Path) -> str:
    parts = path.stem.split(".")
    return "_".join(parts[1:3]) if len(parts) >= 3 else path.stem


def _index_l2_files(l2_files: Iterable[Path]) -> Dict[str, Path]:
    return {_extract_granule_key(path): path for path in l2_files}


def _satellite_from_name(path: Path) -> Optional[str]:
    name = path.name
    for sat, prefix in SATELLITE_PREFIX.items():
        if name.startswith(prefix):
            return sat
    return None
