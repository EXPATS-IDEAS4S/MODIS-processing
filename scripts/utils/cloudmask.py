from __future__ import annotations

from typing import List, Optional
import numpy as np


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


def _cloud_mask_to_binary_values(values: np.ndarray, binary: bool) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if not binary:
        return arr
    mapped = np.full(arr.shape, np.nan, dtype=np.float32)
    mapped[np.isin(arr, [0])] = 1.0
    mapped[np.isin(arr, [1, 2, 3])] = 0.0
    mapped[arr == -1] = np.nan
    return mapped
