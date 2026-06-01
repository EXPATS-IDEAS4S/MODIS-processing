from __future__ import annotations

from typing import Optional, Tuple, List

import logging
import numpy as np
import xarray as xr

LOGGER = logging.getLogger("process_modis.crop")


def _largest_rectangle_from_mask(valid_mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return the largest all-True rectangle as (row_start, row_stop, col_start, col_stop)."""
    if valid_mask.ndim != 2 or not np.any(valid_mask):
        return None

    heights = np.zeros(valid_mask.shape[1], dtype=np.int64)
    best_area = 0
    best_bounds: Optional[Tuple[int, int, int, int]] = None

    for row_idx in range(valid_mask.shape[0]):
        row = valid_mask[row_idx]
        heights = np.where(row, heights + 1, 0)

        stack: List[int] = []
        extended = np.append(heights, 0)
        for col_idx, height in enumerate(extended):
            while stack and extended[stack[-1]] > height:
                top = stack.pop()
                rect_height = int(extended[top])
                rect_left = stack[-1] + 1 if stack else 0
                rect_right = col_idx
                rect_width = rect_right - rect_left
                area = rect_height * rect_width
                if area > best_area:
                    best_area = area
                    best_bounds = (row_idx - rect_height + 1, row_idx + 1, rect_left, rect_right)
            stack.append(col_idx)

    return best_bounds


def _crop_regular_dataset_to_valid_rectangle(dataset: xr.Dataset) -> xr.Dataset:
    if "latitude" not in dataset.dims or "longitude" not in dataset.dims:
        return dataset

    valid_mask = None
    for name in dataset.data_vars:
        arr = np.asarray(dataset[name].values)
        finite = np.isfinite(arr)
        finite = np.squeeze(finite)
        if finite.ndim != 2:
            continue
        valid_mask = finite if valid_mask is None else (valid_mask & finite)

    if valid_mask is None or not np.any(valid_mask):
        return dataset

    bounds = _largest_rectangle_from_mask(valid_mask)
    if bounds is None:
        return dataset

    row_start, row_stop, col_start, col_stop = bounds

    cropped = dataset.isel(latitude=slice(row_start, row_stop), longitude=slice(col_start, col_stop))
    LOGGER.info(
        "Cropped regular-grid dataset to largest valid rectangle: latitude[%d:%d], longitude[%d:%d]",
        row_start,
        row_stop,
        col_start,
        col_stop,
    )
    # Verify cropped area contains no NaNs across all 2D variables. If any remain
    # (unexpected), fall back to iterative edge-trimming until the rectangle is clean.
    try:
        def _is_clean(ds: xr.Dataset) -> bool:
            for name in ds.data_vars:
                arr = np.asarray(ds[name].values)
                arr = np.squeeze(arr)
                if arr.ndim != 2:
                    continue
                if not np.all(np.isfinite(arr)):
                    return False
            return True

        if not _is_clean(cropped):
            LOGGER.warning("Cropped rectangle still contains NaNs; trimming edges as fallback")
            lat_len = cropped.dims.get("latitude", 0)
            lon_len = cropped.dims.get("longitude", 0)
            r0, r1 = 0, lat_len
            c0, c1 = 0, lon_len
            changed = True
            while changed and r0 < r1 and c0 < c1:
                changed = False
                # build combined finite mask for current window
                combined = None
                for name in cropped.data_vars:
                    arr = np.asarray(cropped[name].values)
                    arr = np.squeeze(arr)
                    if arr.ndim != 2:
                        continue
                    finite = np.isfinite(arr)
                    combined = finite if combined is None else (combined & finite)
                if combined is None:
                    break
                # trim top
                if not np.all(combined[0, :]):
                    r0 += 1
                    combined = combined[1:, :]
                    changed = True
                # trim bottom
                if r0 < r1 and not np.all(combined[-1, :]):
                    r1 -= 1
                    combined = combined[:-1, :]
                    changed = True
                # trim left
                if c0 < c1 and not np.all(combined[:, 0]):
                    c0 += 1
                    combined = combined[:, 1:]
                    changed = True
                # trim right
                if c0 < c1 and not np.all(combined[:, -1]):
                    c1 -= 1
                    combined = combined[:, :-1]
                    changed = True
                # update cropped
                if changed and r0 < r1 and c0 < c1:
                    cropped = cropped.isel(latitude=slice(r0, r1), longitude=slice(c0, c1))

        # final check
        if not _is_clean(cropped):
            LOGGER.error("Unable to produce a NaN-free rectangle after trimming; returning original crop")
        else:
            LOGGER.info("Cropped rectangle verified NaN-free")
    except Exception:
        LOGGER.exception("Exception during cropped-rectangle verification")

    return cropped
