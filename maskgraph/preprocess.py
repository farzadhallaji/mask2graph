"""Input preprocessing for binary masks."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from skimage.morphology import remove_small_holes, remove_small_objects

from .config import CleanupConfig


def preprocess_mask(mask: NDArray[np.bool_], cleanup: CleanupConfig) -> NDArray[np.bool_]:
    out = np.asarray(mask, dtype=np.bool_).copy()
    ndim = out.ndim
    connectivity = cleanup.connectivity if cleanup.connectivity is not None else ndim
    if cleanup.remove_objects_max_size is not None:
        max_size = int(cleanup.remove_objects_max_size)
        try:
            out = remove_small_objects(out.astype(bool, copy=False), max_size=max_size, connectivity=connectivity)
        except TypeError:
            # Backward-compatible fallback where only min_size is available.
            out = remove_small_objects(
                out.astype(bool, copy=False),
                min_size=max_size + 1,
                connectivity=connectivity,
            )
    if cleanup.fill_holes_max_size is not None:
        out = remove_small_holes(
            out.astype(bool, copy=False),
            area_threshold=int(cleanup.fill_holes_max_size),
            connectivity=connectivity,
        )
    return np.asarray(out, dtype=np.bool_)
