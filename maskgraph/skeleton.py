"""Skeletonization stage."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from skimage.morphology import skeletonize

from .config import SkeletonConfig


def skeletonize_mask(mask: NDArray[np.bool_], config: SkeletonConfig) -> NDArray[np.bool_]:
    if mask.ndim == 2:
        skel = skeletonize(mask.astype(bool, copy=False), method=config.method_2d)
    elif mask.ndim == 3:
        skel = skeletonize(mask.astype(bool, copy=False), method=config.method_3d)
    else:
        raise ValueError("mask must be 2D or 3D")
    return np.asarray(skel, dtype=np.bool_)
