"""Skeleton degree computation."""

from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi
from numpy.typing import NDArray

from .utils.connectivity import degree_footprint


def compute_degree_map(skeleton: NDArray[np.bool_]) -> NDArray[np.int32]:
    fp = degree_footprint(skeleton.ndim)
    conv = ndi.convolve(skeleton.astype(np.uint8), fp, mode="constant", cval=0)
    degree = conv.astype(np.int32)
    degree[~skeleton] = 0
    return degree
