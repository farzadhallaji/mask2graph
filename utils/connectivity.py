"""Connectivity and neighborhood helpers."""

from __future__ import annotations

from itertools import product

import numpy as np
import scipy.ndimage as ndi
from numpy.typing import NDArray


def full_structure(ndim: int) -> NDArray[np.int8]:
    if ndim not in (2, 3):
        raise ValueError("ndim must be 2 or 3")
    return np.ones((3,) * ndim, dtype=np.int8)


def degree_footprint(ndim: int) -> NDArray[np.int8]:
    fp = full_structure(ndim).copy()
    fp[(1,) * ndim] = 0
    return fp


def neighbor_offsets(ndim: int) -> list[tuple[int, ...]]:
    if ndim not in (2, 3):
        raise ValueError("ndim must be 2 or 3")
    offsets = []
    for delta in product((-1, 0, 1), repeat=ndim):
        if any(d != 0 for d in delta):
            offsets.append(delta)
    return offsets


def in_bounds(index: tuple[int, ...], shape: tuple[int, ...]) -> bool:
    return all(0 <= i < s for i, s in zip(index, shape))


def iter_neighbors(index: tuple[int, ...], shape: tuple[int, ...]) -> list[tuple[int, ...]]:
    offsets = neighbor_offsets(len(index))
    out: list[tuple[int, ...]] = []
    for delta in offsets:
        n = tuple(i + d for i, d in zip(index, delta))
        if in_bounds(n, shape):
            out.append(n)
    return out


def label_components(mask: NDArray[np.bool_]) -> tuple[NDArray[np.int32], int]:
    labels, n = ndi.label(mask, structure=full_structure(mask.ndim))
    return labels.astype(np.int32, copy=False), int(n)
