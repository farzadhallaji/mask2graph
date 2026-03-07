"""Coordinate conversion utilities used across the package."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def index_to_xyz(
    index: tuple[int, ...], spacing: tuple[float, ...], decimals: int | None = None
) -> tuple[float, float, float]:
    if len(index) == 2:
        y, x = index
        sy, sx = spacing
        xyz = (float(x * sx), float(y * sy), 0.0)
    elif len(index) == 3:
        z, y, x = index
        sz, sy, sx = spacing
        xyz = (float(x * sx), float(y * sy), float(z * sz))
    else:
        raise ValueError("index dimensionality must be 2 or 3")
    if decimals is None:
        return xyz
    return tuple(round(v, decimals) for v in xyz)  # type: ignore[return-value]


def indices_to_xyz(
    indices: NDArray[np.integer], spacing: tuple[float, ...], decimals: int | None = None
) -> NDArray[np.float64]:
    idx = np.asarray(indices, dtype=np.int64)
    if idx.ndim != 2:
        raise ValueError("indices must be a 2D array")
    ndim = idx.shape[1]
    out = np.zeros((idx.shape[0], 3), dtype=np.float64)
    if ndim == 2:
        sy, sx = spacing
        out[:, 0] = idx[:, 1] * sx
        out[:, 1] = idx[:, 0] * sy
    elif ndim == 3:
        sz, sy, sx = spacing
        out[:, 0] = idx[:, 2] * sx
        out[:, 1] = idx[:, 1] * sy
        out[:, 2] = idx[:, 0] * sz
    else:
        raise ValueError("indices dimensionality must be 2 or 3")
    if decimals is not None:
        out = np.round(out, decimals=decimals)
    return out
