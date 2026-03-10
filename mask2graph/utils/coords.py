"""Coordinate conversion utilities used across the package."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _check_spacing_for_ndim(spacing: tuple[float, ...], ndim: int) -> None:
    if len(spacing) != ndim:
        raise ValueError(f"spacing ndim mismatch: got {len(spacing)} expected {ndim}")


def index_to_xyz(
    index: tuple[int, ...], spacing: tuple[float, ...], decimals: int | None = None
) -> tuple[float, float, float]:
    if len(index) == 2:
        _check_spacing_for_ndim(spacing, 2)
        y, x = index
        sy, sx = spacing
        xyz = (float(x * sx), float(y * sy), 0.0)
    elif len(index) == 3:
        _check_spacing_for_ndim(spacing, 3)
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
        _check_spacing_for_ndim(spacing, 2)
        sy, sx = spacing
        out[:, 0] = idx[:, 1] * sx
        out[:, 1] = idx[:, 0] * sy
    elif ndim == 3:
        _check_spacing_for_ndim(spacing, 3)
        sz, sy, sx = spacing
        out[:, 0] = idx[:, 2] * sx
        out[:, 1] = idx[:, 1] * sy
        out[:, 2] = idx[:, 0] * sz
    else:
        raise ValueError("indices dimensionality must be 2 or 3")
    if decimals is not None:
        out = np.round(out, decimals=decimals)
    return out


def xyz_to_index(
    xyz: tuple[float, float, float] | NDArray[np.floating],
    spacing: tuple[float, ...],
    ndim: int,
) -> tuple[int, ...]:
    point = np.asarray(xyz, dtype=np.float64).reshape(-1)
    if ndim == 2:
        _check_spacing_for_ndim(spacing, 2)
        sy, sx = spacing
        if point.size < 2:
            raise ValueError("xyz point must have at least 2 coordinates")
        return (int(round(float(point[1]) / max(sy, 1e-6))), int(round(float(point[0]) / max(sx, 1e-6))))
    if ndim == 3:
        _check_spacing_for_ndim(spacing, 3)
        sz, sy, sx = spacing
        if point.size < 3:
            raise ValueError("xyz point must have at least 3 coordinates")
        return (
            int(round(float(point[2]) / max(sz, 1e-6))),
            int(round(float(point[1]) / max(sy, 1e-6))),
            int(round(float(point[0]) / max(sx, 1e-6))),
        )
    raise ValueError("index dimensionality must be 2 or 3")


def xyzs_to_indices(
    xyzs: NDArray[np.floating],
    spacing: tuple[float, ...],
    ndim: int,
) -> NDArray[np.int32]:
    points = np.asarray(xyzs, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError("xyzs must be a 2D array")
    if ndim == 2:
        _check_spacing_for_ndim(spacing, 2)
        if points.shape[1] < 2:
            raise ValueError("2D xyzs must have at least 2 columns")
        sy, sx = spacing
        out = np.zeros((points.shape[0], 2), dtype=np.int32)
        out[:, 0] = np.round(points[:, 1] / max(sy, 1e-6)).astype(np.int32)
        out[:, 1] = np.round(points[:, 0] / max(sx, 1e-6)).astype(np.int32)
        return out
    if ndim == 3:
        _check_spacing_for_ndim(spacing, 3)
        if points.shape[1] < 3:
            raise ValueError("3D xyzs must have at least 3 columns")
        sz, sy, sx = spacing
        out = np.zeros((points.shape[0], 3), dtype=np.int32)
        out[:, 0] = np.round(points[:, 2] / max(sz, 1e-6)).astype(np.int32)
        out[:, 1] = np.round(points[:, 1] / max(sy, 1e-6)).astype(np.int32)
        out[:, 2] = np.round(points[:, 0] / max(sx, 1e-6)).astype(np.int32)
        return out
    raise ValueError("index dimensionality must be 2 or 3")
