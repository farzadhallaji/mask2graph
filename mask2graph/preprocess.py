"""Input preprocessing for binary masks."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt, generate_binary_structure, label

from .config import CleanupConfig
from .types import CleanupReport


def _full_connectivity(ndim: int) -> NDArray[np.bool_]:
    return generate_binary_structure(ndim, ndim)


def _voxel_measure(spacing: tuple[float, ...]) -> float:
    measure = 1.0
    for value in spacing:
        measure *= float(value)
    return float(measure)


def preprocess_mask(
    mask: NDArray[np.bool_],
    cleanup: CleanupConfig,
    *,
    spacing: tuple[float, ...] | None = None,
    return_report: bool = False,
) -> NDArray[np.bool_] | tuple[NDArray[np.bool_], CleanupReport]:
    out = np.asarray(mask != 0, dtype=np.bool_).copy()
    if out.ndim not in (2, 3):
        raise ValueError("mask must be 2D or 3D")
    spacing_t = tuple(1.0 for _ in range(out.ndim)) if spacing is None else tuple(float(v) for v in spacing)
    if len(spacing_t) != out.ndim:
        raise ValueError("spacing length must match mask.ndim")
    if any(v <= 0 for v in spacing_t):
        raise ValueError("spacing values must be positive")

    removed_object_sizes: list[float] = []
    filled_hole_sizes: list[float] = []
    filled_hole_radii: list[float] = []

    if cleanup.enabled:
        connectivity = _full_connectivity(out.ndim)
        voxel_measure = _voxel_measure(spacing_t)

        if cleanup.min_object_size > 0:
            labels_fg, n_fg = label(out, structure=connectivity)
            if n_fg > 0:
                counts_fg = np.bincount(labels_fg.ravel())
                remove_labels = [
                    lab
                    for lab in range(1, len(counts_fg))
                    if (counts_fg[lab] * voxel_measure) < cleanup.min_object_size
                ]
                removed_object_sizes = [float(counts_fg[lab] * voxel_measure) for lab in remove_labels]
                if remove_labels:
                    out[np.isin(labels_fg, np.asarray(remove_labels, dtype=labels_fg.dtype))] = False

        if cleanup.max_hole_size > 0 or cleanup.max_hole_radius > 0:
            labels_bg, n_bg = label(~out, structure=connectivity)
            if n_bg > 0:
                border_labels: set[int] = set()
                for axis in range(out.ndim):
                    low = [slice(None)] * out.ndim
                    high = [slice(None)] * out.ndim
                    low[axis] = 0
                    high[axis] = -1
                    border_labels.update(np.unique(labels_bg[tuple(low)]).tolist())
                    border_labels.update(np.unique(labels_bg[tuple(high)]).tolist())

                counts_bg = np.bincount(labels_bg.ravel())
                for lab in range(1, len(counts_bg)):
                    if lab in border_labels:
                        continue

                    hole_mask = labels_bg == lab
                    hole_size = float(counts_bg[lab] * voxel_measure)
                    size_ok = cleanup.max_hole_size <= 0 or hole_size <= cleanup.max_hole_size

                    radius_ok = True
                    hole_radius = 0.0
                    if cleanup.max_hole_radius > 0:
                        dt_hole = distance_transform_edt(hole_mask, sampling=spacing_t)
                        hole_radius = float(dt_hole[hole_mask].max()) if np.any(hole_mask) else 0.0
                        radius_ok = hole_radius <= cleanup.max_hole_radius

                    if size_ok and radius_ok:
                        out[hole_mask] = True
                        filled_hole_sizes.append(hole_size)
                        filled_hole_radii.append(hole_radius)

    report = CleanupReport(
        n_removed_objects=len(removed_object_sizes),
        n_filled_holes=len(filled_hole_sizes),
        removed_object_sizes=removed_object_sizes,
        filled_hole_sizes=filled_hole_sizes,
        filled_hole_radii=filled_hole_radii,
    )
    out_bool = np.asarray(out, dtype=np.bool_)
    if return_report:
        return out_bool, report
    return out_bool
