"""Deterministic Ramer-Douglas-Peucker implementation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _point_segment_distance(point: NDArray[np.float64], a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom == 0.0:
        return float(np.linalg.norm(point - a))
    t = float(np.dot(point - a, ab) / denom)
    t = min(1.0, max(0.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(point - proj))


def _rdp_indices(points: NDArray[np.float64], epsilon: float) -> list[int]:
    if len(points) <= 2:
        return list(range(len(points)))
    start = points[0]
    end = points[-1]
    max_dist = -1.0
    max_idx = 0
    for i in range(1, len(points) - 1):
        d = _point_segment_distance(points[i], start, end)
        if d > max_dist:
            max_dist = d
            max_idx = i
    if max_dist <= epsilon:
        return [0, len(points) - 1]
    left = _rdp_indices(points[: max_idx + 1], epsilon)
    right = _rdp_indices(points[max_idx:], epsilon)
    return left[:-1] + [i + max_idx for i in right]


def simplify_path(points: NDArray[np.float64], epsilon: float) -> NDArray[np.float64]:
    if epsilon <= 0.0 or len(points) <= 2:
        return points
    keep = _rdp_indices(points, epsilon)
    return points[np.array(keep, dtype=np.int64)]


def simplify_path_with_indices(points: NDArray[np.float64], epsilon: float) -> NDArray[np.int64]:
    if epsilon <= 0.0 or len(points) <= 2:
        return np.arange(len(points), dtype=np.int64)
    keep = _rdp_indices(points, epsilon)
    return np.array(keep, dtype=np.int64)
