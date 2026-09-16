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


def _point_segment_distances(points: NDArray[np.float64], a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    if len(points) == 0:
        return np.zeros(0, dtype=np.float64)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom == 0.0:
        return np.linalg.norm(points - a, axis=1).astype(np.float64, copy=False)
    t = np.clip(((points - a) @ ab) / denom, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.linalg.norm(points - proj, axis=1).astype(np.float64, copy=False)


def _rdp_indices(points: NDArray[np.float64], epsilon: float) -> list[int]:
    n = len(points)
    if n <= 2:
        return list(range(n))

    keep = {0, n - 1}
    stack = [(0, n - 1)]
    while stack:
        start_idx, end_idx = stack.pop()
        if end_idx <= start_idx + 1:
            continue
        distances = _point_segment_distances(
            points[start_idx + 1 : end_idx],
            points[start_idx],
            points[end_idx],
        )
        if len(distances) == 0:
            continue
        rel_idx = int(np.argmax(distances))
        max_dist = float(distances[rel_idx])
        if max_dist <= epsilon:
            continue
        split_idx = start_idx + 1 + rel_idx
        keep.add(split_idx)
        stack.append((split_idx, end_idx))
        stack.append((start_idx, split_idx))
    return sorted(keep)


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
