"""Small exact-sign/robust geometric intersection helpers for embedded graphs."""

from __future__ import annotations

import numpy as np


def _orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _on_segment(a: np.ndarray, b: np.ndarray, p: np.ndarray, tol: float) -> bool:
    if abs(_orient(a, b, p)) > tol:
        return False
    return bool(
        min(a[0], b[0]) - tol <= p[0] <= max(a[0], b[0]) + tol
        and min(a[1], b[1]) - tol <= p[1] <= max(a[1], b[1]) + tol
    )


def segments_intersect_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, tol: float = 1e-9) -> bool:
    o1, o2, o3, o4 = _orient(a, b, c), _orient(a, b, d), _orient(c, d, a), _orient(c, d, b)
    if ((o1 > tol and o2 < -tol) or (o1 < -tol and o2 > tol)) and (
        (o3 > tol and o4 < -tol) or (o3 < -tol and o4 > tol)
    ):
        return True
    return any(
        (
            abs(o) <= tol and _on_segment(x, y, p, tol)
            for o, x, y, p in ((o1, a, b, c), (o2, a, b, d), (o3, c, d, a), (o4, c, d, b))
        )
    )


def point_on_segment_nd(p: np.ndarray, a: np.ndarray, b: np.ndarray, tol: float = 1e-9) -> bool:
    ab = b - a
    den = float(np.dot(ab, ab))
    if den <= tol * tol:
        return float(np.linalg.norm(p - a)) <= tol
    t = float(np.dot(p - a, ab) / den)
    if t < -tol or t > 1.0 + tol:
        return False
    q = a + np.clip(t, 0.0, 1.0) * ab
    return float(np.linalg.norm(p - q)) <= tol


def segment_segment_distance_3d(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
) -> tuple[float, float, float]:
    """Return minimum distance and segment parameters for two 3D segments."""
    u = b - a
    v = d - c
    w = a - c
    aa = float(np.dot(u, u))
    bb = float(np.dot(u, v))
    cc = float(np.dot(v, v))
    dd = float(np.dot(u, w))
    ee = float(np.dot(v, w))
    den = aa * cc - bb * bb
    eps = 1e-30
    if aa <= eps or cc <= eps:
        return float("inf"), 0.0, 0.0
    if den <= eps:
        s = 0.0
        t = float(np.clip(ee / cc, 0.0, 1.0))
    else:
        s = float(np.clip((bb * ee - cc * dd) / den, 0.0, 1.0))
        t = float(np.clip((aa * ee - bb * dd) / den, 0.0, 1.0))
        # Reproject after clamping for endpoint cases.
        p = a + s * u
        t = float(np.clip(np.dot(p - c, v) / cc, 0.0, 1.0))
        q = c + t * v
        s = float(np.clip(np.dot(q - a, u) / aa, 0.0, 1.0))
    p = a + s * u
    q = c + t * v
    return float(np.linalg.norm(p - q)), s, t


def segments_intersect_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, tol: float = 1e-9) -> bool:
    dist, _, _ = segment_segment_distance_3d(a, b, c, d)
    return dist <= tol
