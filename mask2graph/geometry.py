"""Spacing-aware branch geometry profiles."""

from __future__ import annotations

import numpy as np

from .types import Edge


def arclength_profile(path: np.ndarray) -> np.ndarray:
    path = np.asarray(path, dtype=np.float64)
    if len(path) == 0:
        return np.zeros(0, dtype=np.float64)
    if len(path) == 1:
        return np.zeros(1, dtype=np.float64)
    ds = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(ds))).astype(np.float64)


def tangent_profile(path: np.ndarray, window: int = 2) -> np.ndarray:
    path = np.asarray(path, dtype=np.float64)
    out = np.zeros_like(path, dtype=np.float64)
    n = len(path)
    if n <= 1:
        return out
    w = max(1, int(window))
    for i in range(n):
        a = max(0, i - w)
        b = min(n - 1, i + w)
        vec = path[b] - path[a]
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            out[i] = vec / norm
    return out


def turning_angle_profile(tangents: np.ndarray) -> np.ndarray:
    t = np.asarray(tangents, dtype=np.float64)
    out = np.zeros(len(t), dtype=np.float64)
    for i in range(1, max(1, len(t) - 1)):
        if i >= len(t) - 1:
            break
        a, b = t[i - 1], t[i + 1]
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            continue
        dot = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
        out[i] = float(np.arccos(dot))
    return out


def curvature_profile(angles: np.ndarray, arclen: np.ndarray) -> np.ndarray:
    a = np.asarray(angles, dtype=np.float64)
    s = np.asarray(arclen, dtype=np.float64)
    out = np.zeros(len(a), dtype=np.float64)
    for i in range(1, len(a) - 1):
        ds = float(s[i + 1] - s[i - 1])
        if ds > 0:
            out[i] = 2.0 * float(a[i]) / ds
    return out


def update_geometry_profiles(edges: list[Edge], *, tangent_window: int = 2, compute_curvature: bool = True) -> None:
    for edge in edges:
        p = np.asarray(edge.path_xyz, dtype=np.float64)
        s = arclength_profile(p)
        t = tangent_profile(p, window=tangent_window)
        a = turning_angle_profile(t)
        edge.arclen_profile = s
        edge.tangent_profile = t
        edge.turning_angle_profile = a
        edge.curvature_profile = curvature_profile(a, s) if compute_curvature else None
        if len(p) >= 2:
            edge.chord_length = float(np.linalg.norm(p[-1] - p[0]))
        else:
            edge.chord_length = 0.0
        edge.tortuosity = float(edge.length / edge.chord_length) if edge.chord_length > 0 else 1.0
