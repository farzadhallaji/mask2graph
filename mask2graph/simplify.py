"""Non-destructive branch simplification for straight-line PSLG export."""

from __future__ import annotations

import math

import numpy as np

from .config import SimplifyConfig
from .types import Edge
from .utils.rdp import simplify_path_with_indices

_EXACT_OPTIMAL_MAX_SAMPLES = 512


def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    den = float(np.dot(ab, ab))
    if den <= 0.0:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(np.dot(p - a, ab) / den, 0.0, 1.0))
    q = a + t * ab
    return float(np.linalg.norm(p - q))


def max_subpath_error(path: np.ndarray, i: int, j: int) -> float:
    if j <= i + 1:
        return 0.0
    a, b = path[i], path[j]
    points = np.asarray(path[i + 1 : j], dtype=np.float64)
    ab = b - a
    den = float(np.dot(ab, ab))
    if den <= 0.0:
        return float(np.linalg.norm(points - a, axis=1).max())
    t = np.clip(((points - a) @ ab) / den, 0.0, 1.0)
    q = a + t[:, None] * ab
    return float(np.linalg.norm(points - q, axis=1).max())


def _protected_indices(edge: Edge, degrees: float) -> set[int]:
    if degrees <= 0.0 or edge.turning_angle_profile is None:
        return set()
    threshold = math.radians(float(degrees))
    return {i for i, a in enumerate(edge.turning_angle_profile) if float(a) >= threshold}


def _rdp_keep_with_protected(path: np.ndarray, epsilon: float, protected: set[int]) -> np.ndarray:
    n = len(path)
    anchors = [0, *(i for i in sorted(protected) if 0 < i < n - 1), n - 1]
    keep: list[int] = []
    for start, end in zip(anchors[:-1], anchors[1:]):
        local = simplify_path_with_indices(path[start : end + 1], epsilon).astype(np.int32, copy=False)
        shifted = [int(i) + start for i in local]
        if keep:
            shifted = shifted[1:]
        keep.extend(shifted)
    return np.asarray(keep, dtype=np.int32)


def _protected_prefix(protected: set[int], n: int) -> np.ndarray:
    prefix = np.zeros(n + 1, dtype=np.int32)
    for idx in protected:
        if 0 < idx < n - 1:
            prefix[idx + 1] = 1
    return np.cumsum(prefix, dtype=np.int32)


def optimal_keep_indices(edge: Edge, epsilon: float, protect_angle_degrees: float = 0.0) -> np.ndarray:
    """Minimum number of straight segments for an open branch under a max-error bound.

    The candidate graph is a DAG on path samples.  An arc ``i -> j`` is present
    exactly when the subpath is within ``epsilon`` of its chord and the shortcut
    does not erase a protected high-turn sample.  Unit arc costs make shortest
    path equivalent to minimum segment count.
    """
    path = np.asarray(edge.path_xyz, dtype=np.float64)
    n = len(path)
    if n <= 2:
        return np.arange(n, dtype=np.int32)
    if edge.is_self_loop or np.allclose(path[0], path[-1]):
        keep = simplify_path_with_indices(path, epsilon)
        keep = np.unique(np.concatenate(([0], keep, [n - 1]))).astype(np.int32)
        # A polygonal cycle needs at least three nonzero sides (four samples with closure).
        unique = [int(i) for i in keep[:-1] if not np.allclose(path[i], path[keep[-1]])]
        if len(unique) < 3:
            picks = np.linspace(0, n - 2, num=min(3, n - 1), dtype=int)
            keep = np.unique(np.concatenate((picks, [n - 1]))).astype(np.int32)
        return keep

    protected = _protected_indices(edge, protect_angle_degrees)
    if n > _EXACT_OPTIMAL_MAX_SAMPLES:
        return _rdp_keep_with_protected(path, epsilon, protected)

    protected_prefix = _protected_prefix(protected, n)
    best = [10**9] * n
    prev = [-1] * n
    best[0] = 0
    for j in range(1, n):
        for i in range(j - 1, -1, -1):
            if best[i] >= 10**9:
                continue
            if int(protected_prefix[j] - protected_prefix[i + 1]) > 0:
                continue
            if max_subpath_error(path, i, j) > epsilon:
                continue
            cand = best[i] + 1
            # Deterministic tie: prefer the earlier predecessor, which tends to
            # retain longer initial chords and is independent of set ordering.
            if cand < best[j] or (cand == best[j] and (prev[j] < 0 or i < prev[j])):
                best[j] = cand
                prev[j] = i
    if prev[-1] < 0:
        return np.arange(n, dtype=np.int32)
    rev = [n - 1]
    cur = n - 1
    while cur != 0:
        cur = prev[cur]
        if cur < 0:
            return np.arange(n, dtype=np.int32)
        rev.append(cur)
    rev.reverse()
    return np.asarray(rev, dtype=np.int32)


def simplify_edge(edge: Edge, config: SimplifyConfig) -> float:
    """Populate non-destructive simplified geometry; return max approximation error."""
    n = len(edge.path_xyz)
    if n == 0:
        edge.simplified_indices = np.zeros((0, edge.path_index.shape[1]), dtype=np.int32)
        edge.simplified_xyz = np.zeros((0, 3), dtype=np.float64)
        return 0.0
    if not config.enabled or config.method == "none" or config.epsilon <= 0.0:
        keep = np.arange(n, dtype=np.int32)
    elif config.method == "rdp":
        keep = simplify_path_with_indices(edge.path_xyz, config.epsilon).astype(np.int32)
    else:
        keep = optimal_keep_indices(edge, config.epsilon, config.protect_angle_degrees)
    edge.simplified_indices = edge.path_index[keep].copy()
    edge.simplified_xyz = edge.path_xyz[keep].copy()
    err = 0.0
    for a, b in zip(keep[:-1], keep[1:]):
        err = max(err, max_subpath_error(edge.path_xyz, int(a), int(b)))
    return float(err)


def simplify_graph_edges(edges: list[Edge], config: SimplifyConfig) -> float:
    return max((simplify_edge(edge, config) for edge in edges), default=0.0)
