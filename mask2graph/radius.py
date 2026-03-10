"""Geometric metrics and EDT-based radius estimation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt

from .types import Edge, Node


def edge_length(path_xyz: NDArray[np.float64]) -> float:
    if len(path_xyz) < 2:
        return 0.0
    diffs = np.diff(path_xyz, axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


def update_edge_lengths(edges: list[Edge]) -> None:
    for edge in edges:
        edge.length = edge_length(edge.path_xyz)
        edge.voxel_length = int(len(edge.path_index))


def estimate_radii(
    *,
    nodes: list[Node],
    edges: list[Edge],
    node_labels: NDArray[np.int32],
    mask_processed: NDArray[np.bool_],
    spacing: tuple[float, ...],
) -> NDArray[np.float64]:
    dt = distance_transform_edt(mask_processed, sampling=spacing)
    for edge in edges:
        samples = np.array([dt[tuple(int(v) for v in idx)] for idx in edge.path_index], dtype=np.float64)
        edge.radius_profile = samples
        edge.radius_mean = float(samples.mean()) if len(samples) else None
        edge.radius_median = float(np.median(samples)) if len(samples) else None

    for node in nodes:
        if node.type == "cycle":
            samples = np.array([dt[node.index]], dtype=np.float64)
        else:
            label = int(node_labels[node.index]) if node_labels.shape == mask_processed.shape else 0
            if label <= 0:
                samples = np.array([dt[node.index]], dtype=np.float64)
            else:
                coords = np.argwhere(node_labels == label)
                samples = np.array([dt[tuple(int(v) for v in c)] for c in coords], dtype=np.float64)
        node.radius_mean = float(samples.mean()) if len(samples) else None
        node.radius_median = float(np.median(samples)) if len(samples) else None
    return dt
