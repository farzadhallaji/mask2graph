"""Core datatypes for extracted mask graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


class ExtractionError(RuntimeError):
    """Raised when traversal reaches an impossible graph state."""


class SerializationError(ValueError):
    """Raised when graph JSON fails schema validation."""


@dataclass
class Node:
    id: int
    xyz: tuple[float, float, float]
    index: tuple[int, ...]
    type: str
    degree: int
    voxel_count: int
    radius_mean: float | None = None
    radius_median: float | None = None


@dataclass
class Edge:
    id: int
    u: int
    v: int
    path_xyz: NDArray[np.float64]
    path_index: NDArray[np.int32]
    length: float
    voxel_length: int
    radius_mean: float | None = None
    radius_median: float | None = None
    radius_profile: NDArray[np.float64] | None = None
    is_self_loop: bool = False


@dataclass
class GraphMeta:
    version: str
    ndim: int
    shape: tuple[int, ...]
    spacing: tuple[float, ...]
    config: dict[str, Any]
    input_hash: str
    processed_mask_hash: str


@dataclass
class MaskGraph:
    nodes: list[Node]
    edges: list[Edge]
    meta: GraphMeta


@dataclass
class DebugArtifacts:
    mask_input: NDArray[np.bool_]
    mask_processed: NDArray[np.bool_]
    skeleton: NDArray[np.bool_]
    degree_map: NDArray[np.int32]
    node_candidates: NDArray[np.bool_]
    node_labels: NDArray[np.int32]
    pruned_skeleton: NDArray[np.bool_]
    component_labels: NDArray[np.int32]
