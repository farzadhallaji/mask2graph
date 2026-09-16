"""Core datatypes for extracted, augmented, and embedded mask graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


class ExtractionError(RuntimeError):
    """Raised when traversal reaches an impossible graph state."""


class SerializationError(ValueError):
    """Raised when graph JSON fails schema validation."""


class ValidationError(ValueError):
    """Raised when an extracted or embedded graph violates a required invariant."""


@dataclass
class Node:
    id: int
    xyz: tuple[float, float, float]
    # ``index`` is the current raster index only when ``index_valid`` is true.
    # ``source_index`` is immutable provenance back to the original mask sample.
    index: tuple[int, ...]
    type: str
    degree: int
    voxel_count: int
    radius_mean: float | None = None
    radius_median: float | None = None
    radius_min: float | None = None
    radius_max: float | None = None
    support_indices: NDArray[np.int32] | None = None
    junction_mst_edges: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = ()
    on_image_boundary: bool = False
    boundary_axes: tuple[int, ...] = ()
    source_index: tuple[int, ...] | None = None
    index_valid: bool = True
    on_crop_boundary: bool = False
    created_by: str | None = None

    def __post_init__(self) -> None:
        if self.source_index is None and self.index_valid:
            self.source_index = tuple(int(v) for v in self.index)


@dataclass
class Edge:
    id: int
    u: int
    v: int
    path_xyz: NDArray[np.float64]
    # Current raster samples where known.  Synthetic crop intersections use -1
    # sentinels and set ``path_index_valid=False``; source provenance is carried
    # separately instead of fabricating raster coordinates.
    path_index: NDArray[np.int32]
    length: float
    voxel_length: int
    radius_mean: float | None = None
    radius_median: float | None = None
    radius_min: float | None = None
    radius_max: float | None = None
    radius_profile: NDArray[np.float64] | None = None
    arclen_profile: NDArray[np.float64] | None = None
    tangent_profile: NDArray[np.float64] | None = None
    curvature_profile: NDArray[np.float64] | None = None
    turning_angle_profile: NDArray[np.float64] | None = None
    simplified_indices: NDArray[np.int32] | None = None
    simplified_xyz: NDArray[np.float64] | None = None
    chord_length: float = 0.0
    tortuosity: float = 1.0
    is_self_loop: bool = False
    provenance: str = "skeleton_trace"
    source_path_index: NDArray[np.int32] | None = None
    path_index_valid: bool = True
    source_edge_id: int | None = None
    source_arc_range: tuple[float, float] | None = None
    crop_fragment_index: int | None = None
    created_by: str | None = None

    def __post_init__(self) -> None:
        if self.source_path_index is None and self.path_index_valid:
            self.source_path_index = np.asarray(self.path_index, dtype=np.int32).copy()


@dataclass
class GraphMeta:
    version: str
    ndim: int
    shape: tuple[int, ...]
    spacing: tuple[float, ...]
    config: dict[str, Any]
    input_hash: str
    processed_mask_hash: str
    config_hash: str = ""
    graph_hash: str = ""
    source_graph_hash: str = ""
    grid_aligned: bool = True
    transform_exact: bool = True
    rationalization_error: float = 0.0
    augmentation_history: list[dict[str, Any]] = field(default_factory=list)
    augmentation_seed: int | None = None


@dataclass
class CleanupReport:
    n_removed_objects: int
    n_filled_holes: int
    removed_object_sizes: list[float]
    filled_hole_sizes: list[float]
    filled_hole_radii: list[float]


@dataclass
class TopologyDiagnostics:
    raw_skeleton_vertices: int = 0
    raw_skeleton_edges: int = 0
    raw_beta0: int = 0
    raw_beta1: int = 0
    cleaned_skeleton_edges: int = 0
    cleaned_beta0: int = 0
    cleaned_beta1: int = 0
    logical_vertices: int = 0
    logical_edges: int = 0
    logical_beta0: int = 0
    logical_beta1: int = 0
    embedded_vertices: int = 0
    embedded_edges: int = 0
    embedded_beta0: int = 0
    embedded_beta1: int = 0
    expected_trace_segments: int = 0
    covered_trace_segments: int = 0
    coverage_complete: bool = True
    simplification_fallback: bool = False
    max_simplification_error: float = 0.0
    cleanup_topology_edit_enabled: bool = False
    cleanup_topology_changed: bool = False
    new_intersections: int = 0
    topology_before_beta0: int | None = None
    topology_before_beta1: int | None = None
    topology_after_beta0: int | None = None
    topology_after_beta1: int | None = None


@dataclass
class Mask2Graph:
    nodes: list[Node]
    edges: list[Edge]
    meta: GraphMeta
    diagnostics: TopologyDiagnostics | None = None


@dataclass(frozen=True)
class EmbeddedVertex:
    id: int
    xyz: tuple[float, float, float]
    source_node_id: int | None
    source_edge_id: int | None
    source_path_position: int | None
    kind: str
    on_crop_boundary: bool = False
    created_by: str | None = None


@dataclass(frozen=True)
class EmbeddedSegment:
    id: int
    u: int
    v: int
    source_edge_id: int
    source_arc_range: tuple[float, float] | None = None


@dataclass
class EmbeddedGraph:
    ndim: int
    vertices: list[EmbeddedVertex]
    segments: list[EmbeddedSegment]
    diagnostics: TopologyDiagnostics | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MaskGraphResult:
    topology_graph: Mask2Graph
    embedded_graph: EmbeddedGraph
    diagnostics: TopologyDiagnostics
    debug: "DebugArtifacts | None" = None


@dataclass
class DebugArtifacts:
    mask_input: NDArray[np.bool_]
    mask_processed: NDArray[np.bool_]
    cleanup_report: CleanupReport | None
    skeleton: NDArray[np.bool_]
    degree_map: NDArray[np.int32]
    node_candidates: NDArray[np.bool_]
    node_labels: NDArray[np.int32]
    pruned_skeleton: NDArray[np.bool_]
    component_labels: NDArray[np.int32]
    diagnostics: TopologyDiagnostics | None = None
    notes: dict[str, Any] = field(default_factory=dict)
