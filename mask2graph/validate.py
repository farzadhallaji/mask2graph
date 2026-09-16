"""Extraction, topology, and embedded straight-line graph validation."""

from __future__ import annotations

from collections import Counter

import numpy as np

from .config import SimplifyConfig, ValidationConfig
from .intersections import point_on_segment_nd, segments_intersect_2d, segments_intersect_3d
from .topology import update_embedded_diagnostics
from .types import (
    Edge,
    EmbeddedGraph,
    EmbeddedSegment,
    EmbeddedVertex,
    Mask2Graph,
    TopologyDiagnostics,
    ValidationError,
)


def validate_skeleton_coverage(diag: TopologyDiagnostics) -> None:
    if not diag.coverage_complete or diag.expected_trace_segments != diag.covered_trace_segments:
        raise ValidationError(
            f"skeleton coverage incomplete: {diag.covered_trace_segments}/{diag.expected_trace_segments}"
        )


def validate_topology(graph: Mask2Graph, diag: TopologyDiagnostics, *, allow_topology_edit: bool = False) -> None:
    # The raw full-neighbour pixel graph can contain artificial cycles inside a
    # thick junction.  Compare the logical graph against the junction-MST-cleaned
    # skeleton, not against the raw digital adjacency graph.
    if allow_topology_edit:
        return
    if diag.cleaned_beta0 != diag.logical_beta0:
        raise ValidationError(
            f"component count changed after logical contraction: {diag.cleaned_beta0} -> {diag.logical_beta0}"
        )
    if diag.cleaned_beta1 != diag.logical_beta1:
        raise ValidationError(
            f"cycle rank changed after logical contraction: {diag.cleaned_beta1} -> {diag.logical_beta1}"
        )
    node_ids = {n.id for n in graph.nodes}
    for e in graph.edges:
        if e.u not in node_ids or e.v not in node_ids:
            raise ValidationError(f"edge {e.id} references a missing node")
        if len(e.path_xyz) != len(e.path_index):
            raise ValidationError(f"edge {e.id} path xyz/index lengths differ")
        if len(e.path_xyz) < 2 and not (e.u == e.v and len(e.path_xyz) == 1):
            raise ValidationError(f"edge {e.id} has an invalid path")
        if len(e.path_xyz) >= 1:
            u = next(n for n in graph.nodes if n.id == e.u)
            v = next(n for n in graph.nodes if n.id == e.v)
            if not np.allclose(np.asarray(u.xyz), e.path_xyz[0]):
                raise ValidationError(f"edge {e.id} does not start at node {e.u} geometry")
            if not np.allclose(np.asarray(v.xyz), e.path_xyz[-1]):
                raise ValidationError(f"edge {e.id} does not end at node {e.v} geometry")


def _edge_export_xyz(edge: Edge, simplified: bool) -> np.ndarray:
    if simplified and edge.simplified_xyz is not None and len(edge.simplified_xyz) >= 2:
        return np.asarray(edge.simplified_xyz, dtype=np.float64)
    return np.asarray(edge.path_xyz, dtype=np.float64)


def build_embedded_graph(graph: Mask2Graph, *, simplified: bool = True) -> EmbeddedGraph:
    vertices: list[EmbeddedVertex] = []
    segments: list[EmbeddedSegment] = []
    node_to_vertex: dict[int, int] = {}
    for node in sorted(graph.nodes, key=lambda n: n.id):
        vid = len(vertices)
        node_to_vertex[node.id] = vid
        vertices.append(
            EmbeddedVertex(
                id=vid,
                xyz=tuple(float(v) for v in node.xyz),
                source_node_id=node.id,
                source_edge_id=None,
                source_path_position=None,
                kind="topology",
                on_crop_boundary=bool(getattr(node, "on_crop_boundary", False)),
                created_by=getattr(node, "created_by", None),
            )
        )

    for edge in sorted(graph.edges, key=lambda e: e.id):
        path = _edge_export_xyz(edge, simplified)
        if len(path) < 2:
            continue
        vids: list[int] = [node_to_vertex[edge.u]]
        for k in range(1, len(path) - 1):
            vid = len(vertices)
            vertices.append(
                EmbeddedVertex(
                    id=vid,
                    xyz=tuple(float(v) for v in path[k]),
                    source_node_id=None,
                    source_edge_id=edge.source_edge_id if edge.source_edge_id is not None else edge.id,
                    source_path_position=k,
                    kind="geometry",
                    on_crop_boundary=False,
                    created_by=edge.created_by,
                )
            )
            vids.append(vid)
        vids.append(node_to_vertex[edge.v])
        source_eid = edge.source_edge_id if edge.source_edge_id is not None else edge.id
        for a, b in zip(vids[:-1], vids[1:]):
            segments.append(
                EmbeddedSegment(
                    id=len(segments),
                    u=a,
                    v=b,
                    source_edge_id=int(source_eid),
                    source_arc_range=edge.source_arc_range,
                )
            )

    metadata = {
        "source_graph_hash": graph.meta.source_graph_hash or graph.meta.graph_hash,
        "graph_hash": graph.meta.graph_hash,
        "grid_aligned": bool(graph.meta.grid_aligned),
        "transform_exact": bool(graph.meta.transform_exact),
        "rationalization_error": float(graph.meta.rationalization_error),
        "augmentation_history": list(graph.meta.augmentation_history),
        "augmentation_seed": graph.meta.augmentation_seed,
    }
    return EmbeddedGraph(ndim=graph.meta.ndim, vertices=vertices, segments=segments, metadata=metadata)


def validate_embedded_graph(embedded: EmbeddedGraph, *, tolerance: float = 1e-9) -> None:
    verts = {v.id: np.asarray(v.xyz, dtype=np.float64) for v in embedded.vertices}
    if len(verts) != len(embedded.vertices):
        raise ValidationError("duplicate embedded vertex id")
    seen_pairs: set[tuple[int, int]] = set()
    for s in embedded.segments:
        if s.u == s.v:
            raise ValidationError(f"embedded segment {s.id} is a self-loop/zero-topology segment")
        a, b = verts[s.u], verts[s.v]
        if float(np.linalg.norm(a - b)) <= tolerance:
            raise ValidationError(f"embedded segment {s.id} has zero geometric length")
        key = (min(s.u, s.v), max(s.u, s.v))
        if key in seen_pairs:
            raise ValidationError(f"duplicate embedded segment between {key}")
        seen_pairs.add(key)

    if embedded.ndim == 2:
        for i, s in enumerate(embedded.segments):
            a, b = verts[s.u][:2], verts[s.v][:2]
            for t in embedded.segments[i + 1 :]:
                shared = {s.u, s.v} & {t.u, t.v}
                c, d = verts[t.u][:2], verts[t.v][:2]
                if not segments_intersect_2d(a, b, c, d, tolerance):
                    continue
                if shared:
                    # Shared endpoint is valid only if the intersection is at that
                    # declared endpoint and there is no overlapping continuation.
                    common = next(iter(shared))
                    p = verts[common][:2]
                    other_s = verts[s.v if s.u == common else s.u][:2]
                    other_t = verts[t.v if t.u == common else t.u][:2]
                    # If either opposite endpoint lies on the other segment there
                    # is an overlap or an undeclared split.
                    if point_on_segment_nd(other_s, p, other_t, tolerance) or point_on_segment_nd(
                        other_t, p, other_s, tolerance
                    ):
                        raise ValidationError(f"embedded segments {s.id},{t.id} overlap")
                    continue
                raise ValidationError(f"nonincident embedded segments {s.id},{t.id} intersect")
    else:
        # A 3D target is still an embedded straight-line graph: projected
        # crossings are harmless, but true geometric intersections of
        # nonincident segments are not.
        for i, s in enumerate(embedded.segments):
            a, b = verts[s.u], verts[s.v]
            for vid, p in verts.items():
                if vid in (s.u, s.v):
                    continue
                if point_on_segment_nd(p, a, b, tolerance):
                    raise ValidationError(f"vertex {vid} lies on unrelated 3D segment {s.id}")
            for t in embedded.segments[i + 1 :]:
                shared = {s.u, s.v} & {t.u, t.v}
                c, d = verts[t.u], verts[t.v]
                if not segments_intersect_3d(a, b, c, d, tolerance):
                    continue
                if shared:
                    common = next(iter(shared))
                    p = verts[common]
                    other_s = verts[s.v if s.u == common else s.u]
                    other_t = verts[t.v if t.u == common else t.u]
                    if point_on_segment_nd(other_s, p, other_t, tolerance) or point_on_segment_nd(
                        other_t, p, other_s, tolerance
                    ):
                        raise ValidationError(f"3D embedded segments {s.id},{t.id} overlap")
                    continue
                raise ValidationError(f"nonincident 3D embedded segments {s.id},{t.id} intersect")


def validate_embedded_topology(diag: TopologyDiagnostics, embedded: EmbeddedGraph) -> None:
    update_embedded_diagnostics(diag, embedded)
    if (diag.logical_beta0, diag.logical_beta1) != (diag.embedded_beta0, diag.embedded_beta1):
        raise ValidationError(
            "embedded straight-line subdivision changed graph topology: "
            f"logical=({diag.logical_beta0},{diag.logical_beta1}) "
            f"embedded=({diag.embedded_beta0},{diag.embedded_beta1})"
        )


def build_validated_embedded_graph(
    graph: Mask2Graph,
    diag: TopologyDiagnostics,
    *,
    simplify_config: SimplifyConfig,
    validation_config: ValidationConfig,
) -> EmbeddedGraph:
    embedded = build_embedded_graph(graph, simplified=True)
    try:
        if validation_config.validate_embedding:
            validate_embedded_graph(embedded, tolerance=validation_config.tolerance)
        if validation_config.validate_topology:
            validate_embedded_topology(diag, embedded)
        embedded.metadata["embedding_validated"] = bool(validation_config.validate_embedding)
        return embedded
    except ValidationError:
        if not (simplify_config.topology_guard and simplify_config.fallback_to_original):
            raise
        fallback = build_embedded_graph(graph, simplified=False)
        if validation_config.validate_embedding:
            validate_embedded_graph(fallback, tolerance=validation_config.tolerance)
        if validation_config.validate_topology:
            validate_embedded_topology(diag, fallback)
        fallback.metadata["embedding_validated"] = bool(validation_config.validate_embedding)
        diag.simplification_fallback = True
        return fallback
