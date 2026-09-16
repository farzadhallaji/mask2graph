"""Geometric clipping of full branch polylines against axis-aligned boxes."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import ExtractConfig
from .geometry import arclength_profile, update_geometry_profiles
from .radius import edge_length
from .simplify import simplify_graph_edges
from .topology import update_embedded_diagnostics, update_logical_diagnostics
from .types import Edge, GraphMeta, Mask2Graph, MaskGraphResult, Node, TopologyDiagnostics, ValidationError
from .utils.hash import graph_content_hash
from .validate import build_validated_embedded_graph, build_embedded_graph, validate_embedded_graph, validate_embedded_topology


@dataclass
class _ClipPoint:
    xyz: np.ndarray
    source_index: np.ndarray | None
    radius: float | None
    source_arc: float


@dataclass
class _Fragment:
    points: list[_ClipPoint]


def normalize_box_bounds(ndim: int, *, bounds: Sequence[float] | None = None, minimum: Sequence[float] | None = None, maximum: Sequence[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    if bounds is not None:
        vals = [float(v) for v in bounds]
        if len(vals) != 2 * ndim:
            raise ValueError(f"box bounds must contain {2 * ndim} numbers")
        lo = np.asarray(vals[0::2], dtype=np.float64)
        hi = np.asarray(vals[1::2], dtype=np.float64)
    else:
        if minimum is None or maximum is None:
            raise ValueError("provide bounds=... or both minimum=... and maximum=...")
        lo = np.asarray(tuple(float(v) for v in minimum), dtype=np.float64)
        hi = np.asarray(tuple(float(v) for v in maximum), dtype=np.float64)
        if len(lo) != ndim or len(hi) != ndim:
            raise ValueError("crop minimum/maximum must match graph dimension")
    if np.any(~np.isfinite(lo)) or np.any(~np.isfinite(hi)) or np.any(hi < lo):
        raise ValueError("crop bounds must be finite with min <= max")
    return lo, hi


def point_in_box(point: np.ndarray, lo: np.ndarray, hi: np.ndarray, tolerance: float = 1e-9) -> bool:
    p = np.asarray(point, dtype=np.float64)[: len(lo)]
    return bool(np.all(p >= lo - tolerance) and np.all(p <= hi + tolerance))


def point_on_box_boundary(point: np.ndarray, lo: np.ndarray, hi: np.ndarray, tolerance: float = 1e-9) -> bool:
    p = np.asarray(point, dtype=np.float64)[: len(lo)]
    if not point_in_box(p, lo, hi, tolerance):
        return False
    return bool(np.any(np.isclose(p, lo, atol=tolerance, rtol=0.0)) or np.any(np.isclose(p, hi, atol=tolerance, rtol=0.0)))


def clip_segment_box(p0: np.ndarray, p1: np.ndarray, lo: np.ndarray, hi: np.ndarray, tolerance: float = 1e-12) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    """Liang-Barsky style nD segment clipping.

    Returns clipped endpoints plus source-segment parameters ``t0,t1``.
    Tangential zero-length contacts return ``None`` because they do not form a
    graph edge.
    """
    a = np.asarray(p0, dtype=np.float64)
    b = np.asarray(p1, dtype=np.float64)
    d = b - a
    t0, t1 = 0.0, 1.0
    for k in range(len(lo)):
        if abs(float(d[k])) <= tolerance:
            if a[k] < lo[k] - tolerance or a[k] > hi[k] + tolerance:
                return None
            continue
        u0 = float((lo[k] - a[k]) / d[k])
        u1 = float((hi[k] - a[k]) / d[k])
        enter, leave = (u0, u1) if u0 <= u1 else (u1, u0)
        t0 = max(t0, enter)
        t1 = min(t1, leave)
        if t0 > t1 + tolerance:
            return None
    t0 = float(np.clip(t0, 0.0, 1.0))
    t1 = float(np.clip(t1, 0.0, 1.0))
    q0 = a + t0 * d
    q1 = a + t1 * d
    # Snap clipped coordinates exactly to box faces when within tolerance.
    for q in (q0, q1):
        for k in range(len(lo)):
            if abs(float(q[k] - lo[k])) <= 10 * tolerance:
                q[k] = lo[k]
            elif abs(float(q[k] - hi[k])) <= 10 * tolerance:
                q[k] = hi[k]
    if float(np.linalg.norm(q1 - q0)) <= tolerance:
        return None
    return q0, q1, t0, t1


def _interp_radius(edge: Edge, i: int, t: float) -> float | None:
    if edge.radius_profile is None or len(edge.radius_profile) != len(edge.path_xyz):
        return None
    return float((1.0 - t) * edge.radius_profile[i] + t * edge.radius_profile[i + 1])


def _source_index_at(edge: Edge, i: int, endpoint: int, t: float, tol: float) -> np.ndarray | None:
    if not edge.path_index_valid:
        return None
    if t <= tol:
        return np.asarray(edge.path_index[i], dtype=np.int32).copy()
    if t >= 1.0 - tol:
        return np.asarray(edge.path_index[i + endpoint], dtype=np.int32).copy()
    return None


def _clip_point(edge: Edge, i: int, q: np.ndarray, t: float, arc: np.ndarray, seglen: float, *, endpoint: int, tol: float) -> _ClipPoint:
    source_index = _source_index_at(edge, i, endpoint, t, tol)
    return _ClipPoint(
        xyz=np.asarray(q, dtype=np.float64).copy(),
        source_index=source_index,
        radius=_interp_radius(edge, i, t),
        source_arc=float(arc[i] + t * seglen),
    )


def _same_point(a: _ClipPoint, b: _ClipPoint, tol: float) -> bool:
    return bool(np.allclose(a.xyz, b.xyz, atol=tol, rtol=0.0))


def _append_point(points: list[_ClipPoint], point: _ClipPoint, tol: float) -> None:
    if points and _same_point(points[-1], point, tol):
        # Prefer an actual source index/radius over a synthetic boundary sample.
        if points[-1].source_index is None and point.source_index is not None:
            points[-1].source_index = point.source_index
        if points[-1].radius is None and point.radius is not None:
            points[-1].radius = point.radius
        points[-1].source_arc = point.source_arc
        points[-1].xyz = point.xyz
        return
    points.append(point)


def _clip_edge_fragments(
    edge: Edge, lo: np.ndarray, hi: np.ndarray, tolerance: float, *, merge_closed_anchor: bool = True
) -> list[_Fragment]:
    path = np.asarray(edge.path_xyz, dtype=np.float64)
    if len(path) < 2:
        return []
    arc = arclength_profile(path)
    fragments: list[_Fragment] = []
    cur: list[_ClipPoint] = []
    for i in range(len(path) - 1):
        seglen = float(arc[i + 1] - arc[i])
        clipped = clip_segment_box(path[i], path[i + 1], lo, hi, tolerance=max(1e-15, tolerance * 0.1))
        if clipped is None:
            if len(cur) >= 2:
                fragments.append(_Fragment(cur))
            cur = []
            continue
        q0, q1, t0, t1 = clipped
        p0 = _clip_point(edge, i, q0, t0, arc, seglen, endpoint=1, tol=tolerance)
        p1 = _clip_point(edge, i, q1, t1, arc, seglen, endpoint=1, tol=tolerance)
        if cur and not _same_point(cur[-1], p0, tolerance):
            if len(cur) >= 2:
                fragments.append(_Fragment(cur))
            cur = []
        _append_point(cur, p0, tolerance)
        _append_point(cur, p1, tolerance)
        # If the original segment exits before its end, the fragment terminates now.
        if t1 < 1.0 - tolerance:
            if len(cur) >= 2:
                fragments.append(_Fragment(cur))
            cur = []
    if len(cur) >= 2:
        fragments.append(_Fragment(cur))

    # Closed paths can have first/last fragments that join through the arbitrary
    # source anchor.  Merge them so crop output is independent of that anchor.
    closed = bool(edge.is_self_loop or (len(path) >= 2 and np.allclose(path[0], path[-1], atol=tolerance, rtol=0.0)))
    if merge_closed_anchor and closed and len(fragments) >= 2 and _same_point(fragments[-1].points[-1], fragments[0].points[0], tolerance):
        merged = fragments[-1].points + fragments[0].points[1:]
        fragments = [_Fragment(merged), *fragments[1:-1]]
    return [f for f in fragments if len(f.points) >= 2 and float(np.linalg.norm(f.points[-1].xyz - f.points[0].xyz)) > tolerance or len(f.points) >= 3]


def _edge_radius_stats(profile: np.ndarray | None) -> tuple[float | None, float | None, float | None, float | None]:
    if profile is None or len(profile) == 0:
        return None, None, None, None
    return float(np.mean(profile)), float(np.median(profile)), float(np.min(profile)), float(np.max(profile))


def _node_key(xyz: np.ndarray, tol: float) -> tuple[int, int, int]:
    scale = max(tol, 1e-12)
    return tuple(int(round(float(v) / scale)) for v in xyz)  # type: ignore[return-value]


def _update_degrees(nodes: list[Node], edges: list[Edge]) -> None:
    deg = {n.id: 0 for n in nodes}
    for e in edges:
        if e.u == e.v:
            deg[e.u] = deg.get(e.u, 0) + 2
        else:
            deg[e.u] = deg.get(e.u, 0) + 1
            deg[e.v] = deg.get(e.v, 0) + 1
    for n in nodes:
        n.degree = int(deg.get(n.id, 0))


def _config_for_graph(graph: Mask2Graph, config: ExtractConfig | None) -> ExtractConfig:
    return config or ExtractConfig.from_dict(graph.meta.config)


def crop_topology_graph_box(
    graph: Mask2Graph,
    *,
    bounds: Sequence[float] | None = None,
    minimum: Sequence[float] | None = None,
    maximum: Sequence[float] | None = None,
    config: ExtractConfig | None = None,
    tolerance: float | None = None,
) -> Mask2Graph:
    """Clip complete source branch polylines to an axis-aligned 2D/3D box."""
    cfg = _config_for_graph(graph, config)
    tol = float(cfg.validation.tolerance if tolerance is None else tolerance)
    ndim = graph.meta.ndim
    lo, hi = normalize_box_bounds(ndim, bounds=bounds, minimum=minimum, maximum=maximum)
    source_hash = graph.meta.graph_hash or graph_content_hash(graph)

    nodes: list[Node] = []
    edges: list[Edge] = []
    coord_to_node: dict[tuple[int, int, int], int] = {}
    source_node_to_new: dict[int, int] = {}
    source_nodes = {n.id: n for n in graph.nodes}
    boundary_radii: dict[int, list[float]] = {}

    def get_node(point: _ClipPoint, source_node_id: int | None) -> int:
        xyz = np.asarray(point.xyz, dtype=np.float64)
        on_boundary = point_on_box_boundary(xyz, lo, hi, tol)
        if source_node_id is not None and source_node_id in source_node_to_new:
            nid = source_node_to_new[source_node_id]
            nodes[nid].on_crop_boundary = nodes[nid].on_crop_boundary or on_boundary
            return nid
        key = _node_key(xyz, tol)
        if key in coord_to_node:
            nid = coord_to_node[key]
            if source_node_id is not None:
                source_node_to_new[source_node_id] = nid
                # Prefer semantic source-node classification over synthetic crop endpoint.
                src = source_nodes[source_node_id]
                nodes[nid].type = src.type
                nodes[nid].source_index = src.source_index
                nodes[nid].index = src.index
                nodes[nid].index_valid = src.index_valid
                nodes[nid].created_by = src.created_by
                nodes[nid].on_image_boundary = src.on_image_boundary
                nodes[nid].boundary_axes = src.boundary_axes
            return nid

        if source_node_id is not None:
            src = deepcopy(source_nodes[source_node_id])
            src.id = len(nodes)
            src.xyz = tuple(float(v) for v in xyz)
            src.on_crop_boundary = bool(on_boundary)
            node = src
            source_node_to_new[source_node_id] = node.id
        else:
            idx = tuple(int(v) for v in point.source_index) if point.source_index is not None else tuple(-1 for _ in range(ndim))
            node = Node(
                id=len(nodes), xyz=tuple(float(v) for v in xyz), index=idx,
                type="boundary_endpoint", degree=0, voxel_count=0,
                radius_mean=point.radius, radius_median=point.radius, radius_min=point.radius, radius_max=point.radius,
                support_indices=None, junction_mst_edges=(), on_image_boundary=False, boundary_axes=(),
                source_index=None if point.source_index is None else tuple(int(v) for v in point.source_index),
                index_valid=point.source_index is not None,
                on_crop_boundary=True, created_by="crop",
            )
        nodes.append(node)
        coord_to_node[key] = node.id
        if point.radius is not None:
            boundary_radii.setdefault(node.id, []).append(float(point.radius))
        return node.id

    for source_edge in sorted(graph.edges, key=lambda e: e.id):
        merge_anchor = not (
            source_edge.u == source_edge.v
            and source_edge.u in source_nodes
            and source_nodes[source_edge.u].type != "cycle"
        )
        fragments = _clip_edge_fragments(source_edge, lo, hi, tol, merge_closed_anchor=merge_anchor)
        # Use geometry-derived arclength, not the rounded cached ``edge.length``,
        # when deciding whether a clipped endpoint is an original topology node.
        total = float(arclength_profile(np.asarray(source_edge.path_xyz, dtype=np.float64))[-1])
        for frag_idx, frag in enumerate(fragments):
            first, last = frag.points[0], frag.points[-1]
            u_src: int | None = None
            v_src: int | None = None
            if abs(first.source_arc) <= tol:
                u_src = source_edge.u
            elif abs(first.source_arc - total) <= tol:
                u_src = source_edge.v
            if abs(last.source_arc) <= tol:
                v_src = source_edge.u
            elif abs(last.source_arc - total) <= tol:
                v_src = source_edge.v
            u = get_node(first, u_src)
            v = get_node(last, v_src)
            path_xyz = np.vstack([p.xyz for p in frag.points]).astype(np.float64)
            path_index = np.asarray([
                p.source_index if p.source_index is not None else np.full(ndim, -1, dtype=np.int32)
                for p in frag.points
            ], dtype=np.int32)
            path_index_valid = bool(all(p.source_index is not None for p in frag.points) and graph.meta.grid_aligned)
            rp_vals = [p.radius for p in frag.points]
            radius_profile = None if any(vv is None for vv in rp_vals) else np.asarray(rp_vals, dtype=np.float64)
            rmean, rmedian, rmin, rmax = _edge_radius_stats(radius_profile)
            src_eid = source_edge.source_edge_id if source_edge.source_edge_id is not None else source_edge.id
            edge = Edge(
                id=len(edges), u=u, v=v, path_xyz=path_xyz, path_index=path_index,
                length=edge_length(path_xyz), voxel_length=len(path_xyz),
                radius_mean=rmean, radius_median=rmedian, radius_min=rmin, radius_max=rmax,
                radius_profile=radius_profile, is_self_loop=(u == v),
                provenance=f"{source_edge.provenance}|crop",
                source_path_index=path_index.copy(), path_index_valid=path_index_valid,
                source_edge_id=int(src_eid), source_arc_range=(float(first.source_arc), float(last.source_arc)),
                crop_fragment_index=frag_idx, created_by="crop",
            )
            edges.append(edge)

    # Preserve isolated source nodes that lie inside the crop.
    incident_source = {e.u for e in graph.edges} | {e.v for e in graph.edges}
    for src in graph.nodes:
        if src.id in incident_source or not point_in_box(np.asarray(src.xyz), lo, hi, tol):
            continue
        pt = _ClipPoint(np.asarray(src.xyz), np.asarray(src.index, dtype=np.int32) if src.index_valid else None, src.radius_mean, 0.0)
        get_node(pt, src.id)

    # Aggregate radius samples at shared synthetic boundary nodes.
    for nid, vals in boundary_radii.items():
        if not vals:
            continue
        n = nodes[nid]
        n.radius_mean = float(np.mean(vals))
        n.radius_median = float(np.median(vals))
        n.radius_min = float(np.min(vals))
        n.radius_max = float(np.max(vals))

    _update_degrees(nodes, edges)
    update_geometry_profiles(edges, tangent_window=cfg.geometry.tangent_window, compute_curvature=cfg.geometry.compute_curvature)
    simplify_graph_edges(edges, cfg.simplify)

    meta = deepcopy(graph.meta)
    if not meta.source_graph_hash:
        meta.source_graph_hash = source_hash
    meta.augmentation_history = [*meta.augmentation_history, {
        "type": "crop",
        "bounds": [float(x) for pair in zip(lo, hi) for x in pair],
        "transform_exact": bool(meta.transform_exact),
        "topology_changing": True,
        "input_graph_hash": source_hash,
    }]
    out = Mask2Graph(nodes=nodes, edges=edges, meta=meta)
    from .transforms import graph_is_grid_aligned
    out.meta.grid_aligned = bool(graph_is_grid_aligned(out))
    if not out.meta.grid_aligned:
        for n in out.nodes:
            if n.created_by == "crop" and n.source_index is None:
                n.index_valid = False
        for e in out.edges:
            if not e.path_index_valid:
                e.path_index_valid = False
    out.meta.graph_hash = graph_content_hash(out)
    out.meta.augmentation_history[-1]["output_graph_hash"] = out.meta.graph_hash
    return out


def crop_result_box(
    result: MaskGraphResult,
    *,
    bounds: Sequence[float] | None = None,
    minimum: Sequence[float] | None = None,
    maximum: Sequence[float] | None = None,
    config: ExtractConfig | None = None,
    tolerance: float | None = None,
) -> MaskGraphResult:
    cfg = _config_for_graph(result.topology_graph, config)
    topology = crop_topology_graph_box(
        result.topology_graph, bounds=bounds, minimum=minimum, maximum=maximum, config=cfg, tolerance=tolerance
    )
    diag = TopologyDiagnostics(
        cleaned_skeleton_edges=len(topology.edges),
        coverage_complete=True,
        topology_before_beta0=result.diagnostics.logical_beta0,
        topology_before_beta1=result.diagnostics.logical_beta1,
    )
    update_logical_diagnostics(diag, topology)
    diag.cleaned_beta0 = diag.logical_beta0
    diag.cleaned_beta1 = diag.logical_beta1
    diag.logical_vertices = len(topology.nodes)
    diag.logical_edges = len(topology.edges)
    diag.topology_after_beta0 = diag.logical_beta0
    diag.topology_after_beta1 = diag.logical_beta1
    topology.diagnostics = diag
    if cfg.validation.enabled:
        embedded = build_validated_embedded_graph(
            topology, diag, simplify_config=cfg.simplify, validation_config=cfg.validation
        )
    else:
        embedded = build_embedded_graph(topology, simplified=True)
        update_embedded_diagnostics(diag, embedded)
    topology.meta.augmentation_history[-1]["topology_before"] = [
        int(result.diagnostics.logical_beta0), int(result.diagnostics.logical_beta1)
    ]
    topology.meta.augmentation_history[-1]["topology_after"] = [int(diag.logical_beta0), int(diag.logical_beta1)]
    topology.meta.graph_hash = graph_content_hash(topology)
    topology.meta.augmentation_history[-1]["output_graph_hash"] = topology.meta.graph_hash
    embedded.metadata.update({
        "source_graph_hash": topology.meta.source_graph_hash,
        "graph_hash": topology.meta.graph_hash,
        "grid_aligned": topology.meta.grid_aligned,
        "transform_exact": topology.meta.transform_exact,
        "augmentation_history": list(topology.meta.augmentation_history),
    })
    if cfg.validation.enabled and cfg.validation.validate_embedding:
        validate_embedded_graph(embedded, tolerance=cfg.validation.tolerance)
        validate_embedded_topology(diag, embedded)
    return MaskGraphResult(topology_graph=topology, embedded_graph=embedded, diagnostics=diag, debug=None)


def validate_crop_result(result: MaskGraphResult, lo: np.ndarray, hi: np.ndarray, tolerance: float = 1e-9) -> None:
    """Crop-specific invariants; topology equality with the source is intentionally not required."""
    for edge in result.topology_graph.edges:
        if not np.all(edge.path_xyz[:, : len(lo)] >= lo - tolerance) or not np.all(edge.path_xyz[:, : len(lo)] <= hi + tolerance):
            raise ValidationError(f"cropped edge {edge.id} leaves crop box")
        if edge.source_edge_id is None:
            raise ValidationError(f"cropped edge {edge.id} lacks source edge provenance")
    for node in result.topology_graph.nodes:
        p = np.asarray(node.xyz)
        if not point_in_box(p, lo, hi, tolerance):
            raise ValidationError(f"cropped node {node.id} lies outside crop box")
        if node.on_crop_boundary and not point_on_box_boundary(p, lo, hi, tolerance):
            raise ValidationError(f"node {node.id} is flagged on_crop_boundary but is not on the current crop boundary")
