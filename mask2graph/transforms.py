"""Deterministic rigid/isometric graph transforms for 2D and 3D graphs.

Low-level transforms operate only on graph geometry.  They never reskeletonize or
rasterize.  Raster indices remain provenance unless the transformed geometry is
still aligned to the original lattice, in which case current indices are updated.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np

from .topology import update_embedded_diagnostics, update_logical_diagnostics
from .types import EmbeddedGraph, EmbeddedSegment, EmbeddedVertex, Mask2Graph, MaskGraphResult, TopologyDiagnostics, ValidationError
from .utils.coords import xyz_to_index, xyzs_to_indices
from .utils.hash import graph_content_hash
from .validate import build_embedded_graph, validate_embedded_graph, validate_embedded_topology

_TOL = 1e-9


def _as3(v: Sequence[float], ndim: int) -> np.ndarray:
    vals = np.asarray(tuple(float(x) for x in v), dtype=np.float64).reshape(-1)
    if len(vals) != ndim:
        raise ValueError(f"expected {ndim} coordinates, got {len(vals)}")
    out = np.zeros(3, dtype=np.float64)
    out[:ndim] = vals
    return out


def _matrix3(matrix: np.ndarray | Sequence[Sequence[float]], ndim: int) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    if ndim == 2:
        if arr.shape == (2, 2):
            out = np.eye(3, dtype=np.float64)
            out[:2, :2] = arr
            return out
        if arr.shape == (3, 3):
            if not np.allclose(arr[2], [0.0, 0.0, 1.0], atol=_TOL) or not np.allclose(arr[:, 2], [0.0, 0.0, 1.0], atol=_TOL):
                raise ValueError("2D transform matrix must preserve z=0")
            return arr.copy()
        raise ValueError("2D matrix must be 2x2 or z-preserving 3x3")
    if arr.shape != (3, 3):
        raise ValueError("3D matrix must be 3x3")
    return arr.copy()


def _signed_permutation(matrix: np.ndarray, ndim: int, tol: float = 1e-9) -> bool:
    m = np.asarray(matrix, dtype=np.float64)[:ndim, :ndim]
    rounded = np.rint(m)
    if not np.allclose(m, rounded, atol=tol):
        return False
    if not np.all(np.isin(rounded, (-1.0, 0.0, 1.0))):
        return False
    return bool(np.all(np.sum(np.abs(rounded), axis=0) == 1) and np.all(np.sum(np.abs(rounded), axis=1) == 1))


def _orthogonal(matrix: np.ndarray, ndim: int, tol: float = 1e-9) -> bool:
    m = matrix[:ndim, :ndim]
    return bool(np.allclose(m.T @ m, np.eye(ndim), atol=tol))


def _bbox_center(graph: Mask2Graph) -> np.ndarray:
    pts: list[np.ndarray] = []
    pts.extend(np.asarray(n.xyz, dtype=np.float64) for n in graph.nodes)
    pts.extend(np.asarray(e.path_xyz, dtype=np.float64) for e in graph.edges if len(e.path_xyz))
    if not pts:
        return np.zeros(3, dtype=np.float64)
    flat = np.vstack([p.reshape(-1, 3) if p.ndim == 2 else p.reshape(1, 3) for p in pts])
    return (flat.min(axis=0) + flat.max(axis=0)) / 2.0


def _centroid(graph: Mask2Graph) -> np.ndarray:
    if graph.nodes:
        return np.mean(np.asarray([n.xyz for n in graph.nodes], dtype=np.float64), axis=0)
    pts = [p for e in graph.edges for p in np.asarray(e.path_xyz, dtype=np.float64)]
    return np.mean(np.asarray(pts), axis=0) if pts else np.zeros(3, dtype=np.float64)


def resolve_center(graph: Mask2Graph, center: str | Sequence[float] = "bbox_center") -> np.ndarray:
    if isinstance(center, str):
        if center == "bbox_center":
            return _bbox_center(graph)
        if center == "centroid":
            return _centroid(graph)
        if center == "origin":
            return np.zeros(3, dtype=np.float64)
        raise ValueError("center must be 'bbox_center', 'centroid', 'origin', or an explicit coordinate")
    return _as3(center, graph.meta.ndim)


@dataclass(frozen=True)
class GraphTransform:
    """Affine transform ``x' = A x + b`` with isometry metadata."""

    matrix: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    translation: tuple[float, float, float]
    exact: bool
    name: str = "transform"
    parameters: tuple[tuple[str, object], ...] = ()

    @property
    def A(self) -> np.ndarray:
        return np.asarray(self.matrix, dtype=np.float64)

    @property
    def b(self) -> np.ndarray:
        return np.asarray(self.translation, dtype=np.float64)

    def apply_points(self, points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=np.float64)
        return p @ self.A.T + self.b

    def __matmul__(self, other: "GraphTransform") -> "GraphTransform":
        """Composition operator: ``T2 @ T1`` means ``T2(T1(x))``."""
        A = self.A @ other.A
        b = self.A @ other.b + self.b
        return GraphTransform(
            matrix=tuple(tuple(float(v) for v in row) for row in A),  # type: ignore[arg-type]
            translation=tuple(float(v) for v in b),
            exact=bool(self.exact and other.exact),
            name="composition",
            parameters=(("sequence", [other.to_record(), self.to_record()]),),
        )

    def inverse(self) -> "GraphTransform":
        Ainv = np.linalg.inv(self.A)
        if self.exact and _signed_permutation(Ainv, 3):
            Ainv = np.rint(Ainv).astype(np.float64)
        binv = -Ainv @ self.b
        if self.exact:
            binv = np.round(binv, 12)
        return GraphTransform(
            matrix=tuple(tuple(float(v) for v in row) for row in Ainv),  # type: ignore[arg-type]
            translation=tuple(float(v) for v in binv),
            exact=bool(self.exact),
            name=f"inverse({self.name})",
            parameters=(("inverse_of", self.name),),
        )

    def to_record(self) -> dict:
        return {
            "type": self.name,
            "matrix": [[float(v) for v in row] for row in self.matrix],
            "translation": [float(v) for v in self.translation],
            "transform_exact": bool(self.exact),
            "parameters": {k: v for k, v in self.parameters},
        }


def affine_transform(
    matrix: np.ndarray | Sequence[Sequence[float]],
    translation: Sequence[float],
    *,
    ndim: int,
    exact: bool | None = None,
    name: str = "affine",
    parameters: dict[str, object] | None = None,
    require_isometry: bool = True,
) -> GraphTransform:
    A = _matrix3(matrix, ndim)
    b = _as3(translation, ndim)
    if require_isometry and not _orthogonal(A, ndim):
        raise ValueError("augmentation V1 accepts only rigid/isometric affine transforms")
    if exact is None:
        exact = _signed_permutation(A, ndim)
    return GraphTransform(
        matrix=tuple(tuple(float(v) for v in row) for row in A),  # type: ignore[arg-type]
        translation=tuple(float(v) for v in b),
        exact=bool(exact),
        name=name,
        parameters=tuple(sorted((parameters or {}).items())),
    )


def _about_center(A: np.ndarray, center: np.ndarray) -> np.ndarray:
    return center - A @ center


def rotation_transform(
    graph: Mask2Graph,
    angle: float | None = None,
    *,
    center: str | Sequence[float] = "bbox_center",
    degrees: bool = True,
    axis: Sequence[float] | None = None,
    quaternion: Sequence[float] | None = None,
    matrix: Sequence[Sequence[float]] | None = None,
) -> GraphTransform:
    ndim = graph.meta.ndim
    c = resolve_center(graph, center)
    if ndim == 2:
        if axis is not None or quaternion is not None:
            raise ValueError("2D rotation does not accept axis/quaternion")
        if matrix is not None:
            A = _matrix3(matrix, 2)
            if not _orthogonal(A, 2) or np.linalg.det(A[:2, :2]) < 0.0:
                raise ValueError("2D rotation matrix must be orientation-preserving and orthogonal")
            exact = _signed_permutation(A, 2)
            desc = {"center": c[:2].tolist(), "matrix_supplied": True}
        else:
            if angle is None:
                raise ValueError("2D rotation requires angle or matrix")
            theta = math.radians(float(angle)) if degrees else float(angle)
            quarter = (float(angle) / 90.0) if degrees else (float(angle) / (math.pi / 2.0))
            exact = abs(quarter - round(quarter)) <= 1e-12
            if exact:
                k = int(round(quarter)) % 4
                mats = (
                    np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
                    np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
                    np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]),
                    np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
                )
                A = mats[k].astype(np.float64, copy=True)
            else:
                ct, st = math.cos(theta), math.sin(theta)
                A = np.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            desc = {"angle": float(angle), "degrees": bool(degrees), "center": c[:2].tolist()}
        b = _about_center(A, c)
        return affine_transform(A, b[:2], ndim=2, exact=exact, name="rotation", parameters=desc)

    if matrix is not None:
        A = _matrix3(matrix, 3)
    elif quaternion is not None:
        q = np.asarray(tuple(float(v) for v in quaternion), dtype=np.float64)
        if q.shape != (4,):
            raise ValueError("quaternion must be (w,x,y,z)")
        norm = float(np.linalg.norm(q))
        if norm <= 0:
            raise ValueError("quaternion must be nonzero")
        w, x, y, z = q / norm
        A = np.array(
            [
                [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
                [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
                [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
            ], dtype=np.float64,
        )
    else:
        if angle is None or axis is None:
            raise ValueError("3D rotation requires matrix, quaternion, or axis+angle")
        u = np.asarray(tuple(float(v) for v in axis), dtype=np.float64)
        if u.shape != (3,) or float(np.linalg.norm(u)) <= 0:
            raise ValueError("axis must be a nonzero 3-vector")
        u = u / np.linalg.norm(u)
        theta = math.radians(float(angle)) if degrees else float(angle)
        x, y, z = u
        K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
        A = np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)
    if not _orthogonal(A, 3) or np.linalg.det(A) < 0.0:
        raise ValueError("3D rotation must be orientation-preserving and orthogonal")
    exact = _signed_permutation(A, 3)
    if exact:
        A = np.rint(A).astype(np.float64)
    b = _about_center(A, c)
    params: dict[str, object] = {"center": c.tolist()}
    if angle is not None:
        params.update({"angle": float(angle), "degrees": bool(degrees)})
    if axis is not None:
        params["axis"] = [float(v) for v in axis]
    if quaternion is not None:
        params["quaternion"] = [float(v) for v in quaternion]
    if matrix is not None:
        params["matrix_supplied"] = True
    return affine_transform(A, b, ndim=3, exact=exact, name="rotation", parameters=params)


def flip_transform(
    graph: Mask2Graph,
    axis: str | Sequence[str],
    *,
    center: str | Sequence[float] = "bbox_center",
) -> GraphTransform:
    ndim = graph.meta.ndim
    axes = list(axis) if isinstance(axis, str) and all(ch in "xyz" for ch in axis) else list(axis if not isinstance(axis, str) else [axis])
    allowed = ["x", "y"] if ndim == 2 else ["x", "y", "z"]
    if not axes or any(a not in allowed for a in axes):
        raise ValueError(f"flip axes must be drawn from {allowed}")
    c = resolve_center(graph, center)
    A = np.eye(3, dtype=np.float64)
    for a in axes:
        A["xyz".index(a), "xyz".index(a)] *= -1.0
    b = _about_center(A, c)
    return affine_transform(
        A, b[:ndim], ndim=ndim, exact=True, name="flip",
        parameters={"axes": axes, "center": c[:ndim].tolist()},
    )


def translation_transform(graph: Mask2Graph, offset: Sequence[float]) -> GraphTransform:
    ndim = graph.meta.ndim
    b = _as3(offset, ndim)
    return affine_transform(
        np.eye(3), b[:ndim], ndim=ndim, exact=True, name="translation",
        parameters={"offset": [float(v) for v in b[:ndim]]},
    )


def axis_permutation_transform(graph: Mask2Graph, order: str | Sequence[int]) -> GraphTransform:
    ndim = graph.meta.ndim
    if isinstance(order, str):
        expected = "xy" if ndim == 2 else "xyz"
        if len(order) != ndim or sorted(order) != sorted(expected):
            raise ValueError(f"axis order must be a permutation of {expected!r}")
        perm = [expected.index(ch) for ch in order]
    else:
        perm = [int(v) for v in order]
        if sorted(perm) != list(range(ndim)):
            raise ValueError("axis permutation is invalid")
    A = np.eye(3, dtype=np.float64)
    A[:ndim, :ndim] = 0.0
    for out_axis, in_axis in enumerate(perm):
        A[out_axis, in_axis] = 1.0
    return affine_transform(A, np.zeros(ndim), ndim=ndim, exact=True, name="axis_permutation", parameters={"order": perm})


def compose_transforms(transforms: Iterable[GraphTransform]) -> GraphTransform:
    """Compose transforms in application order.

    ``compose_transforms([T1, T2])`` is the transform ``T2(T1(x))``.
    """
    items = list(transforms)
    if not items:
        return GraphTransform(
            matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            translation=(0.0, 0.0, 0.0), exact=True, name="identity",
        )
    A = np.eye(3, dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    exact = True
    for t in items:
        A = t.A @ A
        b = t.A @ b + t.b
        exact = exact and t.exact
    return GraphTransform(
        matrix=tuple(tuple(float(v) for v in row) for row in A),  # type: ignore[arg-type]
        translation=tuple(float(v) for v in b),
        exact=bool(exact),
        name="composition",
        parameters=(("sequence", [t.to_record() for t in items]),),
    )


def _iter_geometry(graph: Mask2Graph) -> Iterable[np.ndarray]:
    for node in graph.nodes:
        yield np.asarray(node.xyz, dtype=np.float64).reshape(1, 3)
    for edge in graph.edges:
        if len(edge.path_xyz):
            yield np.asarray(edge.path_xyz, dtype=np.float64)


def graph_is_grid_aligned(graph: Mask2Graph, tolerance: float = 1e-7) -> bool:
    ndim = graph.meta.ndim
    spacing = graph.meta.spacing
    for arr in _iter_geometry(graph):
        idx = xyzs_to_indices(arr, spacing, ndim)
        from .utils.coords import indices_to_xyz
        back = indices_to_xyz(idx, spacing)
        if not np.allclose(arr, back, atol=tolerance, rtol=0.0):
            return False
    return True


def _update_indices_from_geometry(graph: Mask2Graph) -> None:
    ndim = graph.meta.ndim
    if not graph.meta.grid_aligned:
        for n in graph.nodes:
            n.index_valid = False
        for e in graph.edges:
            e.path_index_valid = False
        return
    for n in graph.nodes:
        n.index = xyz_to_index(np.asarray(n.xyz), graph.meta.spacing, ndim)
        n.index_valid = True
    for e in graph.edges:
        e.path_index = xyzs_to_indices(e.path_xyz, graph.meta.spacing, ndim)
        e.path_index_valid = True
        if e.simplified_xyz is not None:
            e.simplified_indices = xyzs_to_indices(e.simplified_xyz, graph.meta.spacing, ndim)


def _transform_topology_graph(graph: Mask2Graph, transform: GraphTransform) -> Mask2Graph:
    out = deepcopy(graph)
    if not _orthogonal(transform.A, out.meta.ndim):
        raise ValueError("rigid graph augmentation requires an orthogonal transform")
    before_hash = graph.meta.graph_hash or graph_content_hash(graph)
    if not out.meta.source_graph_hash:
        out.meta.source_graph_hash = before_hash
    for node in out.nodes:
        node.xyz = tuple(float(v) for v in transform.apply_points(np.asarray(node.xyz).reshape(1, 3))[0])
    for edge in out.edges:
        edge.path_xyz = transform.apply_points(edge.path_xyz)
        if edge.simplified_xyz is not None:
            edge.simplified_xyz = transform.apply_points(edge.simplified_xyz)
        if edge.tangent_profile is not None:
            edge.tangent_profile = np.asarray(edge.tangent_profile) @ transform.A.T
        # Rigid/isometric transforms preserve all scalar profiles exactly in theory.
    if transform.exact:
        for node in out.nodes:
            node.xyz = tuple(float(v) for v in np.round(np.asarray(node.xyz), 12))
        for edge in out.edges:
            edge.path_xyz = np.round(edge.path_xyz, 12)
            if edge.simplified_xyz is not None:
                edge.simplified_xyz = np.round(edge.simplified_xyz, 12)
            if edge.tangent_profile is not None:
                edge.tangent_profile = np.round(edge.tangent_profile, 12)
    out.meta.transform_exact = bool(out.meta.transform_exact and transform.exact)
    record = {**transform.to_record(), "input_graph_hash": before_hash}
    out.meta.augmentation_history = [*out.meta.augmentation_history, record]
    # Grid alignment is a property of the resulting geometry, not merely the transform class.
    out.meta.grid_aligned = bool(graph_is_grid_aligned(out))
    _update_indices_from_geometry(out)
    out.meta.graph_hash = graph_content_hash(out)
    out.meta.augmentation_history[-1]["output_graph_hash"] = out.meta.graph_hash
    return out


def _transform_embedded_graph(embedded: EmbeddedGraph, transform: GraphTransform, *, graph_hash: str, grid_aligned: bool, transform_exact: bool, history: list[dict]) -> EmbeddedGraph:
    vertices = [
        EmbeddedVertex(
            id=v.id,
            xyz=tuple(float(x) for x in transform.apply_points(np.asarray(v.xyz).reshape(1, 3))[0]),
            source_node_id=v.source_node_id,
            source_edge_id=v.source_edge_id,
            source_path_position=v.source_path_position,
            kind=v.kind,
            on_crop_boundary=v.on_crop_boundary,
            created_by=v.created_by,
        )
        for v in embedded.vertices
    ]
    segments = [
        EmbeddedSegment(
            id=s.id, u=s.u, v=s.v, source_edge_id=s.source_edge_id, source_arc_range=s.source_arc_range
        ) for s in embedded.segments
    ]
    meta = dict(embedded.metadata)
    meta.update({
        "graph_hash": graph_hash,
        "grid_aligned": bool(grid_aligned),
        "transform_exact": bool(transform_exact),
        "augmentation_history": list(history),
    })
    return EmbeddedGraph(ndim=embedded.ndim, vertices=vertices, segments=segments, diagnostics=embedded.diagnostics, metadata=meta)


def _topology_signature(graph: Mask2Graph) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    diag = TopologyDiagnostics()
    update_logical_diagnostics(diag, graph)
    pairs = tuple(sorted((min(e.u, e.v), max(e.u, e.v)) for e in graph.edges))
    return diag.logical_beta0, diag.logical_beta1, pairs


def apply_transform(obj: Mask2Graph | MaskGraphResult, transform: GraphTransform, *, validate: bool = True, tolerance: float = 1e-8):
    """Apply a rigid transform to a topology graph or full paired result."""
    if isinstance(obj, MaskGraphResult):
        before_sig = _topology_signature(obj.topology_graph)
        topology = _transform_topology_graph(obj.topology_graph, transform)
        embedded = _transform_embedded_graph(
            obj.embedded_graph, transform,
            graph_hash=topology.meta.graph_hash,
            grid_aligned=topology.meta.grid_aligned,
            transform_exact=topology.meta.transform_exact,
            history=topology.meta.augmentation_history,
        )
        diag = deepcopy(obj.diagnostics)
        update_logical_diagnostics(diag, topology)
        update_embedded_diagnostics(diag, embedded)
        diag.topology_before_beta0 = obj.diagnostics.logical_beta0
        diag.topology_before_beta1 = obj.diagnostics.logical_beta1
        diag.topology_after_beta0 = diag.logical_beta0
        diag.topology_after_beta1 = diag.logical_beta1
        topology.diagnostics = diag
        embedded.diagnostics = diag
        if validate:
            after_sig = _topology_signature(topology)
            if before_sig != after_sig:
                raise ValidationError(f"rigid augmentation changed topology: {before_sig} -> {after_sig}")
            # A rigid/isometric transform is bijective and preserves segment
            # intersections.  Once the source embedding has been validated, an
            # O(E^2) intersection rescan after every rotate/flip would be
            # redundant and becomes prohibitive on native vessel/road graphs.
            # Unvalidated external results still pay the full validation once.
            if not bool(obj.embedded_graph.metadata.get("embedding_validated", False)):
                validate_embedded_graph(obj.embedded_graph, tolerance=tolerance)
            embedded.metadata["embedding_validated"] = True
            validate_embedded_topology(diag, embedded)
            # Distances/lengths are invariant under rigid transforms.
            before_lengths = [float(e.length) for e in obj.topology_graph.edges]
            after_lengths = [float(e.length) for e in topology.edges]
            if len(before_lengths) != len(after_lengths) or not np.allclose(before_lengths, after_lengths, atol=tolerance, rtol=0.0):
                raise ValidationError("rigid augmentation changed stored branch lengths")
            for before_edge, after_edge in zip(obj.topology_graph.edges, topology.edges):
                for attr in ("radius_mean", "radius_median", "radius_min", "radius_max", "chord_length", "tortuosity"):
                    a, b = getattr(before_edge, attr), getattr(after_edge, attr)
                    if a is None or b is None:
                        if a is not b:
                            raise ValidationError(f"rigid augmentation changed {attr} presence")
                    elif not math.isclose(float(a), float(b), abs_tol=tolerance, rel_tol=0.0):
                        raise ValidationError(f"rigid augmentation changed scalar geometry attribute {attr}")
                if before_edge.radius_profile is not None or after_edge.radius_profile is not None:
                    if before_edge.radius_profile is None or after_edge.radius_profile is None or not np.allclose(
                        before_edge.radius_profile, after_edge.radius_profile, atol=tolerance, rtol=0.0
                    ):
                        raise ValidationError("rigid augmentation changed radius profile")
        return MaskGraphResult(topology_graph=topology, embedded_graph=embedded, diagnostics=diag, debug=None)

    before_sig = _topology_signature(obj)
    topology = _transform_topology_graph(obj, transform)
    if validate:
        if before_sig != _topology_signature(topology):
            raise ValidationError("rigid augmentation changed graph topology")
        embedded = build_embedded_graph(topology, simplified=True)
        validate_embedded_graph(embedded, tolerance=tolerance)
    return topology


def rotate_graph(obj: Mask2Graph | MaskGraphResult, angle: float | None = None, *, center: str | Sequence[float] = "bbox_center", degrees: bool = True, axis: Sequence[float] | None = None, quaternion: Sequence[float] | None = None, matrix: Sequence[Sequence[float]] | None = None, validate: bool = True):
    graph = obj.topology_graph if isinstance(obj, MaskGraphResult) else obj
    return apply_transform(obj, rotation_transform(graph, angle, center=center, degrees=degrees, axis=axis, quaternion=quaternion, matrix=matrix), validate=validate)


def flip_graph(obj: Mask2Graph | MaskGraphResult, axis: str | Sequence[str], *, center: str | Sequence[float] = "bbox_center", validate: bool = True):
    graph = obj.topology_graph if isinstance(obj, MaskGraphResult) else obj
    return apply_transform(obj, flip_transform(graph, axis, center=center), validate=validate)


def translate_graph(obj: Mask2Graph | MaskGraphResult, offset: Sequence[float], *, validate: bool = True):
    graph = obj.topology_graph if isinstance(obj, MaskGraphResult) else obj
    return apply_transform(obj, translation_transform(graph, offset), validate=validate)


def permute_axes(obj: Mask2Graph | MaskGraphResult, order: str | Sequence[int], *, validate: bool = True):
    graph = obj.topology_graph if isinstance(obj, MaskGraphResult) else obj
    return apply_transform(obj, axis_permutation_transform(graph, order), validate=validate)
