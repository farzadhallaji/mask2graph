"""Stable JSON serialization for mask graphs."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import numpy as np

from .types import Edge, GraphMeta, Mask2Graph, Node, SerializationError, TopologyDiagnostics

SCHEMA_VERSION = "3"
SUPPORTED_SCHEMA_VERSIONS = {"1", "2", "3"}


def _arr(value):
    return None if value is None else np.asarray(value).tolist()


def to_dict(graph: Mask2Graph) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "meta": {**asdict(graph.meta)},
        "diagnostics": None if graph.diagnostics is None else asdict(graph.diagnostics),
        "nodes": [
            {
                "id": n.id,
                "xyz": [float(v) for v in n.xyz],
                "index": [int(v) for v in n.index],
                "source_index": None if n.source_index is None else [int(v) for v in n.source_index],
                "index_valid": bool(n.index_valid),
                "type": n.type,
                "degree": n.degree,
                "voxel_count": n.voxel_count,
                "radius_mean": n.radius_mean,
                "radius_median": n.radius_median,
                "radius_min": n.radius_min,
                "radius_max": n.radius_max,
                "support_indices": _arr(n.support_indices),
                "junction_mst_edges": [[list(a), list(b)] for a, b in n.junction_mst_edges],
                "on_image_boundary": n.on_image_boundary,
                "boundary_axes": list(n.boundary_axes),
                "on_crop_boundary": bool(n.on_crop_boundary),
                "created_by": n.created_by,
            }
            for n in graph.nodes
        ],
        "edges": [
            {
                "id": e.id,
                "u": e.u,
                "v": e.v,
                "path_xyz": e.path_xyz.tolist(),
                "path_index": e.path_index.tolist(),
                "source_path_index": _arr(e.source_path_index),
                "path_index_valid": bool(e.path_index_valid),
                "length": float(e.length),
                "voxel_length": int(e.voxel_length),
                "radius_mean": e.radius_mean,
                "radius_median": e.radius_median,
                "radius_min": e.radius_min,
                "radius_max": e.radius_max,
                "radius_profile": _arr(e.radius_profile),
                "arclen_profile": _arr(e.arclen_profile),
                "tangent_profile": _arr(e.tangent_profile),
                "curvature_profile": _arr(e.curvature_profile),
                "turning_angle_profile": _arr(e.turning_angle_profile),
                "simplified_indices": _arr(e.simplified_indices),
                "simplified_xyz": _arr(e.simplified_xyz),
                "chord_length": float(e.chord_length),
                "tortuosity": float(e.tortuosity),
                "is_self_loop": bool(e.is_self_loop),
                "provenance": e.provenance,
                "source_edge_id": e.source_edge_id,
                "source_arc_range": None if e.source_arc_range is None else list(e.source_arc_range),
                "crop_fragment_index": e.crop_fragment_index,
                "created_by": e.created_by,
            }
            for e in graph.edges
        ],
    }


def to_json(graph: Mask2Graph) -> str:
    return json.dumps(to_dict(graph), sort_keys=True, separators=(",", ":"))


def _opt_float(obj: dict, key: str) -> float | None:
    value = obj.get(key)
    return None if value is None else float(value)


def from_dict(payload: dict[str, Any]) -> Mask2Graph:
    if str(payload.get("schema_version")) not in SUPPORTED_SCHEMA_VERSIONS:
        raise SerializationError("Unsupported schema version")
    try:
        meta_obj = payload["meta"]
        meta = GraphMeta(
            version=str(meta_obj["version"]),
            ndim=int(meta_obj["ndim"]),
            shape=tuple(int(v) for v in meta_obj["shape"]),
            spacing=tuple(float(v) for v in meta_obj["spacing"]),
            config=dict(meta_obj["config"]),
            input_hash=str(meta_obj["input_hash"]),
            processed_mask_hash=str(meta_obj["processed_mask_hash"]),
            config_hash=str(meta_obj.get("config_hash", "")),
            graph_hash=str(meta_obj.get("graph_hash", "")),
            source_graph_hash=str(meta_obj.get("source_graph_hash", "")),
            grid_aligned=bool(meta_obj.get("grid_aligned", True)),
            transform_exact=bool(meta_obj.get("transform_exact", True)),
            rationalization_error=float(meta_obj.get("rationalization_error", 0.0)),
            augmentation_history=list(meta_obj.get("augmentation_history", [])),
            augmentation_seed=None if meta_obj.get("augmentation_seed") is None else int(meta_obj["augmentation_seed"]),
        )
        nodes = []
        for n in payload["nodes"]:
            si = n.get("support_indices")
            mst = tuple(
                (tuple(int(v) for v in a), tuple(int(v) for v in b))
                for a, b in n.get("junction_mst_edges", [])
            )
            src_idx = n.get("source_index")
            nodes.append(
                Node(
                    id=int(n["id"]),
                    xyz=tuple(float(v) for v in n["xyz"]),  # type: ignore[arg-type]
                    index=tuple(int(v) for v in n["index"]),
                    type=str(n["type"]),
                    degree=int(n["degree"]),
                    voxel_count=int(n["voxel_count"]),
                    radius_mean=_opt_float(n, "radius_mean"),
                    radius_median=_opt_float(n, "radius_median"),
                    radius_min=_opt_float(n, "radius_min"),
                    radius_max=_opt_float(n, "radius_max"),
                    support_indices=None if si is None else np.asarray(si, dtype=np.int32),
                    junction_mst_edges=mst,
                    on_image_boundary=bool(n.get("on_image_boundary", False)),
                    boundary_axes=tuple(int(v) for v in n.get("boundary_axes", ())),
                    source_index=None if src_idx is None else tuple(int(v) for v in src_idx),
                    index_valid=bool(n.get("index_valid", True)),
                    on_crop_boundary=bool(n.get("on_crop_boundary", False)),
                    created_by=n.get("created_by"),
                )
            )
        edges = []
        for e in payload["edges"]:
            def arr(name: str, dtype=np.float64):
                value = e.get(name)
                return None if value is None else np.asarray(value, dtype=dtype)

            arc = e.get("source_arc_range")
            edges.append(
                Edge(
                    id=int(e["id"]),
                    u=int(e["u"]),
                    v=int(e["v"]),
                    path_xyz=np.asarray(e["path_xyz"], dtype=np.float64),
                    path_index=np.asarray(e["path_index"], dtype=np.int32),
                    length=float(e["length"]),
                    voxel_length=int(e["voxel_length"]),
                    radius_mean=_opt_float(e, "radius_mean"),
                    radius_median=_opt_float(e, "radius_median"),
                    radius_min=_opt_float(e, "radius_min"),
                    radius_max=_opt_float(e, "radius_max"),
                    radius_profile=arr("radius_profile"),
                    arclen_profile=arr("arclen_profile"),
                    tangent_profile=arr("tangent_profile"),
                    curvature_profile=arr("curvature_profile"),
                    turning_angle_profile=arr("turning_angle_profile"),
                    simplified_indices=arr("simplified_indices", np.int32),
                    simplified_xyz=arr("simplified_xyz"),
                    chord_length=float(e.get("chord_length", 0.0)),
                    tortuosity=float(e.get("tortuosity", 1.0)),
                    is_self_loop=bool(e["is_self_loop"]),
                    provenance=str(e.get("provenance", "skeleton_trace")),
                    source_path_index=arr("source_path_index", np.int32),
                    path_index_valid=bool(e.get("path_index_valid", True)),
                    source_edge_id=None if e.get("source_edge_id") is None else int(e["source_edge_id"]),
                    source_arc_range=None if arc is None else (float(arc[0]), float(arc[1])),
                    crop_fragment_index=None if e.get("crop_fragment_index") is None else int(e["crop_fragment_index"]),
                    created_by=e.get("created_by"),
                )
            )
        diag_obj = payload.get("diagnostics")
        diagnostics = None if diag_obj is None else TopologyDiagnostics(**diag_obj)
    except Exception as exc:  # noqa: BLE001
        raise SerializationError(f"Invalid graph schema: {exc}") from exc
    return Mask2Graph(nodes=nodes, edges=edges, meta=meta, diagnostics=diagnostics)


def from_json(raw: str) -> Mask2Graph:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SerializationError("Invalid JSON payload") from exc
    return from_dict(payload)


def embedded_to_dict(embedded) -> dict[str, Any]:
    return {
        "ndim": int(embedded.ndim),
        "metadata": dict(getattr(embedded, "metadata", {})),
        "vertices": [
            {
                "id": int(v.id),
                "xyz": [float(x) for x in v.xyz],
                "source_node_id": v.source_node_id,
                "source_edge_id": v.source_edge_id,
                "source_path_position": v.source_path_position,
                "kind": v.kind,
                "on_crop_boundary": bool(getattr(v, "on_crop_boundary", False)),
                "created_by": getattr(v, "created_by", None),
            }
            for v in embedded.vertices
        ],
        "segments": [
            {
                "id": int(s.id),
                "u": int(s.u),
                "v": int(s.v),
                "source_edge_id": int(s.source_edge_id),
                "source_arc_range": None if s.source_arc_range is None else list(s.source_arc_range),
            }
            for s in embedded.segments
        ],
    }


def result_to_dict(result) -> dict[str, Any]:
    return {
        "schema_version": "mask2graph_result_v2",
        "topology_graph": to_dict(result.topology_graph),
        "embedded_graph": embedded_to_dict(result.embedded_graph),
        "diagnostics": asdict(result.diagnostics),
    }


def result_to_json(result) -> str:
    return json.dumps(result_to_dict(result), sort_keys=True, separators=(",", ":"))


def embedded_from_dict(payload: dict[str, Any]):
    from .types import EmbeddedGraph, EmbeddedSegment, EmbeddedVertex
    vertices = [
        EmbeddedVertex(
            id=int(v["id"]),
            xyz=tuple(float(x) for x in v["xyz"]),
            source_node_id=None if v.get("source_node_id") is None else int(v["source_node_id"]),
            source_edge_id=None if v.get("source_edge_id") is None else int(v["source_edge_id"]),
            source_path_position=None if v.get("source_path_position") is None else int(v["source_path_position"]),
            kind=str(v["kind"]),
            on_crop_boundary=bool(v.get("on_crop_boundary", False)),
            created_by=v.get("created_by"),
        )
        for v in payload.get("vertices", [])
    ]
    segments = []
    for s in payload.get("segments", []):
        arc = s.get("source_arc_range")
        segments.append(
            EmbeddedSegment(
                id=int(s["id"]), u=int(s["u"]), v=int(s["v"]), source_edge_id=int(s["source_edge_id"]),
                source_arc_range=None if arc is None else (float(arc[0]), float(arc[1])),
            )
        )
    return EmbeddedGraph(
        ndim=int(payload["ndim"]), vertices=vertices, segments=segments,
        metadata=dict(payload.get("metadata", {})),
    )


def result_from_dict(payload: dict[str, Any]):
    from .types import MaskGraphResult
    if payload.get("schema_version") not in {"mask2graph_result_v1", "mask2graph_result_v2"}:
        raise SerializationError("Unsupported MaskGraphResult schema version")
    topology = from_dict(payload["topology_graph"])
    embedded = embedded_from_dict(payload["embedded_graph"])
    diag_obj = payload.get("diagnostics")
    diagnostics = topology.diagnostics if diag_obj is None else TopologyDiagnostics(**diag_obj)
    if diagnostics is None:
        diagnostics = TopologyDiagnostics()
    topology.diagnostics = diagnostics
    embedded.diagnostics = diagnostics
    return MaskGraphResult(topology_graph=topology, embedded_graph=embedded, diagnostics=diagnostics, debug=None)


def result_from_json(raw: str):
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SerializationError("Invalid JSON payload") from exc
    return result_from_dict(payload)
