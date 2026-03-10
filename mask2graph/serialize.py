"""Stable JSON serialization for mask graphs."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import numpy as np

from .types import Edge, GraphMeta, Mask2Graph, Node, SerializationError

SCHEMA_VERSION = "1"


def to_dict(graph: Mask2Graph) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "meta": {
            **asdict(graph.meta),
        },
        "nodes": [
            {
                "id": n.id,
                "xyz": [float(v) for v in n.xyz],
                "index": [int(v) for v in n.index],
                "type": n.type,
                "degree": n.degree,
                "voxel_count": n.voxel_count,
                "radius_mean": n.radius_mean,
                "radius_median": n.radius_median,
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
                "length": float(e.length),
                "voxel_length": int(e.voxel_length),
                "radius_mean": e.radius_mean,
                "radius_median": e.radius_median,
                "radius_profile": None if e.radius_profile is None else e.radius_profile.tolist(),
                "arclen_profile": None if e.arclen_profile is None else e.arclen_profile.tolist(),
                "tangent_profile": None if e.tangent_profile is None else e.tangent_profile.tolist(),
                "is_self_loop": bool(e.is_self_loop),
            }
            for e in graph.edges
        ],
    }


def to_json(graph: Mask2Graph) -> str:
    return json.dumps(to_dict(graph), sort_keys=True, separators=(",", ":"))


def from_dict(payload: dict[str, Any]) -> Mask2Graph:
    if payload.get("schema_version") != SCHEMA_VERSION:
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
        )
        nodes = [
            Node(
                id=int(n["id"]),
                xyz=tuple(float(v) for v in n["xyz"]),  # type: ignore[arg-type]
                index=tuple(int(v) for v in n["index"]),
                type=str(n["type"]),
                degree=int(n["degree"]),
                voxel_count=int(n["voxel_count"]),
                radius_mean=None if n["radius_mean"] is None else float(n["radius_mean"]),
                radius_median=None if n["radius_median"] is None else float(n["radius_median"]),
            )
            for n in payload["nodes"]
        ]
        edges = []
        for e in payload["edges"]:
            rp = e.get("radius_profile")
            ap = e.get("arclen_profile")
            tp = e.get("tangent_profile")
            edges.append(
                Edge(
                    id=int(e["id"]),
                    u=int(e["u"]),
                    v=int(e["v"]),
                    path_xyz=np.asarray(e["path_xyz"], dtype=np.float64),
                    path_index=np.asarray(e["path_index"], dtype=np.int32),
                    length=float(e["length"]),
                    voxel_length=int(e["voxel_length"]),
                    radius_mean=None if e["radius_mean"] is None else float(e["radius_mean"]),
                    radius_median=None if e["radius_median"] is None else float(e["radius_median"]),
                    radius_profile=None if rp is None else np.asarray(rp, dtype=np.float64),
                    arclen_profile=None if ap is None else np.asarray(ap, dtype=np.float64),
                    tangent_profile=None if tp is None else np.asarray(tp, dtype=np.float64),
                    is_self_loop=bool(e["is_self_loop"]),
                )
            )
    except Exception as exc:  # noqa: BLE001
        raise SerializationError(f"Invalid graph schema: {exc}") from exc
    return Mask2Graph(nodes=nodes, edges=edges, meta=meta)


def from_json(raw: str) -> Mask2Graph:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SerializationError("Invalid JSON payload") from exc
    return from_dict(payload)
