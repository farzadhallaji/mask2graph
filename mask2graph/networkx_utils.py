"""NetworkX interoperability helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .types import Mask2Graph

if TYPE_CHECKING:
    import networkx as nx


def _to_list(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def to_networkx(graph: Mask2Graph, *, multigraph: bool = True) -> "nx.Graph":
    """Convert a ``Mask2Graph`` instance to a NetworkX graph.

    Parameters
    ----------
    graph:
        Input ``Mask2Graph`` object.
    multigraph:
        When true (default), returns a ``networkx.MultiGraph`` to preserve
        parallel edges and self-loops with stable edge IDs.
    """
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "networkx is required for to_networkx(). "
            "Install it with: pip install networkx"
        ) from exc

    out: nx.Graph = nx.MultiGraph() if multigraph else nx.Graph()
    out.graph["meta"] = {
        "version": graph.meta.version,
        "ndim": graph.meta.ndim,
        "shape": tuple(int(v) for v in graph.meta.shape),
        "spacing": tuple(float(v) for v in graph.meta.spacing),
        "config": dict(graph.meta.config),
        "input_hash": graph.meta.input_hash,
        "processed_mask_hash": graph.meta.processed_mask_hash,
    }

    for node in graph.nodes:
        out.add_node(
            node.id,
            id=node.id,
            xyz=tuple(float(v) for v in node.xyz),
            index=tuple(int(v) for v in node.index),
            type=node.type,
            degree=int(node.degree),
            voxel_count=int(node.voxel_count),
            radius_mean=None if node.radius_mean is None else float(node.radius_mean),
            radius_median=None if node.radius_median is None else float(node.radius_median),
        )

    for edge in graph.edges:
        attrs = {
            "id": int(edge.id),
            "u": int(edge.u),
            "v": int(edge.v),
            "path_xyz": _to_list(edge.path_xyz),
            "path_index": _to_list(edge.path_index),
            "length": float(edge.length),
            "voxel_length": int(edge.voxel_length),
            "radius_mean": None if edge.radius_mean is None else float(edge.radius_mean),
            "radius_median": None if edge.radius_median is None else float(edge.radius_median),
            "radius_profile": _to_list(edge.radius_profile),
            "arclen_profile": _to_list(edge.arclen_profile),
            "tangent_profile": _to_list(edge.tangent_profile),
            "is_self_loop": bool(edge.is_self_loop),
        }
        if multigraph:
            out.add_edge(edge.u, edge.v, key=edge.id, **attrs)
        else:
            out.add_edge(edge.u, edge.v, **attrs)

    return out


__all__ = ["to_networkx"]
