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
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover
        raise ImportError("networkx is required for to_networkx(). Install it with: pip install networkx") from exc

    out: nx.Graph = nx.MultiGraph() if multigraph else nx.Graph()
    out.graph["meta"] = {
        "version": graph.meta.version, "ndim": graph.meta.ndim, "shape": tuple(graph.meta.shape),
        "spacing": tuple(graph.meta.spacing), "config": dict(graph.meta.config),
        "input_hash": graph.meta.input_hash, "processed_mask_hash": graph.meta.processed_mask_hash,
        "config_hash": graph.meta.config_hash, "graph_hash": graph.meta.graph_hash,
        "source_graph_hash": graph.meta.source_graph_hash, "grid_aligned": graph.meta.grid_aligned,
        "transform_exact": graph.meta.transform_exact, "rationalization_error": graph.meta.rationalization_error,
        "augmentation_history": list(graph.meta.augmentation_history), "augmentation_seed": graph.meta.augmentation_seed,
    }
    for node in graph.nodes:
        out.add_node(
            node.id, id=node.id, xyz=tuple(node.xyz), index=tuple(node.index), type=node.type,
            degree=int(node.degree), voxel_count=int(node.voxel_count), radius_mean=node.radius_mean,
            radius_median=node.radius_median, radius_min=node.radius_min, radius_max=node.radius_max,
            support_indices=_to_list(node.support_indices), on_image_boundary=node.on_image_boundary,
            boundary_axes=tuple(node.boundary_axes), source_index=node.source_index, index_valid=node.index_valid,
            on_crop_boundary=node.on_crop_boundary, created_by=node.created_by,
        )
    for edge in graph.edges:
        attrs = {
            "id": edge.id, "u": edge.u, "v": edge.v, "path_xyz": _to_list(edge.path_xyz),
            "path_index": _to_list(edge.path_index), "length": edge.length, "voxel_length": edge.voxel_length,
            "radius_mean": edge.radius_mean, "radius_median": edge.radius_median, "radius_min": edge.radius_min,
            "radius_max": edge.radius_max, "radius_profile": _to_list(edge.radius_profile),
            "arclen_profile": _to_list(edge.arclen_profile), "tangent_profile": _to_list(edge.tangent_profile),
            "curvature_profile": _to_list(edge.curvature_profile),
            "turning_angle_profile": _to_list(edge.turning_angle_profile),
            "simplified_indices": _to_list(edge.simplified_indices), "simplified_xyz": _to_list(edge.simplified_xyz),
            "chord_length": edge.chord_length, "tortuosity": edge.tortuosity,
            "is_self_loop": edge.is_self_loop, "provenance": edge.provenance,
            "source_path_index": _to_list(edge.source_path_index), "path_index_valid": edge.path_index_valid,
            "source_edge_id": edge.source_edge_id, "source_arc_range": edge.source_arc_range,
            "crop_fragment_index": edge.crop_fragment_index, "created_by": edge.created_by,
        }
        if multigraph:
            out.add_edge(edge.u, edge.v, key=edge.id, **attrs)
        else:
            out.add_edge(edge.u, edge.v, **attrs)
    return out


__all__ = ["to_networkx"]
