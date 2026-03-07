"""Public extraction API."""

from __future__ import annotations

from dataclasses import replace
from importlib.metadata import version as pkg_version
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import ExtractConfig, default_spacing
from .degree import compute_degree_map
from .normalize import normalize_graph as _normalize_graph
from .preprocess import preprocess_mask
from .radius import estimate_radii as _estimate_radii
from .radius import update_edge_lengths
from .skeleton import skeletonize_mask
from .trace import RawExtraction, extract_raw_graph as _extract_raw_graph
from .types import DebugArtifacts, GraphMeta, MaskGraph
from .utils.hash import hash_array


def _library_version() -> str:
    try:
        return pkg_version("maskgraph")
    except Exception:  # noqa: BLE001
        return "0.1.0a0"


def _validate_input(mask: NDArray[np.generic], spacing: tuple[float, ...] | None) -> tuple[NDArray[np.bool_], tuple[float, ...]]:
    if mask.ndim not in (2, 3):
        raise ValueError("mask must be 2D or 3D")
    if any(dim <= 0 for dim in mask.shape):
        raise ValueError("mask must have non-empty shape")
    if np.issubdtype(mask.dtype, np.floating):
        raise ValueError("float mask dtype is not allowed")
    if not (np.issubdtype(mask.dtype, np.bool_) or np.issubdtype(mask.dtype, np.integer)):
        raise ValueError("mask dtype must be bool or integer")
    if spacing is None:
        spacing = default_spacing(mask.ndim)
    if len(spacing) != mask.ndim:
        raise ValueError("spacing length must match mask ndim")
    if any(v <= 0 for v in spacing):
        raise ValueError("spacing values must be positive")
    return np.asarray(mask != 0, dtype=np.bool_), tuple(float(v) for v in spacing)


def _round_graph(graph: MaskGraph, decimals: int) -> None:
    for node in graph.nodes:
        node.xyz = tuple(round(float(v), decimals) for v in node.xyz)  # type: ignore[assignment]
        if node.radius_mean is not None:
            node.radius_mean = round(float(node.radius_mean), decimals)
        if node.radius_median is not None:
            node.radius_median = round(float(node.radius_median), decimals)
    for edge in graph.edges:
        edge.path_xyz = np.round(edge.path_xyz.astype(np.float64, copy=False), decimals=decimals)
        edge.length = round(float(edge.length), decimals)
        if edge.radius_mean is not None:
            edge.radius_mean = round(float(edge.radius_mean), decimals)
        if edge.radius_median is not None:
            edge.radius_median = round(float(edge.radius_median), decimals)
        if edge.radius_profile is not None:
            edge.radius_profile = np.round(edge.radius_profile.astype(np.float64, copy=False), decimals=decimals)


def _rotate_self_loop_path(path: NDArray[np.int32]) -> NDArray[np.int32]:
    if len(path) <= 2:
        return path
    core = path[:-1]
    tuples = [tuple(int(v) for v in row) for row in core]
    min_idx = min(range(len(tuples)), key=tuples.__getitem__)
    rot = np.vstack([core[min_idx:], core[:min_idx]])
    return np.vstack([rot, rot[0]]).astype(np.int32, copy=False)


def _enforce_edge_orientation(edge) -> None:
    if edge.u != edge.v and edge.u > edge.v:
        edge.u, edge.v = edge.v, edge.u
        edge.path_index = edge.path_index[::-1].copy()
        edge.path_xyz = edge.path_xyz[::-1].copy()
        if edge.radius_profile is not None:
            edge.radius_profile = edge.radius_profile[::-1].copy()
    if edge.u == edge.v:
        edge.path_index = _rotate_self_loop_path(edge.path_index)
        # reorder xyz/profile to match rotated index order deterministically
        if len(edge.path_xyz) == len(edge.path_index):
            idx = np.arange(len(edge.path_xyz))
            core = idx[:-1]
            min_i = int(np.argmin([tuple(int(v) for v in row) for row in edge.path_index[:-1]]))
            core_rot = np.concatenate([core[min_i:], core[:min_i]])
            order = np.concatenate([core_rot, [core_rot[0]]])
            edge.path_xyz = edge.path_xyz[order]
            if edge.radius_profile is not None:
                edge.radius_profile = edge.radius_profile[order]


def _determinize(graph: MaskGraph, cfg: ExtractConfig) -> MaskGraph:
    nodes = list(graph.nodes)
    edges = list(graph.edges)

    if cfg.determinism.sort_nodes:
        nodes.sort(key=lambda n: n.index)
    old_to_new = {n.id: i for i, n in enumerate(nodes)}
    for i, n in enumerate(nodes):
        n.id = i
    for edge in edges:
        edge.u = old_to_new[edge.u]
        edge.v = old_to_new[edge.v]
        _enforce_edge_orientation(edge)
    if cfg.determinism.sort_edges:
        edges.sort(
            key=lambda e: (
                min(e.u, e.v),
                max(e.u, e.v),
                tuple(int(v) for v in e.path_index[0]),
                int(len(e.path_index)),
                e.id,
            )
        )
    for i, edge in enumerate(edges):
        edge.id = i
    out = replace(graph, nodes=nodes, edges=edges)
    _round_graph(out, cfg.determinism.float_decimals)
    return out


def extract_graph(
    mask: np.ndarray,
    *,
    spacing: tuple[float, ...] | None = None,
    config: ExtractConfig | None = None,
    return_debug: bool = False,
) -> MaskGraph | tuple[MaskGraph, DebugArtifacts]:
    cfg = config or ExtractConfig()
    mask_input, spacing_t = _validate_input(np.asarray(mask), spacing)
    cfg.validate(mask_input.ndim)

    mask_processed, cleanup_report = preprocess_mask(
        mask_input,
        cfg.cleanup,
        spacing=spacing_t,
        return_report=True,
    )
    skeleton = skeletonize_mask(mask_processed, cfg.skeleton)
    degree_map = compute_degree_map(skeleton)
    raw: RawExtraction = _extract_raw_graph(
        skeleton=skeleton,
        degree_map=degree_map,
        spacing=spacing_t,
        float_decimals=cfg.determinism.float_decimals,
        junction_dilation_iters=cfg.normalize.junction_dilation_iters,
    )
    update_edge_lengths(raw.edges)
    _estimate_radii(
        nodes=raw.nodes,
        edges=raw.edges,
        node_labels=raw.node_labels,
        mask_processed=mask_processed,
        spacing=spacing_t,
    )

    meta = GraphMeta(
        version=_library_version(),
        ndim=mask_input.ndim,
        shape=tuple(int(v) for v in mask_input.shape),
        spacing=spacing_t,
        config=cfg.to_dict(),
        input_hash=hash_array(mask_input),
        processed_mask_hash=hash_array(mask_processed),
    )
    graph = MaskGraph(nodes=raw.nodes, edges=raw.edges, meta=meta)
    graph = _normalize_graph(graph, normalize_config=cfg.normalize, simplify_config=cfg.simplify)
    graph = _determinize(graph, cfg)

    if not return_debug:
        return graph
    debug = DebugArtifacts(
        mask_input=mask_input,
        mask_processed=mask_processed,
        cleanup_report=cleanup_report,
        skeleton=skeleton,
        degree_map=degree_map,
        node_candidates=raw.node_candidates,
        node_labels=raw.node_labels,
        pruned_skeleton=skeleton.copy(),
        component_labels=raw.component_labels,
    )
    return graph, debug


def normalize_graph(graph: MaskGraph, config: ExtractConfig) -> MaskGraph:
    return _normalize_graph(graph, normalize_config=config.normalize, simplify_config=config.simplify)


def estimate_radii(
    graph: MaskGraph,
    *,
    node_labels: NDArray[np.int32],
    mask_processed: NDArray[np.bool_],
    spacing: tuple[float, ...],
) -> NDArray[np.float64]:
    return _estimate_radii(
        nodes=graph.nodes,
        edges=graph.edges,
        node_labels=node_labels,
        mask_processed=mask_processed,
        spacing=spacing,
    )


def extract_raw_graph(
    skeleton: NDArray[np.bool_],
    degree_map: NDArray[np.int32],
    *,
    spacing: tuple[float, ...],
    config: ExtractConfig | None = None,
) -> RawExtraction:
    cfg = config or ExtractConfig()
    return _extract_raw_graph(
        skeleton=skeleton,
        degree_map=degree_map,
        spacing=spacing,
        float_decimals=cfg.determinism.float_decimals,
        junction_dilation_iters=cfg.normalize.junction_dilation_iters,
    )


__all__ = [
    "compute_degree_map",
    "estimate_radii",
    "extract_graph",
    "extract_raw_graph",
    "normalize_graph",
    "preprocess_mask",
    "skeletonize_mask",
]
