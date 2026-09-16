"""Public extraction API for deterministic 2D/3D mask-to-graph conversion."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import numpy as np
from numpy.typing import NDArray

from ._version import __version__

from .config import ExtractConfig, default_spacing
from .degree import compute_degree_map
from .geometry import update_geometry_profiles
from .normalize import normalize_graph as _normalize_graph
from .preprocess import preprocess_mask
from .radius import estimate_radii as _estimate_radii
from .radius import update_edge_lengths
from .simplify import simplify_graph_edges
from .skeleton import skeletonize_mask
from .topology import build_topology_diagnostics, update_logical_diagnostics
from .trace import RawExtraction, extract_raw_graph as _extract_raw_graph
from .types import DebugArtifacts, GraphMeta, Mask2Graph, MaskGraphResult, TopologyDiagnostics
from .utils.hash import graph_content_hash, hash_array, stable_config_json
from .validate import (
    build_validated_embedded_graph,
    validate_skeleton_coverage,
    validate_topology,
)


def _library_version() -> str:
    """Return the release version from the package's single version source."""
    return __version__


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


def _round_array(value: np.ndarray | None, decimals: int) -> np.ndarray | None:
    if value is None:
        return None
    return np.round(np.asarray(value, dtype=np.float64), decimals=decimals)


def _round_graph(graph: Mask2Graph, decimals: int) -> None:
    for node in graph.nodes:
        node.xyz = tuple(round(float(v), decimals) for v in node.xyz)  # type: ignore[assignment]
        for name in ("radius_mean", "radius_median", "radius_min", "radius_max"):
            value = getattr(node, name)
            if value is not None:
                setattr(node, name, round(float(value), decimals))
    for edge in graph.edges:
        edge.path_xyz = np.round(edge.path_xyz.astype(np.float64, copy=False), decimals=decimals)
        edge.simplified_xyz = _round_array(edge.simplified_xyz, decimals)
        edge.length = round(float(edge.length), decimals)
        edge.chord_length = round(float(edge.chord_length), decimals)
        edge.tortuosity = round(float(edge.tortuosity), decimals)
        for name in ("radius_mean", "radius_median", "radius_min", "radius_max"):
            value = getattr(edge, name)
            if value is not None:
                setattr(edge, name, round(float(value), decimals))
        for name in (
            "radius_profile", "arclen_profile", "tangent_profile", "curvature_profile", "turning_angle_profile"
        ):
            setattr(edge, name, _round_array(getattr(edge, name), decimals))


def _reverse_edge_arrays(edge) -> None:
    edge.path_index = edge.path_index[::-1].copy()
    if edge.source_path_index is not None:
        edge.source_path_index = edge.source_path_index[::-1].copy()
    edge.path_xyz = edge.path_xyz[::-1].copy()
    if edge.simplified_indices is not None:
        edge.simplified_indices = edge.simplified_indices[::-1].copy()
    if edge.simplified_xyz is not None:
        edge.simplified_xyz = edge.simplified_xyz[::-1].copy()
    for name in (
        "radius_profile", "arclen_profile", "tangent_profile", "curvature_profile", "turning_angle_profile"
    ):
        arr = getattr(edge, name)
        if arr is not None:
            arr = arr[::-1].copy()
            if name == "arclen_profile" and len(arr):
                arr = float(arr[0]) - arr
                arr = arr - float(arr[0])
            elif name == "tangent_profile":
                arr = -arr
            setattr(edge, name, arr)


def _rotate_closed_array(arr: np.ndarray | None, min_idx: int) -> np.ndarray | None:
    if arr is None or len(arr) <= 1:
        return arr
    core = arr[:-1]
    rot = np.concatenate([core[min_idx:], core[:min_idx]], axis=0)
    return np.concatenate([rot, rot[:1]], axis=0)


def _enforce_edge_orientation(edge) -> None:
    if edge.u != edge.v and edge.u > edge.v:
        edge.u, edge.v = edge.v, edge.u
        _reverse_edge_arrays(edge)
        return
    # Self-loops deliberately keep the extraction anchor at both ends.  Rotating
    # them to another lexicographic sample would make the edge geometry disagree
    # with its logical node geometry.  Extraction itself is deterministic.


def _determinize(graph: Mask2Graph, cfg: ExtractConfig) -> Mask2Graph:
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
                min(e.u, e.v), max(e.u, e.v), tuple(int(v) for v in e.path_index[0]), int(len(e.path_index)), e.id
            )
        )
    for i, edge in enumerate(edges):
        edge.id = i
    out = replace(graph, nodes=nodes, edges=edges)
    # Profiles/simplification are recomputed after any orientation change.
    update_geometry_profiles(
        out.edges,
        tangent_window=cfg.geometry.tangent_window,
        compute_curvature=cfg.geometry.compute_curvature,
    )
    simplify_graph_edges(out.edges, cfg.simplify)
    _round_graph(out, cfg.determinism.float_decimals)
    out.meta.graph_hash = graph_content_hash(out)
    return out


def _topology_edit_enabled(cfg: ExtractConfig) -> bool:
    n = cfg.normalize
    return any(
        value > 0
        for value in (
            n.min_component_length,
            n.prune_spurs_below,
            n.min_cycle_length,
            n.max_cycle_area,
            n.cycle_length_to_radius_ratio,
            n.contract_short_edges_below,
        )
    )


def _extract_pipeline(
    mask: np.ndarray,
    *,
    spacing: tuple[float, ...] | None,
    config: ExtractConfig | None,
    return_debug: bool,
) -> MaskGraphResult:
    cfg = config or ExtractConfig()
    mask_input, spacing_t = _validate_input(np.asarray(mask), spacing)
    cfg.validate(mask_input.ndim)

    mask_processed, cleanup_report = preprocess_mask(mask_input, cfg.cleanup, spacing=spacing_t, return_report=True)
    skeleton = skeletonize_mask(mask_processed, cfg.skeleton)
    degree_map = compute_degree_map(skeleton)
    dilation = cfg.normalize.junction_dilation_iters or cfg.junction.dilation_iters
    raw: RawExtraction = _extract_raw_graph(
        skeleton=skeleton,
        degree_map=degree_map,
        spacing=spacing_t,
        float_decimals=cfg.determinism.float_decimals,
        junction_dilation_iters=dilation,
        junction_resolution=cfg.junction.resolution,
    )
    update_edge_lengths(raw.edges)
    _estimate_radii(
        nodes=raw.nodes,
        edges=raw.edges,
        node_labels=raw.node_labels,
        mask_processed=mask_processed,
        spacing=spacing_t,
    )
    update_geometry_profiles(
        raw.edges,
        tangent_window=cfg.geometry.tangent_window,
        compute_curvature=cfg.geometry.compute_curvature,
    )

    meta = GraphMeta(
        version=_library_version(),
        ndim=mask_input.ndim,
        shape=tuple(int(v) for v in mask_input.shape),
        spacing=spacing_t,
        config=cfg.to_dict(),
        input_hash=hash_array(mask_input),
        processed_mask_hash=hash_array(mask_processed),
        config_hash=hashlib.sha256(stable_config_json(cfg.to_dict()).encode("utf-8")).hexdigest(),
    )
    raw_mst_edge_count = sum(len(n.junction_mst_edges) for n in raw.nodes)
    graph = Mask2Graph(nodes=raw.nodes, edges=raw.edges, meta=meta)
    diag = build_topology_diagnostics(
        skeleton,
        graph,
        expected_trace_segments=raw.expected_trace_segments,
        covered_trace_segments=raw.covered_trace_segments,
        junction_mst_edge_count=raw_mst_edge_count,
    )
    graph = _normalize_graph(graph, normalize_config=cfg.normalize, simplify_config=cfg.simplify, simplify=False)
    graph = _determinize(graph, cfg)
    update_logical_diagnostics(diag, graph)
    diag.cleanup_topology_edit_enabled = _topology_edit_enabled(cfg)
    diag.cleanup_topology_changed = (
        (diag.cleaned_beta0, diag.cleaned_beta1) != (diag.logical_beta0, diag.logical_beta1)
        or diag.logical_vertices != diag.raw_skeleton_vertices and diag.cleanup_topology_edit_enabled
    )
    diag.max_simplification_error = max(
        (
            0.0 if e.simplified_xyz is None or len(e.simplified_xyz) < 2 else _edge_error(e)
            for e in graph.edges
        ),
        default=0.0,
    )
    graph.diagnostics = diag

    if cfg.validation.enabled:
        if cfg.validation.validate_coverage:
            validate_skeleton_coverage(diag)
        if cfg.validation.validate_topology:
            validate_topology(graph, diag, allow_topology_edit=diag.cleanup_topology_edit_enabled)

    if cfg.validation.enabled:
        embedded = build_validated_embedded_graph(
            graph,
            diag,
            simplify_config=cfg.simplify,
            validation_config=cfg.validation,
        )
    else:
        from .validate import build_embedded_graph
        embedded = build_embedded_graph(graph, simplified=True)

    debug = None
    if return_debug:
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
            diagnostics=diag,
        )
    return MaskGraphResult(topology_graph=graph, embedded_graph=embedded, diagnostics=diag, debug=debug)


def _edge_error(edge) -> float:
    from .simplify import max_subpath_error

    if edge.simplified_indices is None or edge.simplified_xyz is None or len(edge.simplified_xyz) < 2:
        return 0.0
    pos: list[int] = []
    start = 0
    for sample in edge.simplified_indices:
        matches = np.flatnonzero(np.all(edge.path_index[start:] == sample, axis=1))
        if len(matches) == 0:
            continue
        found = start + int(matches[0])
        pos.append(found)
        start = found + 1
    err = 0.0
    for a, b in zip(pos[:-1], pos[1:]):
        err = max(err, max_subpath_error(edge.path_xyz, int(a), int(b)))
    return float(err)


def extract_graph(
    mask: np.ndarray,
    *,
    spacing: tuple[float, ...] | None = None,
    config: ExtractConfig | None = None,
    return_debug: bool = False,
) -> Mask2Graph | tuple[Mask2Graph, DebugArtifacts]:
    """Backward-compatible topology-graph API.

    The full unified result, including the straight-line embedded graph, is
    available through :func:`mask_to_graph`.
    """
    result = _extract_pipeline(mask, spacing=spacing, config=config, return_debug=return_debug)
    if return_debug:
        assert result.debug is not None
        return result.topology_graph, result.debug
    return result.topology_graph


def mask_to_graph(
    mask: np.ndarray,
    *,
    spacing: tuple[float, ...] | None = None,
    config: ExtractConfig | None = None,
    return_debug: bool = False,
) -> MaskGraphResult:
    """Unified 2D/3D mask -> topology graph + validated straight-line graph."""
    return _extract_pipeline(mask, spacing=spacing, config=config, return_debug=return_debug)


def normalize_graph(graph: Mask2Graph, config: ExtractConfig) -> Mask2Graph:
    out = _normalize_graph(graph, normalize_config=config.normalize, simplify_config=config.simplify, simplify=False)
    update_geometry_profiles(out.edges, tangent_window=config.geometry.tangent_window, compute_curvature=config.geometry.compute_curvature)
    simplify_graph_edges(out.edges, config.simplify)
    return out


def estimate_radii(
    graph: Mask2Graph,
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
    dilation = cfg.normalize.junction_dilation_iters or cfg.junction.dilation_iters
    return _extract_raw_graph(
        skeleton=skeleton,
        degree_map=degree_map,
        spacing=spacing,
        float_decimals=cfg.determinism.float_decimals,
        junction_dilation_iters=dilation,
        junction_resolution=cfg.junction.resolution,
    )
