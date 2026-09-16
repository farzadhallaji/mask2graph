"""Graph cropping utilities.

``crop_graph_box`` performs geometric clipping of complete source branch
polylines.  The radius/connected-subgraph helpers remain deterministic patch
selection utilities and intentionally keep whole topology edges.
"""

from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from typing import Sequence

import numpy as np

from .clip import crop_result_box, crop_topology_graph_box, normalize_box_bounds, validate_crop_result
from .config import ExtractConfig
from .types import Mask2Graph, MaskGraphResult
from .utils.hash import graph_content_hash


def crop_graph_box(
    graph: Mask2Graph | MaskGraphResult,
    minimum: Sequence[float] | None = None,
    maximum: Sequence[float] | None = None,
    *,
    bounds: Sequence[float] | None = None,
    keep_size: bool = False,
    config: ExtractConfig | None = None,
    validate: bool = True,
):
    """Geometrically clip a topology graph/full result to an axis-aligned box.

    Backward-compatible ``minimum, maximum`` positional arguments are accepted;
    the preferred form is ``bounds=(xmin,xmax,ymin,ymax[,zmin,zmax])``.

    ``keep_size=True`` translates the cropped box lower corner to the origin
    after clipping; it is literally crop + translation, not a coordinate hack.
    """
    topology = graph.topology_graph if isinstance(graph, MaskGraphResult) else graph
    lo, hi = normalize_box_bounds(topology.meta.ndim, bounds=bounds, minimum=minimum, maximum=maximum)
    if isinstance(graph, MaskGraphResult):
        out = crop_result_box(graph, bounds=[x for pair in zip(lo, hi) for x in pair], config=config)
        if validate:
            validate_crop_result(out, lo, hi, (config or ExtractConfig.from_dict(topology.meta.config)).validation.tolerance)
        if keep_size:
            from .transforms import translate_graph
            out = translate_graph(out, tuple(float(-v) for v in lo), validate=validate)
        return out

    out_g = crop_topology_graph_box(graph, bounds=[x for pair in zip(lo, hi) for x in pair], config=config)
    if keep_size:
        from .transforms import translate_graph
        out_g = translate_graph(out_g, tuple(float(-v) for v in lo), validate=validate)
    return out_g


def crop_graph(graph: Mask2Graph | MaskGraphResult, *, bounds: Sequence[float], keep_size: bool = False, config: ExtractConfig | None = None, validate: bool = True):
    """Named box-crop alias used by the augmentation API."""
    return crop_graph_box(graph, bounds=bounds, keep_size=keep_size, config=config, validate=validate)


def _finalize_subgraph(graph: Mask2Graph, keep_edge_ids: set[int], keep_node_ids: set[int] | None = None) -> Mask2Graph:
    out = deepcopy(graph)
    edges = [e for e in out.edges if e.id in keep_edge_ids]
    nodes_used = {e.u for e in edges} | {e.v for e in edges}
    if keep_node_ids is not None:
        nodes_used |= set(keep_node_ids)
    nodes = [n for n in out.nodes if n.id in nodes_used]
    old_to_new = {n.id: i for i, n in enumerate(sorted(nodes, key=lambda n: n.id))}
    nodes.sort(key=lambda n: n.id)
    for i, n in enumerate(nodes):
        n.id = i
    for i, e in enumerate(sorted(edges, key=lambda e: e.id)):
        e.u = old_to_new[e.u]
        e.v = old_to_new[e.v]
        e.id = i
    out.nodes = nodes
    out.edges = sorted(edges, key=lambda e: e.id)
    if not out.meta.source_graph_hash:
        out.meta.source_graph_hash = graph.meta.graph_hash or graph_content_hash(graph)
    out.meta.augmentation_history = [*out.meta.augmentation_history, {
        "type": "subgraph_selection", "whole_edges": True, "edge_ids": sorted(int(v) for v in keep_edge_ids)
    }]
    out.meta.graph_hash = graph_content_hash(out)
    out.diagnostics = None
    return out


def crop_graph_radius(graph: Mask2Graph, center: tuple[float, ...], radius: float) -> Mask2Graph:
    """Select whole topology edges lying entirely inside a radius (not geometric clipping)."""
    c = np.asarray(center, dtype=float)
    if len(c) != graph.meta.ndim or radius < 0:
        raise ValueError("invalid center/radius")
    keep = {
        e.id
        for e in graph.edges
        if np.all(np.linalg.norm(e.path_xyz[:, : graph.meta.ndim] - c, axis=1) <= float(radius))
    }
    return _finalize_subgraph(graph, keep)


def crop_graph_connected_subgraph(graph: Mask2Graph, seed_node: int, max_edges: int) -> Mask2Graph:
    """Breadth-first deterministic whole-edge patch around a topology node."""
    if max_edges < 0:
        raise ValueError("max_edges must be >= 0")
    node_ids = {n.id for n in graph.nodes}
    if seed_node not in node_ids:
        raise ValueError("seed_node not present")
    inc: dict[int, list[int]] = defaultdict(list)
    by_edge = {e.id: e for e in graph.edges}
    for e in graph.edges:
        inc[e.u].append(e.id)
        if e.v != e.u:
            inc[e.v].append(e.id)
    for ids in inc.values():
        ids.sort()
    q = deque([seed_node])
    seen_nodes = {seed_node}
    keep: set[int] = set()
    while q and len(keep) < max_edges:
        u = q.popleft()
        for eid in inc.get(u, ()):
            if len(keep) >= max_edges:
                break
            if eid in keep:
                continue
            keep.add(eid)
            e = by_edge[eid]
            v = e.v if e.u == u else e.u
            if v not in seen_nodes:
                seen_nodes.add(v)
                q.append(v)
    return _finalize_subgraph(graph, keep, {seed_node} if not keep else None)
