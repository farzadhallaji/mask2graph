"""Deterministic graph normalization steps."""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from .config import NormalizeConfig, SimplifyConfig
from .radius import edge_length
from .types import Edge, MaskGraph, Node
from .utils.rdp import simplify_path_with_indices


def normalize_graph(
    graph: MaskGraph,
    *,
    normalize_config: NormalizeConfig,
    simplify_config: SimplifyConfig,
) -> MaskGraph:
    nodes, edges = list(graph.nodes), list(graph.edges)
    nodes, edges = _remove_tiny_components(nodes, edges, normalize_config.min_component_length)
    max_iter = max(1, normalize_config.normalization_max_iter)
    for _ in range(max_iter):
        changed = False

        nodes, edges, spurs_changed = _prune_spurs(
            nodes,
            edges,
            normalize_config.prune_spurs_below,
            normalize_config.prune_iterations,
        )
        changed |= spurs_changed

        nodes, edges, cycles_changed = _remove_tiny_cycles(
            nodes,
            edges,
            min_cycle_length=normalize_config.min_cycle_length,
            max_cycle_area=normalize_config.max_cycle_area,
            cycle_length_to_radius_ratio=normalize_config.cycle_length_to_radius_ratio,
            spacing_dims=graph.meta.ndim,
        )
        changed |= cycles_changed

        nodes, edges, contract_changed = _contract_short_internal_edges(
            nodes,
            edges,
            normalize_config.contract_short_edges_below,
        )
        changed |= contract_changed

        prev_counts = (len(nodes), len(edges))
        nodes, edges = _remove_tiny_components(nodes, edges, normalize_config.min_component_length)
        changed |= prev_counts != (len(nodes), len(edges))

        if not changed:
            break

    if normalize_config.contract_degree2:
        nodes, edges = _contract_degree2(nodes, edges)
    if simplify_config.enabled and simplify_config.epsilon > 0.0:
        _simplify_edges(edges, simplify_config.epsilon)
    _reindex_nodes_edges(nodes, edges)
    _update_node_degrees(nodes, edges)
    return MaskGraph(nodes=nodes, edges=edges, meta=graph.meta)


def _incident(edges: list[Edge]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = defaultdict(list)
    for i, edge in enumerate(edges):
        out[edge.u].append(i)
        if edge.v != edge.u:
            out[edge.v].append(i)
    return out


def _connected_components(nodes: list[Node], edges: list[Edge]) -> list[set[int]]:
    by_node = _incident(edges)
    node_ids = {n.id for n in nodes}
    seen: set[int] = set()
    comps: list[set[int]] = []
    for nid in sorted(node_ids):
        if nid in seen:
            continue
        q = deque([nid])
        comp: set[int] = set()
        seen.add(nid)
        while q:
            cur = q.popleft()
            comp.add(cur)
            for ei in by_node.get(cur, []):
                edge = edges[ei]
                nxt = edge.v if edge.u == cur else edge.u
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        comps.append(comp)
    return comps


def _remove_tiny_components(nodes: list[Node], edges: list[Edge], min_len: float) -> tuple[list[Node], list[Edge]]:
    if min_len <= 0.0:
        return nodes, edges
    comps = _connected_components(nodes, edges)
    keep_nodes: set[int] = set()
    for comp in comps:
        total = sum(e.length for e in edges if e.u in comp and e.v in comp)
        if total >= min_len:
            keep_nodes |= comp
    new_nodes = [n for n in nodes if n.id in keep_nodes]
    new_edges = [e for e in edges if e.u in keep_nodes and e.v in keep_nodes]
    return new_nodes, new_edges


def _prune_spurs(
    nodes: list[Node], edges: list[Edge], threshold: float, max_iter: int
) -> tuple[list[Node], list[Edge], bool]:
    if threshold <= 0.0 or max_iter <= 0:
        return nodes, edges, False
    node_by_id = {n.id: n for n in nodes}
    changed_any = False
    for _ in range(max_iter):
        changed = False
        deg = _node_degrees(nodes, edges)
        inc = _incident(edges)
        remove_edges: set[int] = set()
        for node in sorted(nodes, key=lambda n: n.id):
            if node.type == "cycle":
                continue
            if deg.get(node.id, 0) != 1:
                continue
            edge_ids = inc.get(node.id, [])
            if len(edge_ids) != 1:
                continue
            ei = edge_ids[0]
            if edges[ei].length < threshold:
                remove_edges.add(ei)
                changed = True
        if not changed:
            break
        edges = [e for i, e in enumerate(edges) if i not in remove_edges]
        deg = _node_degrees(nodes, edges)
        nodes = [n for n in nodes if deg.get(n.id, 0) > 0 or n.type == "isolate"]
        node_by_id = {n.id: n for n in nodes}
        changed_any = True
    return list(node_by_id.values()), edges, changed_any


def _path_is_closed(edge: Edge) -> bool:
    if len(edge.path_index) < 3:
        return False
    return bool(np.array_equal(edge.path_index[0], edge.path_index[-1]))


def _polygon_area_2d(path_xyz: np.ndarray) -> float:
    if len(path_xyz) < 4:
        return 0.0
    xy = path_xyz[:, :2]
    x = xy[:, 0]
    y = xy[:, 1]
    return float(0.5 * abs(np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])))


def _cycle_radius(edge: Edge, node_by_id: dict[int, Node]) -> float:
    candidates: list[float] = []
    if edge.radius_median is not None:
        candidates.append(float(edge.radius_median))
    if edge.radius_mean is not None:
        candidates.append(float(edge.radius_mean))
    if edge.radius_profile is not None and len(edge.radius_profile):
        candidates.append(float(np.median(edge.radius_profile)))
    for node_id in (edge.u, edge.v):
        node = node_by_id.get(node_id)
        if node is None:
            continue
        if node.radius_median is not None:
            candidates.append(float(node.radius_median))
        if node.radius_mean is not None:
            candidates.append(float(node.radius_mean))
    return float(np.median(candidates)) if candidates else 0.0


def _remove_tiny_cycles(
    nodes: list[Node],
    edges: list[Edge],
    *,
    min_cycle_length: float,
    max_cycle_area: float,
    cycle_length_to_radius_ratio: float,
    spacing_dims: int,
) -> tuple[list[Node], list[Edge], bool]:
    if min_cycle_length <= 0.0 and max_cycle_area <= 0.0 and cycle_length_to_radius_ratio <= 0.0:
        return nodes, edges, False

    node_by_id = {n.id: n for n in nodes}
    remove_edge_ids: set[int] = set()

    for i, edge in enumerate(edges):
        if not edge.is_self_loop and not _path_is_closed(edge):
            continue

        length_ok = min_cycle_length <= 0.0 or edge.length <= min_cycle_length
        if not length_ok:
            continue

        area_ok = True
        if max_cycle_area > 0.0 and spacing_dims == 2:
            area_ok = _polygon_area_2d(edge.path_xyz) <= max_cycle_area
        if not area_ok:
            continue

        radius_ok = True
        if cycle_length_to_radius_ratio > 0.0:
            radius = _cycle_radius(edge, node_by_id)
            if radius <= 0.0:
                radius_ok = False
            else:
                radius_ok = edge.length <= (cycle_length_to_radius_ratio * radius)
        if not radius_ok:
            continue

        remove_edge_ids.add(i)

    if not remove_edge_ids:
        return nodes, edges, False

    kept_edges = [e for i, e in enumerate(edges) if i not in remove_edge_ids]
    deg = _node_degrees(nodes, kept_edges)
    kept_nodes = [n for n in nodes if deg.get(n.id, 0) > 0 or n.type == "isolate"]
    return kept_nodes, kept_edges, True


def _contract_short_internal_edges(
    nodes: list[Node],
    edges: list[Edge],
    threshold: float,
) -> tuple[list[Node], list[Edge], bool]:
    if threshold <= 0.0:
        return nodes, edges, False

    changed = False
    while True:
        node_by_id = {n.id: n for n in nodes}
        deg = _node_degrees(nodes, edges)
        target_idx: int | None = None
        for i, edge in sorted(
            enumerate(edges),
            key=lambda pair: (
                pair[1].length,
                min(pair[1].u, pair[1].v),
                max(pair[1].u, pair[1].v),
                pair[1].id,
            ),
        ):
            if edge.is_self_loop or edge.u == edge.v:
                continue
            if edge.length >= threshold:
                continue
            u_node = node_by_id.get(edge.u)
            v_node = node_by_id.get(edge.v)
            if u_node is None or v_node is None:
                continue
            if u_node.type == "endpoint" or v_node.type == "endpoint":
                continue
            if deg.get(edge.u, 0) <= 1 or deg.get(edge.v, 0) <= 1:
                continue
            target_idx = i
            break

        if target_idx is None:
            break

        nodes, edges = _contract_edge(nodes, edges, target_idx)
        changed = True

    return nodes, edges, changed


def _merge_node_types(left: str, right: str) -> str:
    if "junction" in (left, right):
        return "junction"
    if "cycle" in (left, right):
        return "cycle"
    if "endpoint" in (left, right):
        return "endpoint"
    return left


def _contract_edge(nodes: list[Node], edges: list[Edge], edge_idx: int) -> tuple[list[Node], list[Edge]]:
    edge = edges[edge_idx]
    keep_id, drop_id = (edge.u, edge.v) if edge.u < edge.v else (edge.v, edge.u)
    node_by_id = {n.id: n for n in nodes}
    keep_node = node_by_id[keep_id]
    drop_node = node_by_id[drop_id]

    keep_weight = max(1, int(keep_node.voxel_count))
    drop_weight = max(1, int(drop_node.voxel_count))
    total_weight = keep_weight + drop_weight
    keep_node.xyz = tuple(
        ((keep_node.xyz[i] * keep_weight) + (drop_node.xyz[i] * drop_weight)) / total_weight for i in range(3)
    )  # type: ignore[assignment]
    keep_node.index = min(keep_node.index, drop_node.index)
    keep_node.voxel_count = int(keep_node.voxel_count + drop_node.voxel_count)
    keep_node.type = _merge_node_types(keep_node.type, drop_node.type)

    new_edges: list[Edge] = []
    for i, candidate in enumerate(edges):
        if i == edge_idx:
            continue
        if candidate.u == drop_id:
            candidate.u = keep_id
        if candidate.v == drop_id:
            candidate.v = keep_id
        candidate.is_self_loop = candidate.u == candidate.v
        new_edges.append(candidate)

    new_nodes = [n for n in nodes if n.id != drop_id]
    return new_nodes, new_edges


def _node_degrees(nodes: list[Node], edges: list[Edge]) -> dict[int, int]:
    degrees = {n.id: 0 for n in nodes}
    for edge in edges:
        if edge.u == edge.v:
            degrees[edge.u] = degrees.get(edge.u, 0) + 2
        else:
            degrees[edge.u] = degrees.get(edge.u, 0) + 1
            degrees[edge.v] = degrees.get(edge.v, 0) + 1
    return degrees


def _contract_degree2(nodes: list[Node], edges: list[Edge]) -> tuple[list[Node], list[Edge]]:
    node_by_id = {n.id: n for n in nodes}
    while True:
        deg = _node_degrees(nodes, edges)
        inc = _incident(edges)
        target: int | None = None
        for node in sorted(nodes, key=lambda n: n.id):
            if deg.get(node.id, 0) != 2:
                continue
            if node.type in ("cycle", "isolate"):
                continue
            edge_ids = inc.get(node.id, [])
            if len(edge_ids) != 2:
                continue
            if all(edges[i].u == edges[i].v == node.id for i in edge_ids):
                continue
            target = node.id
            break
        if target is None:
            break
        edge_ids = inc[target]
        e1, e2 = edges[edge_ids[0]], edges[edge_ids[1]]
        merged = _merge_through_node(e1, e2, target)
        keep = [e for i, e in enumerate(edges) if i not in set(edge_ids)]
        keep.append(merged)
        edges = keep
        nodes = [n for n in nodes if n.id != target]
        node_by_id = {n.id: n for n in nodes}
    return list(node_by_id.values()), edges


def _path_to_node(edge: Edge, node_id: int) -> tuple[np.ndarray, np.ndarray]:
    if edge.v == node_id:
        return edge.path_index.copy(), edge.path_xyz.copy()
    if edge.u == node_id:
        return edge.path_index[::-1].copy(), edge.path_xyz[::-1].copy()
    raise ValueError("edge is not incident to node")


def _path_from_node(edge: Edge, node_id: int) -> tuple[np.ndarray, np.ndarray]:
    if edge.u == node_id:
        return edge.path_index.copy(), edge.path_xyz.copy()
    if edge.v == node_id:
        return edge.path_index[::-1].copy(), edge.path_xyz[::-1].copy()
    raise ValueError("edge is not incident to node")


def _merge_through_node(e1: Edge, e2: Edge, node_id: int) -> Edge:
    p1_idx, p1_xyz = _path_to_node(e1, node_id)
    p2_idx, p2_xyz = _path_from_node(e2, node_id)
    new_idx = np.vstack([p1_idx, p2_idx[1:]]).astype(np.int32, copy=False)
    new_xyz = np.vstack([p1_xyz, p2_xyz[1:]]).astype(np.float64, copy=False)
    u = e1.u if e1.u != node_id else e1.v
    v = e2.v if e2.u == node_id else e2.u
    prof = None
    if e1.radius_profile is not None and e2.radius_profile is not None:
        prof1 = e1.radius_profile if e1.v == node_id else e1.radius_profile[::-1]
        prof2 = e2.radius_profile if e2.u == node_id else e2.radius_profile[::-1]
        prof = np.concatenate([prof1, prof2[1:]])
    edge = Edge(
        id=-1,
        u=u,
        v=v,
        path_xyz=new_xyz,
        path_index=new_idx,
        length=edge_length(new_xyz),
        voxel_length=int(len(new_idx)),
        radius_profile=prof,
        radius_mean=float(np.mean(prof)) if prof is not None and len(prof) else None,
        radius_median=float(np.median(prof)) if prof is not None and len(prof) else None,
        is_self_loop=u == v,
    )
    return edge


def _simplify_edges(edges: list[Edge], epsilon: float) -> None:
    for edge in edges:
        keep_idx = simplify_path_with_indices(edge.path_xyz, epsilon)
        edge.path_xyz = edge.path_xyz[keep_idx]
        edge.path_index = edge.path_index[keep_idx]
        if edge.radius_profile is not None:
            edge.radius_profile = edge.radius_profile[keep_idx]
            edge.radius_mean = float(np.mean(edge.radius_profile)) if len(edge.radius_profile) else None
            edge.radius_median = float(np.median(edge.radius_profile)) if len(edge.radius_profile) else None
        edge.length = edge_length(edge.path_xyz)
        edge.voxel_length = int(len(edge.path_index))


def _reindex_nodes_edges(nodes: list[Node], edges: list[Edge]) -> None:
    nodes.sort(key=lambda n: n.index)
    old_to_new = {n.id: i for i, n in enumerate(nodes)}
    for i, n in enumerate(nodes):
        n.id = i
    for edge in edges:
        edge.u = old_to_new[edge.u]
        edge.v = old_to_new[edge.v]
        edge.is_self_loop = edge.u == edge.v


def _update_node_degrees(nodes: list[Node], edges: list[Edge]) -> None:
    deg = _node_degrees(nodes, edges)
    for node in nodes:
        node.degree = int(deg.get(node.id, 0))
