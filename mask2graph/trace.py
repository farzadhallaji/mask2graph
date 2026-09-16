"""Edge tracing from supported logical nodes on 2D/3D skeletons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .nodes import LogicalNode, merge_node_candidate_clusters, node_candidates_from_degree, synthetic_cycle_node
from .types import Edge, ExtractionError, Node
from .utils.connectivity import iter_neighbors, label_components
from .utils.coords import indices_to_xyz


@dataclass
class RawExtraction:
    nodes: list[Node]
    edges: list[Edge]
    node_candidates: NDArray[np.bool_]
    node_labels: NDArray[np.int32]
    component_labels: NDArray[np.int32]
    expected_trace_segments: int = 0
    covered_trace_segments: int = 0


def _segment_key(a: tuple[int, ...], b: tuple[int, ...], shape: tuple[int, ...]) -> tuple[int, int]:
    af = int(np.ravel_multi_index(a, shape))
    bf = int(np.ravel_multi_index(b, shape))
    return (af, bf) if af < bf else (bf, af)


def _expected_trace_segment_keys(skeleton: NDArray[np.bool_], node_labels: NDArray[np.int32]) -> set[tuple[int, int]]:
    """Skeleton adjacencies to be owned by exactly one branch.

    Adjacencies wholly inside the same logical-node support are intentionally
    excluded: they are represented by the junction-support MST provenance.
    """
    out: set[tuple[int, int]] = set()
    shape = tuple(int(v) for v in skeleton.shape)
    for row in np.argwhere(skeleton):
        a = tuple(int(v) for v in row)
        la = int(node_labels[a])
        for b in iter_neighbors(a, shape):
            if not skeleton[b] or not a < b:
                continue
            lb = int(node_labels[b])
            if la > 0 and la == lb:
                continue
            out.add(_segment_key(a, b, shape))
    return out


def _trace_from_seed(
    *,
    skeleton: NDArray[np.bool_],
    node_labels: NDArray[np.int32],
    start_node: LogicalNode,
    boundary_voxel: tuple[int, ...],
    seed_voxel: tuple[int, ...],
    label_to_logical: dict[int, LogicalNode],
    visited_segments: set[tuple[int, int]],
    spacing: tuple[float, ...],
) -> Edge:
    shape = skeleton.shape
    first_key = _segment_key(boundary_voxel, seed_voxel, shape)
    if first_key in visited_segments:
        raise ExtractionError("seed segment visited twice before edge creation")
    visited_segments.add(first_key)

    chain: list[tuple[int, ...]] = []
    prev = boundary_voxel
    cur = seed_voxel

    end_node: LogicalNode | None = None
    while True:
        cur_label = int(node_labels[cur])
        if cur_label > 0:
            end_node = label_to_logical[cur_label]
            break

        chain.append(cur)
        neighbors = [n for n in iter_neighbors(cur, shape) if skeleton[n]]
        forward = [n for n in neighbors if n != prev]
        if len(forward) == 0:
            raise ExtractionError("non-node chain voxel has 0 forward neighbors")
        if len(forward) > 1:
            raise ExtractionError("non-node chain voxel has >1 forward neighbors")
        nxt = forward[0]
        key = _segment_key(cur, nxt, shape)
        if key in visited_segments:
            raise ExtractionError("segment revisited before trace termination")
        visited_segments.add(key)
        prev, cur = cur, nxt

    assert end_node is not None
    # Junction interiors are intentionally contracted to one supported lattice
    # anchor.  The MST is retained as provenance/topology certification, but it
    # is not expanded independently into every incident branch (which would
    # duplicate/overlap junction-internal geometry in the output PSLG).
    path_indices = [start_node.representative, *chain, end_node.representative]
    dedup: list[tuple[int, ...]] = []
    for p in path_indices:
        if not dedup or p != dedup[-1]:
            dedup.append(p)
    path_index = np.asarray(dedup, dtype=np.int32)
    path_xyz = indices_to_xyz(path_index, spacing)
    return Edge(
        id=-1,
        u=start_node.temp_id,
        v=end_node.temp_id,
        path_xyz=path_xyz,
        path_index=path_index,
        length=0.0,
        voxel_length=int(len(path_index)),
        is_self_loop=start_node.temp_id == end_node.temp_id,
    )


def _cycle_edge_for_component(
    component_mask: NDArray[np.bool_],
    spacing: tuple[float, ...],
    float_decimals: int,
) -> tuple[Node, Edge, set[tuple[int, int]]]:
    coords = np.argwhere(component_mask)
    start = min(tuple(int(v) for v in row) for row in coords)
    neighbors = [n for n in iter_neighbors(start, component_mask.shape) if component_mask[n]]
    if len(neighbors) != 2:
        raise ExtractionError("pure cycle component does not have degree-2 structure")
    neighbors.sort()
    prev = start
    cur = neighbors[0]
    chain = [cur]
    seen: set[tuple[int, int]] = {_segment_key(start, cur, component_mask.shape)}
    while True:
        nbrs = [n for n in iter_neighbors(cur, component_mask.shape) if component_mask[n] and n != prev]
        if len(nbrs) != 1:
            raise ExtractionError("cycle trace encountered non degree-2 voxel")
        nxt = nbrs[0]
        seg = _segment_key(cur, nxt, component_mask.shape)
        if seg in seen and nxt != start:
            raise ExtractionError("cycle trace re-entered same segment before closure")
        seen.add(seg)
        if nxt == start:
            break
        chain.append(nxt)
        prev, cur = cur, nxt

    node = synthetic_cycle_node(start, spacing, float_decimals)
    node.voxel_count = int(np.count_nonzero(component_mask))
    node.support_indices = coords.astype(np.int32, copy=True)
    path_index = np.asarray([start, *chain, start], dtype=np.int32)
    edge = Edge(
        id=-1,
        u=-1,
        v=-1,
        path_xyz=indices_to_xyz(path_index, spacing),
        path_index=path_index,
        length=0.0,
        voxel_length=int(len(path_index)),
        is_self_loop=True,
    )
    return node, edge, seen


def extract_raw_graph(
    skeleton: NDArray[np.bool_],
    degree_map: NDArray[np.int32],
    spacing: tuple[float, ...],
    float_decimals: int,
    junction_dilation_iters: int = 0,
    junction_resolution: str = "mst",
) -> RawExtraction:
    node_candidates = node_candidates_from_degree(
        skeleton,
        degree_map,
        junction_dilation_iters=junction_dilation_iters,
    )
    logical_nodes, nodes, node_labels = merge_node_candidate_clusters(
        node_candidates, degree_map, spacing, float_decimals, resolution=junction_resolution
    )
    label_to_logical = {ln.label: ln for ln in logical_nodes}

    edges: list[Edge] = []
    visited_segments: set[tuple[int, int]] = set()
    junction_artifact_segments: set[tuple[int, int]] = set()

    for ln in sorted(logical_nodes, key=lambda x: x.representative):
        cluster = {tuple(int(v) for v in row) for row in ln.voxels}
        for bv in sorted(cluster):
            neighbors = [n for n in iter_neighbors(bv, skeleton.shape) if skeleton[n] and n not in cluster]
            for seed in sorted(neighbors):
                key = _segment_key(bv, seed, skeleton.shape)
                if key in visited_segments:
                    continue
                before = set(visited_segments)
                edge = _trace_from_seed(
                    skeleton=skeleton,
                    node_labels=node_labels,
                    start_node=ln,
                    boundary_voxel=bv,
                    seed_voxel=seed,
                    label_to_logical=label_to_logical,
                    visited_segments=visited_segments,
                    spacing=spacing,
                )
                # Collapsing a thick junction support can turn a raw digital
                # micro-cycle into the two-edge walk rep->p->rep.  This is a
                # junction raster artifact, not a realizable geometric cycle.
                # Treat its external adjacencies as part of junction cleanup.
                unique_path = {tuple(int(v) for v in row) for row in edge.path_index}
                if edge.is_self_loop and len(unique_path) < 3:
                    junction_artifact_segments |= (visited_segments - before)
                else:
                    edges.append(edge)

    component_labels, n_components = label_components(skeleton)
    for comp_id in range(1, n_components + 1):
        comp_mask = component_labels == comp_id
        if np.any(node_candidates & comp_mask):
            continue
        cycle_node, cycle_edge, cycle_seen = _cycle_edge_for_component(comp_mask, spacing, float_decimals)
        cycle_node.id = len(nodes)
        cycle_edge.u = cycle_node.id
        cycle_edge.v = cycle_node.id
        nodes.append(cycle_node)
        edges.append(cycle_edge)
        visited_segments |= cycle_seen

    for edge in edges:
        edge.path_xyz = indices_to_xyz(edge.path_index, spacing)
        edge.voxel_length = int(len(edge.path_index))
    _update_node_degrees(nodes, edges)

    expected = _expected_trace_segment_keys(skeleton, node_labels) - junction_artifact_segments
    covered = visited_segments - junction_artifact_segments
    if covered != expected:
        missing = len(expected - covered)
        extra = len(covered - expected)
        raise ExtractionError(f"branch coverage invariant failed: missing={missing} extra={extra}")

    return RawExtraction(
        nodes=nodes,
        edges=edges,
        node_candidates=node_candidates,
        node_labels=node_labels,
        component_labels=component_labels,
        expected_trace_segments=len(expected),
        covered_trace_segments=len(covered),
    )


def _update_node_degrees(nodes: list[Node], edges: list[Edge]) -> None:
    degrees = {n.id: 0 for n in nodes}
    for edge in edges:
        if edge.u == edge.v:
            degrees[edge.u] += 2
        else:
            degrees[edge.u] += 1
            degrees[edge.v] += 1
    for node in nodes:
        node.degree = int(degrees.get(node.id, 0))
