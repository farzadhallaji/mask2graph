"""Node candidate detection, deterministic junction support, and clustering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_dilation, generate_binary_structure

from .types import Node
from .utils.connectivity import iter_neighbors, label_components
from .utils.coords import index_to_xyz

Index = tuple[int, ...]


@dataclass
class LogicalNode:
    temp_id: int
    label: int
    representative: Index
    voxels: NDArray[np.int32]
    node_type: str
    mst_edges: tuple[tuple[Index, Index], ...] = ()

    def mst_path(self, start: Index, end: Index) -> list[Index]:
        """Unique deterministic path in the junction-support MST."""
        if start == end:
            return [start]
        adj: dict[Index, list[Index]] = {}
        for a, b in self.mst_edges:
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)
        stack: list[tuple[Index, list[Index]]] = [(start, [start])]
        seen: set[Index] = set()
        while stack:
            cur, path = stack.pop()
            if cur == end:
                return path
            if cur in seen:
                continue
            seen.add(cur)
            for nxt in sorted(adj.get(cur, ()), reverse=True):
                if nxt not in seen:
                    stack.append((nxt, path + [nxt]))
        # Singleton endpoints/isolate clusters have no MST edges.
        if start == end:
            return [start]
        raise RuntimeError("junction support MST is disconnected")


def node_candidates_from_degree(
    skeleton: NDArray[np.bool_],
    degree_map: NDArray[np.int32],
    *,
    junction_dilation_iters: int = 0,
) -> NDArray[np.bool_]:
    junction_seed = np.asarray(skeleton & (degree_map >= 3), dtype=np.bool_)
    endpoint_seed = np.asarray(skeleton & (degree_map == 1), dtype=np.bool_)
    isolate_seed = np.asarray(skeleton & (degree_map == 0), dtype=np.bool_)

    junction_zone = junction_seed
    if junction_dilation_iters > 0 and np.any(junction_seed):
        structure = generate_binary_structure(skeleton.ndim, skeleton.ndim)
        junction_zone = np.asarray(
            binary_dilation(junction_seed, structure=structure, iterations=junction_dilation_iters),
            dtype=np.bool_,
        )
        junction_zone &= skeleton

    return np.asarray(junction_zone | endpoint_seed | isolate_seed, dtype=np.bool_)


def _node_type_for_cluster(cluster_degrees: NDArray[np.int32]) -> str:
    if np.all(cluster_degrees == 0):
        return "isolate"
    if np.any(cluster_degrees >= 3):
        return "junction"
    return "endpoint"


def _boundary_axes(index: Index, shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(i for i, (v, n) in enumerate(zip(index, shape)) if v == 0 or v == n - 1)


def _physical_step(a: Index, b: Index, spacing: tuple[float, ...]) -> float:
    return float(np.linalg.norm(np.asarray([(x - y) * s for x, y, s in zip(a, b, spacing)], dtype=float)))


def _deterministic_mst(coords: list[Index], shape: tuple[int, ...], spacing: tuple[float, ...]) -> tuple[tuple[Index, Index], ...]:
    """Kruskal MST of the induced full-neighborhood lattice graph.

    The output is purely deterministic: weight first, then lexicographic endpoints.
    For singleton supports the MST is empty.
    """
    if len(coords) <= 1:
        return ()
    support = set(coords)
    candidates: list[tuple[float, Index, Index]] = []
    for a in sorted(support):
        for b in iter_neighbors(a, shape):
            if b not in support or not a < b:
                continue
            candidates.append((_physical_step(a, b, spacing), a, b))
    candidates.sort(key=lambda item: (round(item[0], 15), item[1], item[2]))

    parent = {x: x for x in support}
    rank = {x: 0 for x in support}

    def find(x: Index) -> Index:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: Index, b: Index) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb] or (rank[ra] == rank[rb] and rb < ra):
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        return True

    out: list[tuple[Index, Index]] = []
    for _, a, b in candidates:
        if union(a, b):
            out.append((a, b))
            if len(out) == len(coords) - 1:
                break
    if len(out) != len(coords) - 1:
        raise RuntimeError("node candidate cluster is not connected")
    return tuple(out)


def merge_node_candidate_clusters(
    node_candidates: NDArray[np.bool_],
    degree_map: NDArray[np.int32],
    spacing: tuple[float, ...],
    float_decimals: int,
    *,
    resolution: str = "mst",
) -> tuple[list[LogicalNode], list[Node], NDArray[np.int32]]:
    labels, n_labels = label_components(node_candidates)
    logical_nodes: list[LogicalNode] = []
    nodes: list[Node] = []
    shape = tuple(int(v) for v in node_candidates.shape)
    for label in range(1, n_labels + 1):
        coords = np.argwhere(labels == label).astype(np.int32)
        if len(coords) == 0:
            continue
        coords_list = [tuple(int(v) for v in row) for row in coords]
        rep = min(coords_list)
        deg_vals = np.array([degree_map[idx] for idx in coords_list], dtype=np.int32)
        node_type = _node_type_for_cluster(deg_vals)
        # A connected support tree is required for geometrically supported branch
        # endpoints in every mode.  "mst" is the production/default resolver;
        # "cluster" remains a compatibility spelling but uses the same safe tree.
        mst = _deterministic_mst(coords_list, shape, spacing)
        axes = _boundary_axes(rep, shape)
        if node_type == "endpoint" and axes:
            node_type = "boundary_endpoint"

        logical_nodes.append(
            LogicalNode(
                temp_id=len(logical_nodes),
                label=label,
                representative=rep,
                voxels=coords,
                node_type=node_type,
                mst_edges=mst,
            )
        )
        node = Node(
            id=len(nodes),
            # Supported lattice anchor: branch geometry and logical node geometry agree.
            xyz=index_to_xyz(rep, spacing, decimals=float_decimals),
            index=rep,
            type=node_type,
            degree=0,
            voxel_count=int(len(coords)),
            support_indices=coords.copy(),
            junction_mst_edges=mst,
            on_image_boundary=bool(axes),
            boundary_axes=axes,
        )
        nodes.append(node)
    logical_nodes.sort(key=lambda n: n.representative)
    nodes.sort(key=lambda n: n.index)
    for i, node in enumerate(nodes):
        node.id = i
    rep_to_id = {node.index: node.id for node in nodes}
    for ln in logical_nodes:
        ln.temp_id = rep_to_id[ln.representative]
    return logical_nodes, nodes, labels


def synthetic_cycle_node(
    index: Index, spacing: tuple[float, ...], float_decimals: int
) -> Node:
    return Node(
        id=-1,
        xyz=index_to_xyz(index, spacing, decimals=float_decimals),
        index=index,
        type="cycle",
        degree=2,
        voxel_count=1,
        support_indices=np.asarray([index], dtype=np.int32),
        on_image_boundary=False,
    )
