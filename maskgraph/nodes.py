"""Node candidate detection and logical-node clustering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_dilation, generate_binary_structure

from .types import Node
from .utils.connectivity import label_components
from .utils.coords import index_to_xyz, indices_to_xyz


@dataclass
class LogicalNode:
    temp_id: int
    label: int
    representative: tuple[int, ...]
    voxels: NDArray[np.int32]
    node_type: str


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


def merge_node_candidate_clusters(
    node_candidates: NDArray[np.bool_],
    degree_map: NDArray[np.int32],
    spacing: tuple[float, ...],
    float_decimals: int,
) -> tuple[list[LogicalNode], list[Node], NDArray[np.int32]]:
    labels, n_labels = label_components(node_candidates)
    logical_nodes: list[LogicalNode] = []
    nodes: list[Node] = []
    for label in range(1, n_labels + 1):
        coords = np.argwhere(labels == label).astype(np.int32)
        if len(coords) == 0:
            continue
        coords_list = [tuple(int(v) for v in row) for row in coords]
        rep = min(coords_list)
        deg_vals = np.array([degree_map[idx] for idx in coords_list], dtype=np.int32)
        node_type = _node_type_for_cluster(deg_vals)

        xyz_cluster = indices_to_xyz(coords, spacing, decimals=None)
        centroid = tuple(float(v) for v in xyz_cluster.mean(axis=0))
        centroid = tuple(round(v, float_decimals) for v in centroid)

        logical_nodes.append(
            LogicalNode(
                temp_id=len(logical_nodes),
                label=label,
                representative=rep,
                voxels=coords,
                node_type=node_type,
            )
        )
        node = Node(
            id=len(nodes),
            xyz=centroid,  # type: ignore[arg-type]
            index=rep,
            type=node_type,
            degree=0,
            voxel_count=int(len(coords)),
            radius_mean=None,
            radius_median=None,
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
    index: tuple[int, ...], spacing: tuple[float, ...], float_decimals: int
) -> Node:
    return Node(
        id=-1,
        xyz=index_to_xyz(index, spacing, decimals=float_decimals),
        index=index,
        type="cycle",
        degree=2,
        voxel_count=1,
    )
