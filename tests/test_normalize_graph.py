from __future__ import annotations

import numpy as np

from mask2graph.config import NormalizeConfig, SimplifyConfig
from mask2graph.normalize import normalize_graph
from mask2graph.types import Edge, GraphMeta, MaskGraph, Node


def _meta(ndim: int = 2) -> GraphMeta:
    return GraphMeta(
        version="test",
        ndim=ndim,
        shape=(16, 16) if ndim == 2 else (8, 8, 8),
        spacing=(1.0, 1.0) if ndim == 2 else (1.0, 1.0, 1.0),
        config={},
        input_hash="in",
        processed_mask_hash="out",
    )


def test_contract_short_internal_edge():
    nodes = [
        Node(id=0, xyz=(4.0, 4.0, 0.0), index=(4, 4), type="junction", degree=0, voxel_count=2),
        Node(id=1, xyz=(5.0, 4.0, 0.0), index=(4, 5), type="junction", degree=0, voxel_count=2),
        Node(id=2, xyz=(1.0, 4.0, 0.0), index=(4, 1), type="endpoint", degree=0, voxel_count=1),
        Node(id=3, xyz=(8.0, 4.0, 0.0), index=(4, 8), type="endpoint", degree=0, voxel_count=1),
    ]
    edges = [
        Edge(
            id=0,
            u=0,
            v=1,
            path_xyz=np.asarray([[4.0, 4.0, 0.0], [5.0, 4.0, 0.0]], dtype=np.float64),
            path_index=np.asarray([[4, 4], [4, 5]], dtype=np.int32),
            length=0.5,
            voxel_length=2,
        ),
        Edge(
            id=1,
            u=0,
            v=2,
            path_xyz=np.asarray([[4.0, 4.0, 0.0], [1.0, 4.0, 0.0]], dtype=np.float64),
            path_index=np.asarray([[4, 4], [4, 1]], dtype=np.int32),
            length=3.0,
            voxel_length=2,
        ),
        Edge(
            id=2,
            u=1,
            v=3,
            path_xyz=np.asarray([[5.0, 4.0, 0.0], [8.0, 4.0, 0.0]], dtype=np.float64),
            path_index=np.asarray([[4, 5], [4, 8]], dtype=np.int32),
            length=3.0,
            voxel_length=2,
        ),
    ]
    graph = MaskGraph(nodes=nodes, edges=edges, meta=_meta())
    cfg = NormalizeConfig(
        contract_short_edges_below=1.0,
        contract_degree2=False,
        normalization_max_iter=5,
    )
    out = normalize_graph(graph, normalize_config=cfg, simplify_config=SimplifyConfig())
    assert len(out.nodes) == 3
    assert all(edge.length >= 1.0 for edge in out.edges)


def test_remove_tiny_cycle_with_length_area_radius_gates():
    node = Node(id=0, xyz=(1.0, 1.0, 0.0), index=(1, 1), type="cycle", degree=2, voxel_count=4)
    edge = Edge(
        id=0,
        u=0,
        v=0,
        path_xyz=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.float64,
        ),
        path_index=np.asarray([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.int32),
        length=4.0,
        voxel_length=5,
        radius_median=0.5,
        is_self_loop=True,
    )
    graph = MaskGraph(nodes=[node], edges=[edge], meta=_meta())
    cfg = NormalizeConfig(
        min_cycle_length=5.0,
        max_cycle_area=2.0,
        cycle_length_to_radius_ratio=10.0,
        contract_degree2=False,
    )
    out = normalize_graph(graph, normalize_config=cfg, simplify_config=SimplifyConfig())
    assert len(out.edges) == 0


def test_keep_cycle_when_radius_ratio_gate_fails():
    node = Node(id=0, xyz=(1.0, 1.0, 0.0), index=(1, 1), type="cycle", degree=2, voxel_count=4)
    edge = Edge(
        id=0,
        u=0,
        v=0,
        path_xyz=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.float64,
        ),
        path_index=np.asarray([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.int32),
        length=4.0,
        voxel_length=5,
        radius_median=0.5,
        is_self_loop=True,
    )
    graph = MaskGraph(nodes=[node], edges=[edge], meta=_meta())
    cfg = NormalizeConfig(
        min_cycle_length=5.0,
        max_cycle_area=2.0,
        cycle_length_to_radius_ratio=2.0,
        contract_degree2=False,
    )
    out = normalize_graph(graph, normalize_config=cfg, simplify_config=SimplifyConfig())
    assert len(out.edges) == 1
    assert out.edges[0].is_self_loop
