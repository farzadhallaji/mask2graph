from __future__ import annotations

import numpy as np
import pytest

from maskgraph import ExtractConfig, extract_graph, to_networkx
from maskgraph.types import Edge, MaskGraph

nx = pytest.importorskip("networkx")


def _line_mask() -> np.ndarray:
    m = np.zeros((11, 11), dtype=np.uint8)
    m[5, 2:9] = 1
    return m


def _loop_mask() -> np.ndarray:
    m = np.zeros((12, 12), dtype=np.uint8)
    m[3, 3:9] = 1
    m[8, 3:9] = 1
    m[3:9, 3] = 1
    m[3:9, 8] = 1
    return m


def test_to_networkx_multigraph_preserves_counts_and_attributes():
    g = extract_graph(_line_mask(), spacing=(1.0, 1.0))
    nxg = to_networkx(g, multigraph=True)
    assert isinstance(nxg, nx.MultiGraph)
    assert nxg.number_of_nodes() == len(g.nodes)
    assert nxg.number_of_edges() == len(g.edges)
    node0 = next(iter(nxg.nodes))
    assert "xyz" in nxg.nodes[node0]
    assert "degree" in nxg.nodes[node0]
    u, v, k, attrs = next(iter(nxg.edges(data=True, keys=True)))
    assert k == attrs["id"]
    assert isinstance(attrs["path_xyz"], list)
    assert isinstance(attrs["path_index"], list)


def test_to_networkx_preserves_self_loop():
    cfg = ExtractConfig()
    g = extract_graph(_loop_mask(), config=cfg)
    nxg = to_networkx(g, multigraph=True)
    loops = list(nx.selfloop_edges(nxg, data=True, keys=True))
    assert loops
    _, _, _, attrs = loops[0]
    assert attrs["is_self_loop"] is True


def test_to_networkx_multiedge_semantics():
    n0 = gnode(0, (0.0, 0.0, 0.0), (0, 0))
    n1 = gnode(1, (1.0, 0.0, 0.0), (0, 1))
    e0 = gedge(0, 0, 1)
    e1 = gedge(1, 0, 1)
    graph = MaskGraph(nodes=[n0, n1], edges=[e0, e1], meta=extract_graph(_line_mask()).meta)

    multi = to_networkx(graph, multigraph=True)
    simple = to_networkx(graph, multigraph=False)

    assert multi.number_of_edges() == 2
    assert simple.number_of_edges() == 1


def gnode(node_id: int, xyz: tuple[float, float, float], index: tuple[int, int]):
    from maskgraph.types import Node

    return Node(
        id=node_id,
        xyz=xyz,
        index=index,
        type="endpoint",
        degree=1,
        voxel_count=1,
    )


def gedge(edge_id: int, u: int, v: int) -> Edge:
    path_index = np.asarray([[0, 0], [0, 1]], dtype=np.int32)
    path_xyz = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    return Edge(
        id=edge_id,
        u=u,
        v=v,
        path_xyz=path_xyz,
        path_index=path_index,
        length=1.0,
        voxel_length=2,
        is_self_loop=False,
    )
