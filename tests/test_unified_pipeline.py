from __future__ import annotations

import itertools

import numpy as np

from mask2graph import (
    ExtractConfig,
    crop_graph_connected_subgraph,
    mask_to_graph,
    mask_to_min_ipd,
    to_json,
)
from mask2graph.simplify import max_subpath_error, optimal_keep_indices
from mask2graph.types import Edge


def _t_mask() -> np.ndarray:
    m = np.zeros((11, 11), dtype=np.uint8)
    m[5, 2:9] = 1
    m[2:6, 5] = 1
    return m


def _ring() -> np.ndarray:
    m = np.zeros((12, 12), dtype=np.uint8)
    m[3, 3:9] = 1
    m[8, 3:9] = 1
    m[3:9, 3] = 1
    m[3:9, 8] = 1
    return m


def test_junction_anchor_is_supported_and_edge_geometry_matches_nodes():
    r = mask_to_graph(_t_mask())
    g = r.topology_graph
    by_id = {n.id: n for n in g.nodes}
    junction = next(n for n in g.nodes if n.type == "junction")
    assert junction.support_indices is not None
    assert tuple(junction.index) in {tuple(int(v) for v in row) for row in junction.support_indices}
    assert len(junction.junction_mst_edges) == len(junction.support_indices) - 1
    for e in g.edges:
        assert np.allclose(e.path_xyz[0], by_id[e.u].xyz)
        assert np.allclose(e.path_xyz[-1], by_id[e.v].xyz)


def test_mst_junction_cleanup_removes_digital_artifact_cycles():
    r = mask_to_graph(_t_mask())
    d = r.diagnostics
    assert d.raw_beta1 > d.cleaned_beta1
    assert d.cleaned_beta1 == d.logical_beta1 == d.embedded_beta1 == 0
    assert d.coverage_complete


def test_simplification_is_non_destructive_and_within_error():
    m = np.zeros((32, 32), dtype=np.uint8)
    for x in range(3, 28):
        y = int(round(15 + 5 * np.sin(x / 5.0)))
        m[y, x] = 1
    cfg = ExtractConfig()
    cfg.simplify.epsilon = 1.25
    r = mask_to_graph(m, config=cfg)
    e = max(r.topology_graph.edges, key=lambda x: len(x.path_index))
    assert e.simplified_xyz is not None
    assert len(e.path_index) >= len(e.simplified_xyz)
    assert len(e.path_index) > 2
    assert r.diagnostics.max_simplification_error <= cfg.simplify.epsilon + 1e-9


def test_optimal_dag_matches_bruteforce_minimum_segment_count():
    pts = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [2.0, 0.0, 0.0], [3.0, 1.0, 0.0], [4.0, 1.0, 0.0]],
        dtype=float,
    )
    idx = np.asarray([[0, i] for i in range(len(pts))], dtype=np.int32)
    e = Edge(id=0, u=0, v=1, path_xyz=pts, path_index=idx, length=5.0, voxel_length=len(pts))
    eps = 0.2
    keep = optimal_keep_indices(e, eps)
    best = len(pts) - 1
    interior = list(range(1, len(pts) - 1))
    for r in range(len(interior) + 1):
        for chosen in itertools.combinations(interior, r):
            order = (0, *chosen, len(pts) - 1)
            if all(max_subpath_error(pts, a, b) <= eps for a, b in zip(order[:-1], order[1:])):
                best = min(best, len(order) - 1)
    assert len(keep) - 1 == best


def test_cycle_expands_to_non_self_loop_pslg_and_preserves_beta1():
    r = mask_to_graph(_ring())
    assert r.diagnostics.logical_beta1 == 1
    assert r.diagnostics.embedded_beta1 == 1
    assert len(r.embedded_graph.segments) >= 3
    assert all(s.u != s.v for s in r.embedded_graph.segments)


def test_boundary_endpoint_is_explicit():
    m = np.zeros((9, 9), dtype=np.uint8)
    m[4, 0:7] = 1
    r = mask_to_graph(m)
    boundary = [n for n in r.topology_graph.nodes if n.on_image_boundary]
    assert boundary
    assert any(n.type == "boundary_endpoint" for n in boundary)
    assert any(n.boundary_axes for n in boundary)


def test_3d_separated_crossings_remain_two_components():
    m = np.zeros((10, 11, 11), dtype=np.uint8)
    m[2, 5, 2:9] = 1
    m[7, 2:9, 5] = 1
    r = mask_to_graph(m)
    assert r.topology_graph.meta.ndim == 3
    assert r.diagnostics.logical_beta0 == 2
    assert len(r.topology_graph.edges) == 2


def test_connected_crop_is_deterministic_and_bounded():
    r = mask_to_graph(_t_mask())
    g = crop_graph_connected_subgraph(r.topology_graph, seed_node=0, max_edges=2)
    assert len(g.edges) <= 2
    assert to_json(g) == to_json(crop_graph_connected_subgraph(r.topology_graph, seed_node=0, max_edges=2))


def test_min_ipd_export_2d_and_3d_schema():
    m2 = np.zeros((9, 9), dtype=np.uint8)
    m2[4, 2:7] = 1
    d2 = mask_to_min_ipd(m2, name="line2")
    assert d2["schema"] == "min_ipd_instance_v1"
    assert len(d2["segments"]) == 1
    assert len(d2["outer_boundary"]) == 4

    m3 = np.zeros((9, 9, 9), dtype=np.uint8)
    m3[4, 4, 2:7] = 1
    d3 = mask_to_min_ipd(m3, name="line3")
    assert d3["schema"] == "min_ipd_instance3_v1"
    assert len(d3["segments"]) == 1
    assert len(d3["domain"]["halfspaces"]) == 6


def test_hash_and_serialization_are_deterministic():
    r1 = mask_to_graph(_t_mask())
    r2 = mask_to_graph(_t_mask())
    assert r1.topology_graph.meta.graph_hash
    assert r1.topology_graph.meta.graph_hash == r2.topology_graph.meta.graph_hash
    assert to_json(r1.topology_graph) == to_json(r2.topology_graph)


def test_geometry_profiles_and_radius_extrema_are_populated():
    r = mask_to_graph(_t_mask())
    for e in r.topology_graph.edges:
        assert e.arclen_profile is not None and len(e.arclen_profile) == len(e.path_xyz)
        assert e.tangent_profile is not None and len(e.tangent_profile) == len(e.path_xyz)
        assert e.turning_angle_profile is not None and len(e.turning_angle_profile) == len(e.path_xyz)
        assert e.curvature_profile is not None and len(e.curvature_profile) == len(e.path_xyz)
        assert e.radius_min is not None and e.radius_max is not None
        assert e.radius_min <= e.radius_max


def test_scaled_exact_export_supports_fractional_spacing():
    m = np.zeros((7, 7), dtype=np.uint8)
    m[3, 1:6] = 1
    cfg = ExtractConfig()
    cfg.export.coordinate_mode = "scaled_exact"
    d = mask_to_min_ipd(m, spacing=(0.5, 0.25), config=cfg)
    flat = [x for p in d["points"] for x in p]
    assert any(isinstance(x, str) and "/" in x for x in flat)


def test_legacy_cluster_resolution_remains_supported_but_safe():
    cfg = ExtractConfig()
    cfg.junction.resolution = "cluster"
    r = mask_to_graph(_t_mask(), config=cfg)
    assert r.diagnostics.cleaned_beta1 == r.diagnostics.logical_beta1
    junction = next(n for n in r.topology_graph.nodes if n.type == "junction")
    assert len(junction.junction_mst_edges) == len(junction.support_indices) - 1


def test_default_configuration_marks_no_graph_topology_edit_policy():
    r = mask_to_graph(_t_mask())
    assert not r.diagnostics.cleanup_topology_edit_enabled


def test_three_dimensional_profiles_and_embedding_validate():
    m = np.zeros((11, 11, 11), dtype=np.uint8)
    m[5, 5, 2:9] = 1
    m[2:6, 5, 5] = 1
    r = mask_to_graph(m)
    assert r.topology_graph.meta.ndim == 3
    assert r.diagnostics.coverage_complete
    assert r.diagnostics.embedded_beta0 == r.diagnostics.logical_beta0
    assert r.diagnostics.embedded_beta1 == r.diagnostics.logical_beta1
