from __future__ import annotations

import numpy as np
import pytest

from maskgraph import (
    ExtractConfig,
    ExtractionError,
    extract_graph,
    extract_raw_graph,
    from_json,
    to_json,
)
from maskgraph.nodes import node_candidates_from_degree
from maskgraph.utils.connectivity import label_components
from maskgraph.skeleton import skeletonize_mask


def _line_mask_2d(length: int = 9) -> np.ndarray:
    m = np.zeros((length, length), dtype=np.uint8)
    m[length // 2, 1:-1] = 1
    return m


def test_straight_line():
    g = extract_graph(_line_mask_2d())
    assert len(g.nodes) == 2
    assert len(g.edges) == 1
    assert sorted(n.type for n in g.nodes) == ["endpoint", "endpoint"]


def test_y_junction():
    m = np.zeros((11, 11), dtype=np.uint8)
    m[5, 2:9] = 1
    m[2:6, 5] = 1
    g = extract_graph(m)
    assert sum(n.type == "junction" for n in g.nodes) == 1
    assert sum(n.type == "endpoint" for n in g.nodes) == 3
    assert len(g.edges) == 3


def test_cross_clustered_center_single_junction():
    m = np.zeros((11, 11), dtype=np.uint8)
    m[5, 2:9] = 1
    m[2:9, 5] = 1
    m[4:7, 4:7] = 1
    g = extract_graph(m)
    assert sum(n.type == "junction" for n in g.nodes) == 1


def test_ring_loop_self_loop():
    m = np.zeros((12, 12), dtype=np.uint8)
    m[3, 3:9] = 1
    m[8, 3:9] = 1
    m[3:9, 3] = 1
    m[3:9, 8] = 1
    g = extract_graph(m)
    cycle_nodes = [n for n in g.nodes if n.type == "cycle"]
    assert len(cycle_nodes) == 1
    assert len(g.edges) == 1
    assert g.edges[0].is_self_loop


def test_two_loops_with_bridge_preserved():
    m = np.zeros((24, 24), dtype=np.uint8)
    m[4, 3:9] = 1
    m[9, 3:9] = 1
    m[4:10, 3] = 1
    m[4:10, 8] = 1
    m[14, 13:19] = 1
    m[19, 13:19] = 1
    m[14:20, 13] = 1
    m[14:20, 18] = 1
    m[9:15, 8] = 1
    m[14, 8:13] = 1
    g = extract_graph(m)
    assert len(g.edges) >= 3


def test_tiny_spur_pruning():
    m = _line_mask_2d(13)
    m[6, 6] = 1
    m[5, 6] = 1
    cfg = ExtractConfig()
    cfg.normalize.prune_spurs_below = 2.0
    cfg.normalize.prune_iterations = 20
    g = extract_graph(m, config=cfg)
    assert len(g.edges) == 1


def test_component_removal_by_total_length():
    m = np.zeros((20, 20), dtype=np.uint8)
    m[10, 2:16] = 1
    m[2, 2:4] = 1
    cfg = ExtractConfig()
    cfg.normalize.min_component_length = 5.0
    g = extract_graph(m, config=cfg)
    assert len(g.edges) == 1


def test_anisotropic_spacing_changes_length_and_radius():
    m = _line_mask_2d(9)
    g_iso = extract_graph(m, spacing=(1.0, 1.0))
    g_len = extract_graph(m, spacing=(1.0, 2.0))
    g_rad = extract_graph(m, spacing=(2.0, 1.0))
    assert g_iso.edges[0].length != g_len.edges[0].length
    assert g_iso.edges[0].radius_mean != g_rad.edges[0].radius_mean


def test_3d_t_junction_uses_26_neighbors():
    m = np.zeros((11, 11, 11), dtype=np.uint8)
    m[5, 5, 2:9] = 1
    m[2:6, 5, 5] = 1
    g = extract_graph(m)
    assert sum(n.type == "junction" for n in g.nodes) >= 1
    assert len(g.edges) >= 3


def test_junction_dilation_merges_nearby_junction_seeds():
    sk = np.zeros((7, 7), dtype=bool)
    sk[3, 1:6] = True
    degree = np.zeros((7, 7), dtype=np.int32)
    degree[3, 2] = 3
    degree[3, 4] = 3
    degree[3, 1] = 1
    degree[3, 5] = 1
    degree[3, 3] = 2

    plain = node_candidates_from_degree(sk, degree, junction_dilation_iters=0)
    dilated = node_candidates_from_degree(sk, degree, junction_dilation_iters=1)

    plain_junction_only = plain & (degree >= 3)
    dilated_junction_zone = dilated & (degree >= 2)
    _, n_plain = label_components(plain_junction_only)
    _, n_dilated = label_components(dilated_junction_zone)
    assert n_plain == 2
    assert n_dilated == 1


def test_determinism_same_json_bytes():
    m = _line_mask_2d(11)
    cfg = ExtractConfig()
    j1 = to_json(extract_graph(m, config=cfg))
    j2 = to_json(extract_graph(m, config=cfg))
    assert j1.encode("utf-8") == j2.encode("utf-8")


def test_cleanup_bool_only_behavior_and_documented_change():
    m = np.zeros((16, 16), dtype=np.uint8)
    m[8, 1:15] = 1
    m[2, 2] = 1
    cfg = ExtractConfig()
    cfg.cleanup.min_object_size = 2.0
    g = extract_graph(m, config=cfg)
    assert len(g.edges) == 1


def test_cleanup_fills_small_enclosed_hole_and_reports():
    m = np.zeros((15, 15), dtype=np.uint8)
    m[3:12, 3:12] = 1
    m[7, 7] = 0
    cfg = ExtractConfig()
    cfg.cleanup.max_hole_size = 2.0
    cfg.cleanup.max_hole_radius = 1.5
    _, debug = extract_graph(m, config=cfg, return_debug=True)
    assert bool(debug.mask_processed[7, 7])
    assert debug.cleanup_report is not None
    assert debug.cleanup_report.n_filled_holes == 1


def test_cleanup_does_not_fill_border_connected_background():
    m = np.ones((15, 15), dtype=np.uint8)
    m[:, 7] = 0
    cfg = ExtractConfig()
    cfg.cleanup.max_hole_size = 100.0
    cfg.cleanup.max_hole_radius = 10.0
    _, debug = extract_graph(m, config=cfg, return_debug=True)
    assert not bool(debug.mask_processed[0, 7])
    assert debug.cleanup_report is not None
    assert debug.cleanup_report.n_filled_holes == 0


def test_cleanup_hole_radius_gate_prevents_filling_long_slit():
    m = np.zeros((21, 21), dtype=np.uint8)
    m[2:19, 2:19] = 1
    m[5:16, 10] = 0
    cfg = ExtractConfig()
    cfg.cleanup.max_hole_size = 20.0
    cfg.cleanup.max_hole_radius = 0.6
    _, debug = extract_graph(m, config=cfg, return_debug=True)
    assert not bool(debug.mask_processed[10, 10])
    assert debug.cleanup_report is not None
    assert debug.cleanup_report.n_filled_holes == 0


def test_serialization_round_trip():
    g = extract_graph(_line_mask_2d(9))
    payload = to_json(g)
    g2 = from_json(payload)
    assert to_json(g2) == payload


def test_tiny_cycle_removed_with_length_area_radius_gates():
    m = np.zeros((10, 10), dtype=np.uint8)
    m[2, 2:8] = 1
    m[7, 2:8] = 1
    m[2:8, 2] = 1
    m[2:8, 7] = 1
    cfg = ExtractConfig()
    cfg.normalize.min_cycle_length = 100.0
    cfg.normalize.max_cycle_area = 100.0
    cfg.normalize.cycle_length_to_radius_ratio = 100.0
    cfg.normalize.contract_degree2 = False
    g = extract_graph(m, config=cfg)
    assert len(g.edges) == 0


def test_cycle_preserved_when_area_gate_blocks_removal():
    m = np.zeros((10, 10), dtype=np.uint8)
    m[2, 2:8] = 1
    m[7, 2:8] = 1
    m[2:8, 2] = 1
    m[2:8, 7] = 1
    cfg = ExtractConfig()
    cfg.normalize.min_cycle_length = 100.0
    cfg.normalize.max_cycle_area = 1.0
    cfg.normalize.cycle_length_to_radius_ratio = 100.0
    g = extract_graph(m, config=cfg)
    assert len(g.edges) == 1
    assert g.edges[0].is_self_loop


def test_impossible_traversal_raises():
    m = np.zeros((9, 9), dtype=np.uint8)
    m[4, 2:7] = 1
    m[2:7, 4] = 1
    sk = skeletonize_mask(m.astype(bool), ExtractConfig().skeleton)
    bad_degree = np.where(sk, 2, 0).astype(np.int32)
    with pytest.raises(ExtractionError):
        extract_raw_graph(sk, bad_degree, spacing=(1.0, 1.0))
