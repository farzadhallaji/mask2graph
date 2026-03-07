from __future__ import annotations

import numpy as np
import pytest

from maskgraph import ExtractConfig, ExtractionError, extract_graph, extract_raw_graph, from_json, to_json
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
    cfg.cleanup.remove_objects_max_size = 1
    g = extract_graph(m, config=cfg)
    assert len(g.edges) == 1


def test_serialization_round_trip():
    g = extract_graph(_line_mask_2d(9))
    payload = to_json(g)
    g2 = from_json(payload)
    assert to_json(g2) == payload


def test_impossible_traversal_raises():
    m = np.zeros((9, 9), dtype=np.uint8)
    m[4, 2:7] = 1
    m[2:7, 4] = 1
    sk = skeletonize_mask(m.astype(bool), ExtractConfig().skeleton)
    bad_degree = np.where(sk, 2, 0).astype(np.int32)
    with pytest.raises(ExtractionError):
        extract_raw_graph(sk, bad_degree, spacing=(1.0, 1.0))
