from __future__ import annotations

import numpy as np

from mask2graph import (
    ExtractConfig,
    GraphAugmentationPipeline,
    RandomCrop,
    RandomFlip,
    RandomRotation,
    compose_transforms,
    crop_graph_box,
    flip_graph,
    flip_transform,
    graph_to_min_ipd,
    mask_to_graph,
    permute_axes,
    result_to_json,
    rotate_graph,
    rotation_transform,
    translate_graph,
    translation_transform,
)
from mask2graph.transforms import apply_transform


def _line2() -> np.ndarray:
    m = np.zeros((11, 13), dtype=np.uint8)
    m[5, 2:11] = 1
    return m


def _diag2() -> np.ndarray:
    m = np.zeros((12, 12), dtype=np.uint8)
    for i in range(2, 10):
        m[i, i] = 1
    return m


def _t2() -> np.ndarray:
    m = np.zeros((13, 13), dtype=np.uint8)
    m[6, 2:11] = 1
    m[2:7, 6] = 1
    return m


def _ring2() -> np.ndarray:
    m = np.zeros((14, 14), dtype=np.uint8)
    m[3, 3:11] = 1
    m[10, 3:11] = 1
    m[3:11, 3] = 1
    m[3:11, 10] = 1
    return m


def _line3() -> np.ndarray:
    m = np.zeros((11, 11, 13), dtype=np.uint8)
    m[5, 5, 2:11] = 1
    return m


def _coords(result):
    return np.asarray([n.xyz for n in result.topology_graph.nodes], dtype=float)


def test_exact_2d_quarter_rotations_and_flip_are_involutions():
    r = mask_to_graph(_t2())
    source_hash = r.topology_graph.meta.graph_hash
    q = r
    for _ in range(4):
        q = rotate_graph(q, 90)
    assert q.topology_graph.meta.graph_hash == source_hash
    assert q.diagnostics.logical_beta0 == r.diagnostics.logical_beta0
    assert q.diagnostics.logical_beta1 == r.diagnostics.logical_beta1

    f = flip_graph(flip_graph(r, "x"), "x")
    assert f.topology_graph.meta.graph_hash == source_hash


def test_arbitrary_rotation_preserves_topology_and_metric_but_marks_nonexact():
    r = mask_to_graph(_t2())
    lengths = [e.length for e in r.topology_graph.edges]
    radii = [e.radius_mean for e in r.topology_graph.edges]
    out = rotate_graph(r, 13.0)
    assert not out.topology_graph.meta.transform_exact
    assert not out.topology_graph.meta.grid_aligned
    assert all(not n.index_valid for n in out.topology_graph.nodes)
    assert [n.source_index for n in out.topology_graph.nodes] == [n.source_index for n in r.topology_graph.nodes]
    assert np.allclose([e.length for e in out.topology_graph.edges], lengths)
    assert [e.radius_mean for e in out.topology_graph.edges] == radii
    assert (out.diagnostics.logical_beta0, out.diagnostics.logical_beta1) == (
        r.diagnostics.logical_beta0,
        r.diagnostics.logical_beta1,
    )


def test_composed_transform_matches_sequential_transform():
    r = mask_to_graph(_t2())
    t1 = rotation_transform(r.topology_graph, 90)
    t2 = flip_transform(r.topology_graph, "x")
    t3 = translation_transform(r.topology_graph, (4.0, -2.0))
    seq = apply_transform(apply_transform(apply_transform(r, t1), t2), t3)
    combined = apply_transform(r, compose_transforms([t1, t2, t3]))
    assert np.allclose(_coords(seq), _coords(combined))
    for a, b in zip(seq.topology_graph.edges, combined.topology_graph.edges):
        assert np.allclose(a.path_xyz, b.path_xyz)


def test_translation_grid_alignment_is_detected_not_assumed():
    r = mask_to_graph(_line2())
    integer = translate_graph(r, (2.0, -3.0))
    assert integer.topology_graph.meta.grid_aligned
    assert all(n.index_valid for n in integer.topology_graph.nodes)
    half = translate_graph(r, (0.5, 0.0))
    assert not half.topology_graph.meta.grid_aligned
    assert all(not n.index_valid for n in half.topology_graph.nodes)


def test_3d_axis_rotations_inverse_and_permutation():
    r = mask_to_graph(_line3())
    rotated = rotate_graph(r, 90, axis=(0, 1, 0), center="origin")
    inv = rotate_graph(rotated, -90, axis=(0, 1, 0), center="origin")
    assert inv.topology_graph.meta.graph_hash == r.topology_graph.meta.graph_hash
    p = permute_axes(r, "zxy")
    assert p.topology_graph.meta.transform_exact
    assert (p.diagnostics.logical_beta0, p.diagnostics.logical_beta1) == (1, 0)


def test_crop_crossing_line_creates_boundary_endpoints_and_provenance():
    r = mask_to_graph(_line2())
    out = crop_graph_box(r, bounds=(4.0, 8.0, 4.0, 6.0))
    assert len(out.topology_graph.edges) == 1
    assert len(out.topology_graph.nodes) == 2
    assert all(n.on_crop_boundary for n in out.topology_graph.nodes)
    assert all(n.type == "boundary_endpoint" for n in out.topology_graph.nodes)
    e = out.topology_graph.edges[0]
    assert e.source_edge_id is not None
    assert e.source_arc_range is not None
    assert np.all(e.path_xyz[:, 0] >= 4.0) and np.all(e.path_xyz[:, 0] <= 8.0)
    assert out.diagnostics.topology_before_beta0 == 1
    assert out.diagnostics.topology_after_beta0 == 1


def test_crop_non_lattice_intersection_never_fakes_pixel_index():
    r = mask_to_graph(_diag2())
    out = crop_graph_box(r, bounds=(2.5, 7.5, 2.5, 7.5))
    created = [n for n in out.topology_graph.nodes if n.created_by == "crop"]
    assert created
    assert any(not n.index_valid and n.source_index is None for n in created)
    assert any(not e.path_index_valid for e in out.topology_graph.edges)


def test_crop_fully_inside_preserves_cycle_and_cut_cycle_becomes_open():
    r = mask_to_graph(_ring2())
    full = crop_graph_box(r, bounds=(2.0, 11.0, 2.0, 11.0))
    assert full.diagnostics.logical_beta1 == 1
    cut = crop_graph_box(r, bounds=(2.0, 7.0, 2.0, 11.0))
    assert cut.diagnostics.logical_beta1 == 0
    assert all(not e.is_self_loop for e in cut.topology_graph.edges)


def test_crop_fully_outside_is_empty_and_valid():
    r = mask_to_graph(_line2())
    out = crop_graph_box(r, bounds=(100.0, 110.0, 100.0, 110.0))
    assert not out.topology_graph.nodes
    assert not out.topology_graph.edges
    assert out.diagnostics.logical_beta0 == 0
    assert out.diagnostics.logical_beta1 == 0


def test_crop_keep_size_is_crop_plus_translation():
    r = mask_to_graph(_line2())
    out = crop_graph_box(r, bounds=(4.0, 8.0, 4.0, 6.0), keep_size=True)
    xs = [v.xyz[0] for v in out.embedded_graph.vertices]
    ys = [v.xyz[1] for v in out.embedded_graph.vertices]
    assert min(xs) >= -1e-9 and max(xs) <= 4.0 + 1e-9
    assert min(ys) >= -1e-9 and max(ys) <= 2.0 + 1e-9
    assert [h["type"] for h in out.topology_graph.meta.augmentation_history[-2:]] == ["crop", "translation"]


def test_junction_exactly_on_crop_boundary_remains_junction():
    r = mask_to_graph(_t2())
    junction = next(n for n in r.topology_graph.nodes if n.type == "junction")
    x, y = junction.xyz[:2]
    out = crop_graph_box(r, bounds=(x, x + 5.0, y - 5.0, y + 5.0))
    js = [n for n in out.topology_graph.nodes if np.allclose(n.xyz[:2], (x, y))]
    assert js
    assert js[0].type == "junction"
    assert js[0].on_crop_boundary


def test_3d_box_crop_clips_full_branch():
    r = mask_to_graph(_line3())
    out = crop_graph_box(r, bounds=(4.0, 8.0, 4.0, 6.0, 4.0, 6.0))
    assert len(out.topology_graph.edges) == 1
    xyz = out.topology_graph.edges[0].path_xyz
    assert np.all(xyz[:, 0] >= 4.0) and np.all(xyz[:, 0] <= 8.0)
    assert all(n.on_crop_boundary for n in out.topology_graph.nodes)


def test_random_pipeline_is_seed_deterministic():
    r = mask_to_graph(_t2())
    pipeline = GraphAugmentationPipeline([
        RandomFlip(("x", "y"), probability=0.5),
        RandomRotation((0.0, 90.0, 180.0, 270.0)),
        RandomCrop((5.0, 5.0), keep_size=True),
    ])
    a = pipeline(r, seed=12345)
    b = pipeline(r, seed=12345)
    assert result_to_json(a) == result_to_json(b)
    assert a.topology_graph.meta.augmentation_history
    assert any("augmentation_seed" in h for h in a.topology_graph.meta.augmentation_history)


def test_arbitrary_rotation_rationalized_export_reports_approximation():
    r = rotate_graph(mask_to_graph(_line2()), 37.0)
    cfg = ExtractConfig()
    cfg.export.coordinate_mode = "rationalized"
    cfg.export.max_denominator = 10000
    inst = graph_to_min_ipd(r.embedded_graph, config=cfg.export, name="rotated")
    assert inst["metadata"]["transform_exact"] is False
    assert inst["metadata"]["grid_aligned"] is False
    assert inst["metadata"]["rationalization_error"] >= 0.0
    assert inst["metadata"]["augmentation_history"]


def test_all_2d_quarter_turns_are_exact_and_topology_preserving():
    r = mask_to_graph(_t2())
    for angle in (0.0, 90.0, 180.0, 270.0):
        out = rotate_graph(r, angle)
        assert out.topology_graph.meta.transform_exact
        assert (out.diagnostics.logical_beta0, out.diagnostics.logical_beta1) == (
            r.diagnostics.logical_beta0, r.diagnostics.logical_beta1
        )


def test_3d_flips_and_coordinate_axis_rotations_are_exact():
    r = mask_to_graph(_line3())
    for axes in ("x", "y", "z", "xy", "xz", "yz", "xyz"):
        out = flip_graph(r, axes)
        assert out.topology_graph.meta.transform_exact
        assert (out.diagnostics.logical_beta0, out.diagnostics.logical_beta1) == (1, 0)
    for axis in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        out = rotate_graph(r, 90, axis=axis, center="origin")
        assert out.topology_graph.meta.transform_exact


def test_3d_quaternion_and_matrix_rotation_interfaces():
    r = mask_to_graph(_line3())
    # 90 degrees around z: q=(cos45,0,0,sin45)
    s = float(np.sqrt(0.5))
    q = rotate_graph(r, quaternion=(s, 0.0, 0.0, s), center="origin")
    m = rotate_graph(r, matrix=((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)), center="origin")
    for a, b in zip(q.topology_graph.edges, m.topology_graph.edges):
        assert np.allclose(a.path_xyz, b.path_xyz, atol=1e-9)


def test_explicit_transform_inverse_recovers_exact_transform_geometry():
    r = mask_to_graph(_t2())
    t = rotation_transform(r.topology_graph, 90, center="origin")
    out = apply_transform(apply_transform(r, t), t.inverse())
    assert out.topology_graph.meta.graph_hash == r.topology_graph.meta.graph_hash


def test_arbitrary_rotation_angles_preserve_lengths_and_topology():
    r = mask_to_graph(_t2())
    lengths = np.asarray([e.length for e in r.topology_graph.edges])
    for angle in (13.0, 37.0, 121.0):
        out = rotate_graph(r, angle)
        assert not out.topology_graph.meta.transform_exact
        assert np.allclose([e.length for e in out.topology_graph.edges], lengths)
        assert (out.diagnostics.logical_beta0, out.diagnostics.logical_beta1) == (
            r.diagnostics.logical_beta0, r.diagnostics.logical_beta1
        )


def test_crop_line_fully_inside_preserves_geometry_and_line_outside_disappears():
    r = mask_to_graph(_line2())
    inside = crop_graph_box(r, bounds=(1.0, 11.0, 4.0, 6.0))
    assert inside.topology_graph.meta.graph_hash == r.topology_graph.meta.graph_hash
    outside = crop_graph_box(r, bounds=(-10.0, -1.0, -10.0, -1.0))
    assert not outside.topology_graph.edges


def test_cycle_can_produce_multiple_crop_fragments_from_one_source_edge():
    r = mask_to_graph(_ring2())
    out = crop_graph_box(r, bounds=(5.0, 8.0, 2.0, 11.0))
    assert len(out.topology_graph.edges) == 2
    assert out.diagnostics.logical_beta0 == 2
    assert out.diagnostics.logical_beta1 == 0
    assert {e.source_edge_id for e in out.topology_graph.edges} == {0}
    assert {e.crop_fragment_index for e in out.topology_graph.edges} == {0, 1}


def test_crop_boundary_segment_and_corner_entry_are_kept_without_zero_edges():
    r = mask_to_graph(_line2())
    # Original line y=5 lies exactly on the lower crop boundary.
    on_face = crop_graph_box(r, bounds=(3.0, 9.0, 5.0, 7.0))
    assert len(on_face.topology_graph.edges) == 1
    assert all(abs(p[1] - 5.0) < 1e-9 for p in on_face.topology_graph.edges[0].path_xyz)

    d = mask_to_graph(_diag2())
    corner = crop_graph_box(d, bounds=(4.5, 8.0, 4.5, 8.0))
    assert corner.topology_graph.edges
    first = corner.topology_graph.edges[0].path_xyz[0]
    assert np.allclose(first[:2], (4.5, 4.5))


def test_tangent_only_segment_contact_is_not_emitted_as_edge():
    from mask2graph.clip import clip_segment_box
    lo = np.array([0.0, 0.0])
    hi = np.array([1.0, 1.0])
    # Segment touches the box only at the lower-left corner.
    clipped = clip_segment_box(np.array([-1.0, 1.0, 0.0]), np.array([1.0, -1.0, 0.0]), lo, hi)
    assert clipped is None


def test_3d_planar_loop_crop_preserves_or_breaks_cycle_correctly():
    m = np.zeros((12, 14, 14), dtype=np.uint8)
    z = 5
    m[z, 3, 3:11] = 1
    m[z, 10, 3:11] = 1
    m[z, 3:11, 3] = 1
    m[z, 3:11, 10] = 1
    r = mask_to_graph(m)
    assert r.diagnostics.logical_beta1 == 1
    full = crop_graph_box(r, bounds=(2.0, 11.0, 2.0, 11.0, 4.0, 6.0))
    assert full.diagnostics.logical_beta1 == 1
    cut = crop_graph_box(r, bounds=(5.0, 8.0, 2.0, 11.0, 4.0, 6.0))
    assert cut.diagnostics.logical_beta1 == 0


def test_random_pipeline_records_seed_and_full_reproducibility_metadata():
    r = mask_to_graph(_t2())
    pipeline = GraphAugmentationPipeline([RandomFlip(("x",), 1.0), RandomRotation((90.0,))])
    out = pipeline(r, seed=77)
    meta = out.topology_graph.meta
    assert meta.source_graph_hash == r.topology_graph.meta.graph_hash
    assert meta.augmentation_seed == 77
    assert meta.graph_hash
    assert len(meta.augmentation_history) == 2
    assert all(h.get("augmentation_seed") == 77 for h in meta.augmentation_history)


def test_crop_attached_self_loop_splits_at_real_junction_not_through_it():
    m = np.zeros((16, 16), dtype=np.uint8)
    m[3, 3:11] = 1
    m[10, 3:11] = 1
    m[3:11, 3] = 1
    m[3:11, 10] = 1
    m[6, 10:15] = 1
    r = mask_to_graph(m)
    junction = next(n for n in r.topology_graph.nodes if n.type == "junction")
    x, y = junction.xyz[:2]
    out = crop_graph_box(r, bounds=(x - 2.0, x + 3.0, y - 3.0, y + 3.0))
    j2 = next(n for n in out.topology_graph.nodes if n.type == "junction")
    # The cropped loop sides and tail all remain incident to the junction rather
    # than being represented as boundary-to-boundary edges that pass through it.
    assert j2.degree == 3
    assert sum(1 for e in out.topology_graph.edges if e.u == j2.id or e.v == j2.id) == 3


def test_augmented_full_result_json_round_trip_is_byte_stable():
    from mask2graph import result_from_json
    r = mask_to_graph(_t2())
    out = crop_graph_box(rotate_graph(r, 37.0), bounds=(3.0, 9.0, 2.0, 10.0))
    raw = result_to_json(out)
    restored = result_from_json(raw)
    assert result_to_json(restored) == raw
