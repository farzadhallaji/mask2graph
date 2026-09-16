from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from mask2graph import ConfigError, load_run_config, run_experiment


def _extract_payload() -> dict:
    return {
        "cleanup": {"min_object_size": 0.0, "max_hole_size": 0.0, "max_hole_radius": 0.0, "enabled": True},
        "skeleton": {"method_2d": "zhang", "method_3d": "lee"},
        "junction": {"resolution": "mst", "dilation_iters": 0, "supported_anchor": True},
        "normalize": {
            "junction_dilation_iters": 0,
            "min_component_length": 0.0,
            "prune_spurs_below": 0.0,
            "min_cycle_length": 0.0,
            "max_cycle_area": 0.0,
            "cycle_length_to_radius_ratio": 0.0,
            "contract_short_edges_below": 0.0,
            "normalization_max_iter": 10,
            "prune_iterations": 100,
            "contract_degree2": True,
        },
        "geometry": {"tangent_window": 2, "compute_curvature": True},
        "simplify": {
            "enabled": True,
            "epsilon": 0.5,
            "method": "rdp",
            "protect_angle_degrees": 45.0,
            "topology_guard": True,
            "fallback_to_original": True,
        },
        "validation": {
            "enabled": True,
            "validate_coverage": True,
            "validate_topology": True,
            "validate_embedding": True,
            "tolerance": 1.0e-9,
        },
        "export": {"coordinate_mode": "rationalized", "max_denominator": 1000000, "domain_margin": 1.0},
        "determinism": {"float_decimals": 6, "sort_nodes": True, "sort_edges": True},
    }


def _payload(tmp_path: Path, *, ndim: int = 2, run_name: str = "run") -> tuple[dict, Path]:
    shape = (9, 9) if ndim == 2 else (7, 7, 7)
    mask = np.zeros(shape, dtype=np.uint8)
    if ndim == 2:
        mask[4, 1:8] = 1
    else:
        mask[3, 3, 1:6] = 1
    mask_path = tmp_path / f"mask{ndim}d.npy"
    np.save(mask_path, mask)
    payload = {
        "schema_version": 1,
        "name": f"strict_{ndim}d",
        "input": {
            "path": mask_path.name,
            "expected_ndim": ndim,
            "spacing": [1.0] * ndim,
            "foreground_value": 1,
            "background_value": 0,
            "require_only_configured_values": True,
            "squeeze_singleton_axes": False,
            "channel": None,
            "npz_array_key": None,
        },
        "extract": _extract_payload(),
        "augmentation": {"enabled": True, "seed": 17, "operations": []},
        "visualization": {
            "enabled": False,
            "representation": "embedded",
            "show_nodes": False,
            "dpi": 100,
            "save_original_mask": False,
            "save_graph_overlay": False,
            "save_each_stage": False,
            "save_sequence": False,
        },
        "output": {
            "run_dir": run_name,
            "overwrite": False,
            "save_stage_graph_json": True,
            "save_final_graph_json": True,
            "save_min_ipd_json": True,
            "min_ipd_name": "strict_smoke",
        },
    }
    yaml_path = tmp_path / f"config_{run_name}.yaml"
    return payload, yaml_path


def _write(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_demo_yaml_is_strict_and_self_contained():
    root = Path(__file__).resolve().parents[1]
    cfg = load_run_config(root / "configs" / "retinal_augmentation_demo.yaml")
    assert cfg.input.expected_ndim == 2
    assert cfg.augmentation.seed == 1234
    assert [op["type"] for op in cfg.augmentation.operations] == ["rotate", "flip", "central_crop"]


def test_unknown_root_key_fails(tmp_path: Path):
    payload, path = _payload(tmp_path)
    payload["typo"] = 1
    _write(path, payload)
    with pytest.raises(ConfigError, match="unknown keys"):
        load_run_config(path)


def test_unknown_nested_key_fails(tmp_path: Path):
    payload, path = _payload(tmp_path)
    payload["extract"]["simplify"]["epslion"] = 1.0
    _write(path, payload)
    with pytest.raises(ConfigError, match="unknown keys"):
        load_run_config(path)


def test_missing_required_key_fails(tmp_path: Path):
    payload, path = _payload(tmp_path)
    del payload["extract"]["validation"]["validate_embedding"]
    _write(path, payload)
    with pytest.raises(ConfigError, match="missing required keys"):
        load_run_config(path)


def test_wrong_scalar_type_fails_without_coercion(tmp_path: Path):
    payload, path = _payload(tmp_path)
    payload["extract"]["geometry"]["tangent_window"] = "2"
    _write(path, payload)
    with pytest.raises(ConfigError, match="must be int"):
        load_run_config(path)


def test_three_dimensional_config_is_first_class(tmp_path: Path):
    payload, path = _payload(tmp_path, ndim=3)
    payload["augmentation"]["operations"] = [
        {"type": "rotate", "angle_degrees": 90.0, "center": "bbox_center", "axis_3d": [0.0, 0.0, 1.0]},
        {"type": "flip", "axes": ["z"], "center": "bbox_center"},
    ]
    _write(path, payload)
    cfg = load_run_config(path)
    assert cfg.input.spacing == (1.0, 1.0, 1.0)
    assert cfg.augmentation.operations[0]["axis_3d"] == (0.0, 0.0, 1.0)


def test_3d_mask_plot_policy_fails_before_execution(tmp_path: Path):
    payload, path = _payload(tmp_path, ndim=3)
    payload["visualization"]["enabled"] = True
    payload["visualization"]["save_original_mask"] = True
    _write(path, payload)
    with pytest.raises(ConfigError, match="3D visualization"):
        load_run_config(path)


def test_yaml_run_archives_provenance_and_outputs(tmp_path: Path):
    payload, path = _payload(tmp_path, run_name="archive")
    payload["augmentation"]["operations"] = [
        {"type": "flip", "axes": ["x"], "center": "bbox_center"}
    ]
    _write(path, payload)
    run = run_experiment(path)
    out = run.run_dir
    assert (out / "source.yaml").is_file()
    assert (out / "resolved.yaml").is_file()
    assert (out / "environment.json").is_file()
    assert (out / "code_snapshot" / "mask2graph" / "runner.py").is_file()
    assert (out / "stages" / "00_extracted.json").is_file()
    assert (out / "stages" / "01_flip.json").is_file()
    assert (out / "final_graph.json").is_file()
    assert (out / "min_ipd.json").is_file()
    assert (out / "summary.json").is_file()
    assert run.final.topology_graph.meta.augmentation_seed is None  # deterministic op uses declared seed only at run level


def test_seeded_random_yaml_is_reproducible(tmp_path: Path):
    p1, y1 = _payload(tmp_path, run_name="r1")
    p1["augmentation"]["operations"] = [
        {"type": "random_flip", "axes": ["x", "y"], "probability": 0.5, "center": "bbox_center"},
        {"type": "random_rotation", "angles_degrees": [0.0, 90.0, 180.0, 270.0], "axes_3d": None, "center": "bbox_center"},
    ]
    _write(y1, p1)
    r1 = run_experiment(y1)

    p2 = deepcopy(p1)
    p2["output"]["run_dir"] = "r2"
    y2 = tmp_path / "config_r2.yaml"
    _write(y2, p2)
    r2 = run_experiment(y2)
    assert r1.final.topology_graph.meta.graph_hash == r2.final.topology_graph.meta.graph_hash
    assert r1.final.topology_graph.meta.augmentation_history == r2.final.topology_graph.meta.augmentation_history


def test_input_values_are_not_silently_thresholded(tmp_path: Path):
    payload, path = _payload(tmp_path)
    mask_path = tmp_path / payload["input"]["path"]
    mask = np.load(mask_path)
    mask[0, 0] = 2
    np.save(mask_path, mask)
    _write(path, payload)
    with pytest.raises(ValueError, match="not declared"):
        run_experiment(path)
