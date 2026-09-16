"""Single YAML-driven paper execution path for mask2graph."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from importlib import metadata as importlib_metadata
from typing import Any

import numpy as np
from skimage import io
import yaml

from .api import mask_to_graph
from .augment import RandomCrop, RandomFlip, RandomRotation
from .crop import crop_graph_box
from .paper_config import PaperRunConfig, dump_resolved_yaml, load_run_config
from .pslg import to_min_ipd, validate_min_ipd_export
from .serialize import result_to_json
from .transforms import flip_graph, permute_axes, rotate_graph, translate_graph
from .types import MaskGraphResult
from .visualize import central_crop_bounds, plot_mask, plot_result


@dataclass(frozen=True)
class RunStage:
    name: str
    result: MaskGraphResult


@dataclass
class PaperRunResult:
    config: PaperRunConfig
    mask: np.ndarray
    stages: list[RunStage]
    run_dir: Path
    min_ipd_instance: dict[str, Any] | None

    @property
    def original(self) -> MaskGraphResult:
        return self.stages[0].result

    @property
    def final(self) -> MaskGraphResult:
        return self.stages[-1].result


def _load_raw(path: Path, npz_array_key: str | None) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        if npz_array_key is not None:
            raise ValueError("input.npz_array_key must be null for .npy input")
        return np.asarray(np.load(path, allow_pickle=False))
    if suffix == ".npz":
        if npz_array_key is None:
            raise ValueError("input.npz_array_key is required for .npz input")
        data = np.load(path, allow_pickle=False)
        if npz_array_key not in data.files:
            raise ValueError(f"npz array key {npz_array_key!r} not found; available={data.files}")
        return np.asarray(data[npz_array_key])
    if npz_array_key is not None:
        raise ValueError("input.npz_array_key must be null for image input")
    return np.asarray(io.imread(path))


def load_configured_mask(config: PaperRunConfig) -> np.ndarray:
    """Load a mask without thresholding/coercion hidden from YAML."""
    inp = config.input
    if not inp.path.is_file():
        raise FileNotFoundError(inp.path)
    raw = _load_raw(inp.path, inp.npz_array_key)
    if inp.channel is not None:
        if raw.ndim < 1 or raw.shape[-1] <= inp.channel:
            raise ValueError(f"configured channel {inp.channel} unavailable for shape {raw.shape}")
        raw = raw[..., inp.channel]
    if inp.squeeze_singleton_axes:
        raw = np.squeeze(raw)
    if raw.ndim != inp.expected_ndim:
        raise ValueError(
            f"configured expected_ndim={inp.expected_ndim}, but loaded mask shape {raw.shape} has ndim={raw.ndim}"
        )
    if inp.foreground_value == inp.background_value:
        raise ValueError("foreground_value and background_value must differ")
    if inp.require_only_configured_values:
        unique = np.unique(raw)
        allowed = {inp.foreground_value, inp.background_value}
        bad = [v.item() if hasattr(v, "item") else v for v in unique if v not in allowed]
        if bad:
            raise ValueError(f"input contains values not declared as foreground/background: {bad[:16]}")
    if not np.any(raw == inp.foreground_value):
        raise ValueError("configured foreground_value is absent from input")
    return np.asarray(raw == inp.foreground_value, dtype=bool)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_info(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:  # pragma: no cover - git may be unavailable in installed wheels
            return None

    status = run("status", "--porcelain")
    diff = run("diff", "--no-ext-diff", "--binary")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
        "status_porcelain": status,
        "diff_sha256": None if diff is None else __import__("hashlib").sha256(diff.encode("utf-8")).hexdigest(),
    }


def _environment_metadata(root: Path) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("mask2graph", "numpy", "scipy", "scikit-image", "PyYAML", "matplotlib"):
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cwd": str(Path.cwd().resolve()),
        "packages": packages,
        "git": _git_info(root),
    }


def _prepare_run_dir(config: PaperRunConfig) -> Path:
    run_dir = config.output.run_dir
    root = _repo_root().resolve()
    dangerous = {Path("/").resolve(), root, root.parent}
    if run_dir.resolve() in dangerous:
        raise ValueError(f"refusing dangerous output.run_dir: {run_dir}")
    if run_dir.exists() and any(run_dir.iterdir()):
        if not config.output.overwrite:
            raise FileExistsError(f"run directory already exists and is nonempty: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _snapshot_code(run_dir: Path) -> None:
    root = _repo_root()
    dst = run_dir / "code_snapshot"
    dst.mkdir(parents=True, exist_ok=False)
    shutil.copytree(
        root / "mask2graph",
        dst / "mask2graph",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
    )
    for name in ("pyproject.toml", "CONFIG_POLICY.md", "IMPLEMENTATION_RULES.md", "README.md", "CHANGELOG.md"):
        src = root / name
        if src.is_file():
            shutil.copy2(src, dst / name)


def _archive_run_provenance(config: PaperRunConfig, run_dir: Path) -> None:
    shutil.copy2(config.source_yaml, run_dir / "source.yaml")
    (run_dir / "resolved.yaml").write_text(dump_resolved_yaml(config), encoding="utf-8")
    (run_dir / "environment.json").write_text(
        json.dumps(_environment_metadata(_repo_root()), indent=2, sort_keys=True), encoding="utf-8"
    )
    _snapshot_code(run_dir)


def _stage_name(i: int, op_type: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in op_type)
    return f"{i:02d}_{safe}"


def _apply_augmentation(config: PaperRunConfig, initial: MaskGraphResult) -> list[RunStage]:
    stages = [RunStage("00_extracted", initial)]
    aug = config.augmentation
    if not aug.enabled:
        return stages
    rng = np.random.default_rng(aug.seed)
    current = initial
    for i, op in enumerate(aug.operations, start=1):
        typ = op["type"]
        if typ == "rotate":
            current = rotate_graph(
                current,
                op["angle_degrees"],
                center=op["center"],
                degrees=True,
                axis=op["axis_3d"],
            )
        elif typ == "flip":
            current = flip_graph(current, op["axes"], center=op["center"])
        elif typ == "translate":
            current = translate_graph(current, op["offset"])
        elif typ == "axis_permutation":
            current = permute_axes(current, op["order"])
        elif typ == "crop_box":
            current = crop_graph_box(
                current,
                bounds=op["bounds"],
                keep_size=op["keep_size"],
                config=config.extract,
            )
        elif typ == "central_crop":
            bounds = central_crop_bounds(current, fraction=op["fraction"])
            current = crop_graph_box(
                current,
                bounds=bounds,
                keep_size=op["keep_size"],
                config=config.extract,
            )
        elif typ == "random_flip":
            current = RandomFlip(tuple(op["axes"]), probability=op["probability"], center=op["center"]).apply(
                current, rng, seed=aug.seed
            )
        elif typ == "random_rotation":
            axes = () if op["axes_3d"] is None else tuple(tuple(v) for v in op["axes_3d"])
            current = RandomRotation(
                tuple(op["angles_degrees"]), axes_3d=axes, center=op["center"], degrees=True
            ).apply(current, rng, seed=aug.seed)
        elif typ == "random_crop":
            current = RandomCrop(tuple(op["size"]), keep_size=op["keep_size"]).apply(
                current, rng, seed=aug.seed
            )
        else:  # validated before execution
            raise AssertionError(typ)
        stages.append(RunStage(_stage_name(i, typ), current))
    return stages


def _save_figures(config: PaperRunConfig, mask: np.ndarray, stages: list[RunStage], run_dir: Path) -> None:
    viz = config.visualization
    if not viz.enabled:
        return
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("visualization.enabled requires mask2graph[viz]") from exc
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if config.input.expected_ndim == 3:
        if viz.save_original_mask or viz.save_graph_overlay or viz.save_sequence:
            raise ValueError("3D runs must disable original-mask/overlay/sequence figures; save_each_stage supports 3D graphs")
    if viz.save_original_mask:
        ax = plot_mask(mask, spacing=config.input.spacing, title="Original binary mask")
        ax.figure.savefig(fig_dir / "00_original_mask.png", dpi=viz.dpi, bbox_inches="tight")
        plt.close(ax.figure)
    if viz.save_graph_overlay:
        ax = plot_result(
            stages[0].result,
            representation=viz.representation,
            background=mask,
            show_nodes=viz.show_nodes,
            title="Extracted graph over original mask",
        )
        ax.figure.savefig(fig_dir / "01_graph_overlay.png", dpi=viz.dpi, bbox_inches="tight")
        plt.close(ax.figure)
    if viz.save_each_stage:
        for stage in stages:
            ax = plot_result(
                stage.result,
                representation=viz.representation,
                show_nodes=viz.show_nodes,
                title=stage.name,
            )
            ax.figure.savefig(fig_dir / f"{stage.name}.png", dpi=viz.dpi, bbox_inches="tight")
            plt.close(ax.figure)
    if viz.save_sequence:
        n = 1 + len(stages)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        if n == 1:
            axes = [axes]
        plot_mask(mask, spacing=config.input.spacing, ax=axes[0], title="Binary mask")
        for ax, stage in zip(axes[1:], stages):
            plot_result(stage.result, representation=viz.representation, ax=ax, show_nodes=viz.show_nodes, title=stage.name)
        fig.tight_layout()
        fig.savefig(fig_dir / "augmentation_sequence.png", dpi=viz.dpi, bbox_inches="tight")
        plt.close(fig)


def execute_run(config: PaperRunConfig) -> PaperRunResult:
    """Execute one already-validated typed paper configuration."""
    run_dir = _prepare_run_dir(config)
    _archive_run_provenance(config, run_dir)
    mask = load_configured_mask(config)
    result = mask_to_graph(mask, spacing=config.input.spacing, config=config.extract)
    stages = _apply_augmentation(config, result)

    if config.output.save_stage_graph_json:
        stage_dir = run_dir / "stages"
        stage_dir.mkdir(parents=True, exist_ok=True)
        for stage in stages:
            (stage_dir / f"{stage.name}.json").write_text(result_to_json(stage.result), encoding="utf-8")
    if config.output.save_final_graph_json:
        (run_dir / "final_graph.json").write_text(result_to_json(stages[-1].result), encoding="utf-8")

    instance = None
    if config.output.save_min_ipd_json:
        instance = to_min_ipd(stages[-1].result.embedded_graph, config=config.extract.export, name=config.output.min_ipd_name)
        validate_min_ipd_export(instance)
        (run_dir / "min_ipd.json").write_text(json.dumps(instance, indent=2, sort_keys=True), encoding="utf-8")

    _save_figures(config, mask, stages, run_dir)
    resolved_yaml = dump_resolved_yaml(config)
    summary = {
        "name": config.name,
        "input_hash": stages[0].result.topology_graph.meta.input_hash,
        "extract_config_hash": stages[0].result.topology_graph.meta.config_hash,
        "resolved_config_sha256": hashlib.sha256(resolved_yaml.encode("utf-8")).hexdigest(),
        "source_graph_hash": stages[0].result.topology_graph.meta.graph_hash,
        "final_graph_hash": stages[-1].result.topology_graph.meta.graph_hash,
        "augmentation_seed": config.augmentation.seed,
        "stages": [
            {
                "name": s.name,
                "nodes": len(s.result.topology_graph.nodes),
                "edges": len(s.result.topology_graph.edges),
                "embedded_vertices": len(s.result.embedded_graph.vertices),
                "embedded_segments": len(s.result.embedded_graph.segments),
                "beta0": s.result.diagnostics.logical_beta0,
                "beta1": s.result.diagnostics.logical_beta1,
                "graph_hash": s.result.topology_graph.meta.graph_hash,
            }
            for s in stages
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return PaperRunResult(config=config, mask=mask, stages=stages, run_dir=run_dir, min_ipd_instance=instance)


def run_experiment(config_path: str | Path) -> PaperRunResult:
    """Paper-facing API: one YAML path, no behavioral keyword flags."""
    return execute_run(load_run_config(config_path))


__all__ = ["PaperRunResult", "RunStage", "execute_run", "load_configured_mask", "run_experiment"]
