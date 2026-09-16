"""Strict YAML configuration for paper-facing mask2graph runs.

The public experiment surface intentionally takes one YAML file.  Low-level
geometry functions remain ordinary Python APIs, but result-affecting experiment
policy belongs here and is validated before any extraction work starts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import ExtractConfig


class ConfigError(ValueError):
    """Raised when a paper YAML violates the strict configuration contract."""


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ConfigError(f"{where} must be a mapping")
    return value


def _keys(raw: dict[str, Any], required: set[str], where: str) -> None:
    unknown = set(raw) - required
    missing = required - set(raw)
    if unknown:
        raise ConfigError(f"{where} has unknown keys: {sorted(unknown)}")
    if missing:
        raise ConfigError(f"{where} is missing required keys: {sorted(missing)}")


def _bool(value: Any, where: str) -> bool:
    if type(value) is not bool:
        raise ConfigError(f"{where} must be bool")
    return value


def _int(value: Any, where: str) -> int:
    if type(value) is not int:
        raise ConfigError(f"{where} must be int")
    return value


def _float(value: Any, where: str) -> float:
    if type(value) is not float:
        raise ConfigError(f"{where} must be float (write e.g. 1.0, not 1 or '1.0')")
    return value


def _str(value: Any, where: str) -> str:
    if type(value) is not str or not value:
        raise ConfigError(f"{where} must be a non-empty string")
    return value


def _nullable_str(value: Any, where: str) -> str | None:
    if value is None:
        return None
    return _str(value, where)


def _scalar(value: Any, where: str) -> bool | int | float:
    if type(value) not in (bool, int, float):
        raise ConfigError(f"{where} must be a YAML bool/int/float scalar")
    return value


def _float_list(value: Any, where: str, *, length: int | None = None) -> tuple[float, ...]:
    if type(value) is not list:
        raise ConfigError(f"{where} must be a list")
    if length is not None and len(value) != length:
        raise ConfigError(f"{where} must have length {length}")
    return tuple(_float(v, f"{where}[{i}]") for i, v in enumerate(value))


def _int_list(value: Any, where: str, *, length: int | None = None) -> tuple[int, ...]:
    if type(value) is not list:
        raise ConfigError(f"{where} must be a list")
    if length is not None and len(value) != length:
        raise ConfigError(f"{where} must have length {length}")
    return tuple(_int(v, f"{where}[{i}]") for i, v in enumerate(value))


def _str_list(value: Any, where: str) -> tuple[str, ...]:
    if type(value) is not list:
        raise ConfigError(f"{where} must be a list")
    return tuple(_str(v, f"{where}[{i}]") for i, v in enumerate(value))


@dataclass(frozen=True)
class InputRunConfig:
    path: Path
    expected_ndim: int
    spacing: tuple[float, ...]
    foreground_value: bool | int | float
    background_value: bool | int | float
    require_only_configured_values: bool
    squeeze_singleton_axes: bool
    channel: int | None
    npz_array_key: str | None


@dataclass(frozen=True)
class VisualizationRunConfig:
    enabled: bool
    representation: str
    show_nodes: bool
    dpi: int
    save_original_mask: bool
    save_graph_overlay: bool
    save_each_stage: bool
    save_sequence: bool


@dataclass(frozen=True)
class OutputRunConfig:
    run_dir: Path
    overwrite: bool
    save_stage_graph_json: bool
    save_final_graph_json: bool
    save_min_ipd_json: bool
    min_ipd_name: str


@dataclass(frozen=True)
class AugmentationRunConfig:
    enabled: bool
    seed: int
    operations: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class PaperRunConfig:
    schema_version: int
    name: str
    input: InputRunConfig
    extract: ExtractConfig
    augmentation: AugmentationRunConfig
    visualization: VisualizationRunConfig
    output: OutputRunConfig
    source_yaml: Path

    def resolved_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "input": {
                **asdict(self.input),
                "path": str(self.input.path),
                "spacing": list(self.input.spacing),
            },
            "extract": self.extract.to_dict(),
            "augmentation": {
                "enabled": self.augmentation.enabled,
                "seed": self.augmentation.seed,
                "operations": [dict(op) for op in self.augmentation.operations],
            },
            "visualization": asdict(self.visualization),
            "output": {**asdict(self.output), "run_dir": str(self.output.run_dir)},
        }


def _parse_input(raw: Any, base: Path) -> InputRunConfig:
    d = _mapping(raw, "input")
    required = {
        "path", "expected_ndim", "spacing", "foreground_value", "background_value",
        "require_only_configured_values", "squeeze_singleton_axes", "channel", "npz_array_key",
    }
    _keys(d, required, "input")
    ndim = _int(d["expected_ndim"], "input.expected_ndim")
    if ndim not in (2, 3):
        raise ConfigError("input.expected_ndim must be 2 or 3")
    spacing = _float_list(d["spacing"], "input.spacing", length=ndim)
    if any(v <= 0.0 for v in spacing):
        raise ConfigError("input.spacing values must be > 0")
    channel = d["channel"]
    if channel is not None:
        channel = _int(channel, "input.channel")
        if channel < 0:
            raise ConfigError("input.channel must be >= 0 or null")
    p = Path(_str(d["path"], "input.path"))
    if not p.is_absolute():
        p = (base / p).resolve()
    return InputRunConfig(
        path=p,
        expected_ndim=ndim,
        spacing=spacing,
        foreground_value=_scalar(d["foreground_value"], "input.foreground_value"),
        background_value=_scalar(d["background_value"], "input.background_value"),
        require_only_configured_values=_bool(d["require_only_configured_values"], "input.require_only_configured_values"),
        squeeze_singleton_axes=_bool(d["squeeze_singleton_axes"], "input.squeeze_singleton_axes"),
        channel=channel,
        npz_array_key=_nullable_str(d["npz_array_key"], "input.npz_array_key"),
    )


def _validate_center(value: Any, where: str) -> str | tuple[float, ...]:
    if type(value) is str:
        if value not in {"origin", "centroid", "bbox_center"}:
            raise ConfigError(f"{where} must be origin, centroid, bbox_center, or an explicit float list")
        return value
    return _float_list(value, where)


def _parse_operation(raw: Any, i: int, ndim: int) -> dict[str, Any]:
    d = _mapping(raw, f"augmentation.operations[{i}]")
    if "type" not in d:
        raise ConfigError(f"augmentation.operations[{i}] is missing required key 'type'")
    typ = _str(d["type"], f"augmentation.operations[{i}].type")
    w = f"augmentation.operations[{i}]"

    if typ == "rotate":
        required = {"type", "angle_degrees", "center", "axis_3d"}
        _keys(d, required, w)
        center = _validate_center(d["center"], f"{w}.center")
        axis = d["axis_3d"]
        if ndim == 2:
            if axis is not None:
                raise ConfigError(f"{w}.axis_3d must be null for 2D")
            axis_out = None
        else:
            if axis is None:
                raise ConfigError(f"{w}.axis_3d is required for 3D rotation")
            axis_out = _float_list(axis, f"{w}.axis_3d", length=3)
        return {"type": typ, "angle_degrees": _float(d["angle_degrees"], f"{w}.angle_degrees"), "center": center, "axis_3d": axis_out}

    if typ == "flip":
        required = {"type", "axes", "center"}
        _keys(d, required, w)
        axes = _str_list(d["axes"], f"{w}.axes")
        allowed = {"x", "y"} if ndim == 2 else {"x", "y", "z"}
        if not axes or any(a not in allowed for a in axes):
            raise ConfigError(f"{w}.axes must contain only {sorted(allowed)}")
        return {"type": typ, "axes": list(axes), "center": _validate_center(d["center"], f"{w}.center")}

    if typ == "translate":
        required = {"type", "offset"}
        _keys(d, required, w)
        return {"type": typ, "offset": list(_float_list(d["offset"], f"{w}.offset", length=ndim))}

    if typ == "axis_permutation":
        required = {"type", "order"}
        _keys(d, required, w)
        order = _int_list(d["order"], f"{w}.order", length=ndim)
        if set(order) != set(range(ndim)):
            raise ConfigError(f"{w}.order must be a permutation of 0..{ndim - 1}")
        return {"type": typ, "order": list(order)}

    if typ == "crop_box":
        required = {"type", "bounds", "keep_size"}
        _keys(d, required, w)
        bounds = _float_list(d["bounds"], f"{w}.bounds", length=2 * ndim)
        for k in range(ndim):
            if not bounds[2 * k] < bounds[2 * k + 1]:
                raise ConfigError(f"{w}.bounds lower value must be < upper value for axis {k}")
        return {"type": typ, "bounds": list(bounds), "keep_size": _bool(d["keep_size"], f"{w}.keep_size")}

    if typ == "central_crop":
        required = {"type", "fraction", "keep_size"}
        _keys(d, required, w)
        fraction = _float(d["fraction"], f"{w}.fraction")
        if not 0.0 < fraction <= 1.0:
            raise ConfigError(f"{w}.fraction must be in (0,1]")
        return {"type": typ, "fraction": fraction, "keep_size": _bool(d["keep_size"], f"{w}.keep_size")}

    if typ == "random_flip":
        required = {"type", "axes", "probability", "center"}
        _keys(d, required, w)
        axes = _str_list(d["axes"], f"{w}.axes")
        allowed = {"x", "y"} if ndim == 2 else {"x", "y", "z"}
        if not axes or any(a not in allowed for a in axes):
            raise ConfigError(f"{w}.axes must contain only {sorted(allowed)}")
        probability = _float(d["probability"], f"{w}.probability")
        if not 0.0 <= probability <= 1.0:
            raise ConfigError(f"{w}.probability must be in [0,1]")
        return {"type": typ, "axes": list(axes), "probability": probability, "center": _validate_center(d["center"], f"{w}.center")}

    if typ == "random_rotation":
        required = {"type", "angles_degrees", "axes_3d", "center"}
        _keys(d, required, w)
        angles = _float_list(d["angles_degrees"], f"{w}.angles_degrees")
        if not angles:
            raise ConfigError(f"{w}.angles_degrees must be non-empty")
        axes_raw = d["axes_3d"]
        axes: list[list[float]] = []
        if ndim == 2:
            if axes_raw is not None:
                raise ConfigError(f"{w}.axes_3d must be null for 2D")
        else:
            if type(axes_raw) is not list or not axes_raw:
                raise ConfigError(f"{w}.axes_3d must be a non-empty list of 3-float axes for 3D")
            for j, axis in enumerate(axes_raw):
                axes.append(list(_float_list(axis, f"{w}.axes_3d[{j}]", length=3)))
        return {"type": typ, "angles_degrees": list(angles), "axes_3d": None if ndim == 2 else axes, "center": _validate_center(d["center"], f"{w}.center")}

    if typ == "random_crop":
        required = {"type", "size", "keep_size"}
        _keys(d, required, w)
        size = _float_list(d["size"], f"{w}.size", length=ndim)
        if any(v <= 0.0 for v in size):
            raise ConfigError(f"{w}.size values must be > 0")
        return {"type": typ, "size": list(size), "keep_size": _bool(d["keep_size"], f"{w}.keep_size")}

    raise ConfigError(f"{w}.type has unsupported operation {typ!r}")


def _parse_augmentation(raw: Any, ndim: int) -> AugmentationRunConfig:
    d = _mapping(raw, "augmentation")
    _keys(d, {"enabled", "seed", "operations"}, "augmentation")
    enabled = _bool(d["enabled"], "augmentation.enabled")
    seed = _int(d["seed"], "augmentation.seed")
    if seed < 0:
        raise ConfigError("augmentation.seed must be >= 0")
    ops_raw = d["operations"]
    if type(ops_raw) is not list:
        raise ConfigError("augmentation.operations must be a list")
    operations = tuple(_parse_operation(op, i, ndim) for i, op in enumerate(ops_raw))
    if not enabled and operations:
        raise ConfigError("augmentation.operations must be empty when augmentation.enabled is false")
    return AugmentationRunConfig(enabled=enabled, seed=seed, operations=operations)


def _parse_visualization(raw: Any) -> VisualizationRunConfig:
    d = _mapping(raw, "visualization")
    required = {"enabled", "representation", "show_nodes", "dpi", "save_original_mask", "save_graph_overlay", "save_each_stage", "save_sequence"}
    _keys(d, required, "visualization")
    rep = _str(d["representation"], "visualization.representation")
    if rep not in {"topology", "embedded"}:
        raise ConfigError("visualization.representation must be topology or embedded")
    dpi = _int(d["dpi"], "visualization.dpi")
    if dpi <= 0:
        raise ConfigError("visualization.dpi must be > 0")
    return VisualizationRunConfig(
        enabled=_bool(d["enabled"], "visualization.enabled"),
        representation=rep,
        show_nodes=_bool(d["show_nodes"], "visualization.show_nodes"),
        dpi=dpi,
        save_original_mask=_bool(d["save_original_mask"], "visualization.save_original_mask"),
        save_graph_overlay=_bool(d["save_graph_overlay"], "visualization.save_graph_overlay"),
        save_each_stage=_bool(d["save_each_stage"], "visualization.save_each_stage"),
        save_sequence=_bool(d["save_sequence"], "visualization.save_sequence"),
    )


def _parse_output(raw: Any, base: Path) -> OutputRunConfig:
    d = _mapping(raw, "output")
    required = {"run_dir", "overwrite", "save_stage_graph_json", "save_final_graph_json", "save_min_ipd_json", "min_ipd_name"}
    _keys(d, required, "output")
    run_dir = Path(_str(d["run_dir"], "output.run_dir"))
    if not run_dir.is_absolute():
        run_dir = (base / run_dir).resolve()
    return OutputRunConfig(
        run_dir=run_dir,
        overwrite=_bool(d["overwrite"], "output.overwrite"),
        save_stage_graph_json=_bool(d["save_stage_graph_json"], "output.save_stage_graph_json"),
        save_final_graph_json=_bool(d["save_final_graph_json"], "output.save_final_graph_json"),
        save_min_ipd_json=_bool(d["save_min_ipd_json"], "output.save_min_ipd_json"),
        min_ipd_name=_str(d["min_ipd_name"], "output.min_ipd_name"),
    )


def load_run_config(path: str | Path) -> PaperRunConfig:
    """Load and strictly validate one self-contained experiment YAML."""
    source = Path(path).resolve()
    if not source.is_file():
        raise ConfigError(f"configuration file does not exist: {source}")
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML: {exc}") from exc
    d = _mapping(raw, "root")
    required = {"schema_version", "name", "input", "extract", "augmentation", "visualization", "output"}
    _keys(d, required, "root")
    version = _int(d["schema_version"], "schema_version")
    if version != 1:
        raise ConfigError("schema_version must be exactly 1")
    input_cfg = _parse_input(d["input"], source.parent)
    try:
        extract_cfg = ExtractConfig.from_dict_strict(_mapping(d["extract"], "extract"), require_all=True)
        extract_cfg.validate(input_cfg.expected_ndim)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"extract: {exc}") from exc
    augmentation_cfg = _parse_augmentation(d["augmentation"], input_cfg.expected_ndim)
    viz_cfg = _parse_visualization(d["visualization"])
    if input_cfg.expected_ndim == 3 and (viz_cfg.save_original_mask or viz_cfg.save_graph_overlay or viz_cfg.save_sequence):
        raise ConfigError(
            "3D visualization must set save_original_mask=false, save_graph_overlay=false, "
            "and save_sequence=false; save_each_stage supports 3D graph views"
        )
    out_cfg = _parse_output(d["output"], source.parent)
    return PaperRunConfig(
        schema_version=version,
        name=_str(d["name"], "name"),
        input=input_cfg,
        extract=extract_cfg,
        augmentation=augmentation_cfg,
        visualization=viz_cfg,
        output=out_cfg,
        source_yaml=source,
    )


def dump_resolved_yaml(config: PaperRunConfig) -> str:
    return yaml.safe_dump(config.resolved_dict(), sort_keys=False, allow_unicode=True)


__all__ = [
    "AugmentationRunConfig", "ConfigError", "InputRunConfig", "OutputRunConfig",
    "PaperRunConfig", "VisualizationRunConfig", "dump_resolved_yaml", "load_run_config",
]
