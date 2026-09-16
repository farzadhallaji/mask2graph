"""Configuration model for mask2graph extraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, get_type_hints


@dataclass
class CleanupConfig:
    min_object_size: float = 0.0
    max_hole_size: float = 0.0
    max_hole_radius: float = 0.0
    enabled: bool = True


@dataclass
class SkeletonConfig:
    method_2d: str = "zhang"
    method_3d: str = "lee"


@dataclass
class JunctionConfig:
    resolution: str = "mst"
    dilation_iters: int = 0
    supported_anchor: bool = True


@dataclass
class NormalizeConfig:
    # Backward-compatible alias. JunctionConfig takes precedence in the new API
    # unless this value is explicitly nonzero.
    junction_dilation_iters: int = 0
    min_component_length: float = 0.0
    prune_spurs_below: float = 0.0
    min_cycle_length: float = 0.0
    max_cycle_area: float = 0.0
    cycle_length_to_radius_ratio: float = 0.0
    contract_short_edges_below: float = 0.0
    normalization_max_iter: int = 10
    prune_iterations: int = 100
    contract_degree2: bool = True


@dataclass
class GeometryConfig:
    tangent_window: int = 2
    compute_curvature: bool = True


@dataclass
class SimplifyConfig:
    enabled: bool = True
    epsilon: float = 1.0
    method: str = "optimal"  # optimal | rdp | none
    protect_angle_degrees: float = 45.0
    topology_guard: bool = True
    fallback_to_original: bool = True


@dataclass
class ValidationConfig:
    enabled: bool = True
    validate_coverage: bool = True
    validate_topology: bool = True
    validate_embedding: bool = True
    tolerance: float = 1e-9


@dataclass
class ExportConfig:
    coordinate_mode: str = "lattice_exact"  # lattice_exact | scaled_exact | rationalized
    max_denominator: int = 1_000_000
    domain_margin: float = 1.0


@dataclass
class DeterminismConfig:
    float_decimals: int = 6
    sort_nodes: bool = True
    sort_edges: bool = True


@dataclass
class ExtractConfig:
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)
    skeleton: SkeletonConfig = field(default_factory=SkeletonConfig)
    junction: JunctionConfig = field(default_factory=JunctionConfig)
    normalize: NormalizeConfig = field(default_factory=NormalizeConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    simplify: SimplifyConfig = field(default_factory=SimplifyConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    determinism: DeterminismConfig = field(default_factory=DeterminismConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def _strict_value(value: Any, annotation: Any, where: str) -> Any:
        # The config dataclasses intentionally use only primitive scalar types.
        if annotation is bool:
            if type(value) is not bool:
                raise TypeError(f"{where} must be bool")
            return value
        if annotation is int:
            if type(value) is not int:
                raise TypeError(f"{where} must be int")
            return value
        if annotation is float:
            if type(value) is not float:
                raise TypeError(f"{where} must be float")
            return value
        if annotation is str:
            if type(value) is not str:
                raise TypeError(f"{where} must be str")
            return value
        raise TypeError(f"unsupported config annotation at {where}: {annotation!r}")

    @classmethod
    def from_dict_strict(cls, payload: dict[str, Any], *, require_all: bool) -> "ExtractConfig":
        """Rebuild nested config with unknown/missing/type errors.

        `require_all=True` is the paper-YAML contract: every behavior-affecting
        field must be explicit.  `require_all=False` is useful only for loading
        versioned internal metadata that already came from a complete config.
        """
        if type(payload) is not dict:
            raise TypeError("ExtractConfig payload must be a mapping")
        section_types = {
            "cleanup": CleanupConfig,
            "skeleton": SkeletonConfig,
            "junction": JunctionConfig,
            "normalize": NormalizeConfig,
            "geometry": GeometryConfig,
            "simplify": SimplifyConfig,
            "validation": ValidationConfig,
            "export": ExportConfig,
            "determinism": DeterminismConfig,
        }
        unknown_sections = set(payload) - set(section_types)
        if unknown_sections:
            raise ValueError(f"unknown ExtractConfig sections: {sorted(unknown_sections)}")
        if require_all:
            missing_sections = set(section_types) - set(payload)
            if missing_sections:
                raise ValueError(f"missing ExtractConfig sections: {sorted(missing_sections)}")

        def build(tp, name: str):
            if name not in payload:
                return tp()
            raw = payload[name]
            if type(raw) is not dict:
                raise TypeError(f"{name} must be a mapping")
            hints = get_type_hints(tp)
            allowed = {f.name for f in fields(tp)}
            unknown = set(raw) - allowed
            if unknown:
                raise ValueError(f"{name} has unknown keys: {sorted(unknown)}")
            if require_all:
                missing = allowed - set(raw)
                if missing:
                    raise ValueError(f"{name} is missing required keys: {sorted(missing)}")
            kwargs = {}
            for f in fields(tp):
                if f.name in raw:
                    kwargs[f.name] = cls._strict_value(raw[f.name], hints[f.name], f"{name}.{f.name}")
            return tp(**kwargs)

        return cls(**{name: build(tp, name) for name, tp in section_types.items()})

    @classmethod
    def from_dict(cls, payload: dict | None) -> "ExtractConfig":
        """Strictly rebuild nested configuration from serialized metadata."""
        if payload is None:
            return cls()
        return cls.from_dict_strict(payload, require_all=False)

    def validate(self, ndim: int) -> None:
        if ndim not in (2, 3):
            raise ValueError("ndim must be 2 or 3")
        for name, value in (
            ("cleanup.min_object_size", self.cleanup.min_object_size),
            ("cleanup.max_hole_size", self.cleanup.max_hole_size),
            ("cleanup.max_hole_radius", self.cleanup.max_hole_radius),
            ("normalize.min_component_length", self.normalize.min_component_length),
            ("normalize.prune_spurs_below", self.normalize.prune_spurs_below),
            ("normalize.min_cycle_length", self.normalize.min_cycle_length),
            ("normalize.max_cycle_area", self.normalize.max_cycle_area),
            ("normalize.cycle_length_to_radius_ratio", self.normalize.cycle_length_to_radius_ratio),
            ("normalize.contract_short_edges_below", self.normalize.contract_short_edges_below),
            ("simplify.epsilon", self.simplify.epsilon),
            ("validation.tolerance", self.validation.tolerance),
            ("export.domain_margin", self.export.domain_margin),
        ):
            if value < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.normalize.junction_dilation_iters < 0 or self.junction.dilation_iters < 0:
            raise ValueError("junction dilation iterations must be >= 0")
        if self.normalize.normalization_max_iter < 0 or self.normalize.prune_iterations < 0:
            raise ValueError("normalization iteration limits must be >= 0")
        if self.geometry.tangent_window < 1:
            raise ValueError("geometry.tangent_window must be >= 1")
        if self.simplify.method not in {"optimal", "rdp", "none"}:
            raise ValueError("simplify.method must be 'optimal', 'rdp', or 'none'")
        if self.junction.resolution not in {"mst", "cluster"}:
            raise ValueError("junction.resolution must be 'mst' or 'cluster'")
        if not 0.0 <= self.simplify.protect_angle_degrees <= 180.0:
            raise ValueError("simplify.protect_angle_degrees must be in [0,180]")
        if self.export.coordinate_mode not in {"lattice_exact", "scaled_exact", "rationalized"}:
            raise ValueError("invalid export.coordinate_mode")
        if self.export.max_denominator < 1:
            raise ValueError("export.max_denominator must be >= 1")
        if self.determinism.float_decimals < 0:
            raise ValueError("determinism.float_decimals must be >= 0")


def default_spacing(ndim: int) -> tuple[float, ...]:
    return tuple(1.0 for _ in range(ndim))
