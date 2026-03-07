"""Configuration model for maskgraph extraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


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
class NormalizeConfig:
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
class SimplifyConfig:
    enabled: bool = False
    epsilon: float = 0.0


@dataclass
class DeterminismConfig:
    float_decimals: int = 6
    sort_nodes: bool = True
    sort_edges: bool = True


@dataclass
class ExtractConfig:
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)
    skeleton: SkeletonConfig = field(default_factory=SkeletonConfig)
    normalize: NormalizeConfig = field(default_factory=NormalizeConfig)
    simplify: SimplifyConfig = field(default_factory=SimplifyConfig)
    determinism: DeterminismConfig = field(default_factory=DeterminismConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self, ndim: int) -> None:
        if ndim not in (2, 3):
            raise ValueError("ndim must be 2 or 3")
        if self.cleanup.min_object_size < 0:
            raise ValueError("cleanup.min_object_size must be >= 0")
        if self.cleanup.max_hole_size < 0:
            raise ValueError("cleanup.max_hole_size must be >= 0")
        if self.cleanup.max_hole_radius < 0:
            raise ValueError("cleanup.max_hole_radius must be >= 0")
        if self.normalize.junction_dilation_iters < 0:
            raise ValueError("normalize.junction_dilation_iters must be >= 0")
        if self.normalize.min_component_length < 0:
            raise ValueError("normalize.min_component_length must be >= 0")
        if self.normalize.prune_spurs_below < 0:
            raise ValueError("normalize.prune_spurs_below must be >= 0")
        if self.normalize.min_cycle_length < 0:
            raise ValueError("normalize.min_cycle_length must be >= 0")
        if self.normalize.max_cycle_area < 0:
            raise ValueError("normalize.max_cycle_area must be >= 0")
        if self.normalize.cycle_length_to_radius_ratio < 0:
            raise ValueError("normalize.cycle_length_to_radius_ratio must be >= 0")
        if self.normalize.contract_short_edges_below < 0:
            raise ValueError("normalize.contract_short_edges_below must be >= 0")
        if self.normalize.normalization_max_iter < 0:
            raise ValueError("normalize.normalization_max_iter must be >= 0")
        if self.normalize.prune_iterations < 0:
            raise ValueError("normalize.prune_iterations must be >= 0")
        if self.simplify.epsilon < 0:
            raise ValueError("simplify.epsilon must be >= 0")
        if self.determinism.float_decimals < 0:
            raise ValueError("determinism.float_decimals must be >= 0")


def default_spacing(ndim: int) -> tuple[float, ...]:
    return tuple(1.0 for _ in range(ndim))
