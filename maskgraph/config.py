"""Configuration model for maskgraph extraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class CleanupConfig:
    fill_holes_max_size: int | None = None
    remove_objects_max_size: int | None = None
    connectivity: int | None = None


@dataclass
class SkeletonConfig:
    method_2d: str = "zhang"
    method_3d: str = "lee"


@dataclass
class NormalizeConfig:
    min_component_length: float = 0.0
    prune_spurs_below: float = 0.0
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
        if self.cleanup.connectivity is not None and self.cleanup.connectivity <= 0:
            raise ValueError("cleanup.connectivity must be positive when provided")
        if self.normalize.min_component_length < 0:
            raise ValueError("normalize.min_component_length must be >= 0")
        if self.normalize.prune_spurs_below < 0:
            raise ValueError("normalize.prune_spurs_below must be >= 0")
        if self.normalize.prune_iterations < 0:
            raise ValueError("normalize.prune_iterations must be >= 0")
        if self.simplify.epsilon < 0:
            raise ValueError("simplify.epsilon must be >= 0")
        if self.determinism.float_decimals < 0:
            raise ValueError("determinism.float_decimals must be >= 0")


def default_spacing(ndim: int) -> tuple[float, ...]:
    return tuple(1.0 for _ in range(ndim))
