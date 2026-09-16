"""Graph-to-graph augmentation pipeline.

Randomness exists only in the sampling wrappers here.  Every low-level transform
and crop operator is deterministic for explicit parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

import numpy as np

from .crop import crop_graph_box
from .transforms import GraphTransform, apply_transform, compose_transforms, flip_graph, rotate_graph, translate_graph
from .types import Mask2Graph, MaskGraphResult
from .utils.hash import graph_content_hash


def augment_result(result: MaskGraphResult, transform: GraphTransform, *, validate: bool = True) -> MaskGraphResult:
    """Apply one explicit rigid transform to both topology and embedded graphs."""
    return apply_transform(result, transform, validate=validate)


def augment_graph(
    graph: Mask2Graph | MaskGraphResult,
    *,
    transforms: Iterable[GraphTransform] = (),
    crop_bounds: Sequence[float] | None = None,
    keep_size: bool = False,
    validate: bool = True,
):
    """Apply explicit transforms, then optional geometric crop."""
    out = graph
    items = list(transforms)
    if items:
        out = apply_transform(out, compose_transforms(items), validate=validate)
    if crop_bounds is not None:
        out = crop_graph_box(out, bounds=crop_bounds, keep_size=keep_size, validate=validate)
    return out


def _graph_bbox(result: MaskGraphResult) -> tuple[np.ndarray, np.ndarray]:
    g = result.topology_graph
    pts: list[np.ndarray] = [np.asarray(n.xyz, dtype=float) for n in g.nodes]
    pts.extend(np.asarray(p, dtype=float) for e in g.edges for p in e.path_xyz)
    if not pts:
        z = np.zeros(g.meta.ndim, dtype=float)
        return z, z
    arr = np.asarray(pts, dtype=float)[:, : g.meta.ndim]
    return arr.min(axis=0), arr.max(axis=0)


def _annotate_last(result: MaskGraphResult, *, seed: int | None, random_op: str, sampled: dict) -> MaskGraphResult:
    history = result.topology_graph.meta.augmentation_history
    if history:
        history[-1] = {**history[-1], "random_op": random_op, "sampled": sampled}
        if seed is not None:
            history[-1]["augmentation_seed"] = int(seed)
    if seed is not None:
        result.topology_graph.meta.augmentation_seed = int(seed)
    result.topology_graph.meta.graph_hash = graph_content_hash(result.topology_graph)
    result.embedded_graph.metadata["graph_hash"] = result.topology_graph.meta.graph_hash
    result.embedded_graph.metadata["augmentation_history"] = list(history)
    return result


class RandomGraphAugmentation(Protocol):
    def apply(self, result: MaskGraphResult, rng: np.random.Generator, *, seed: int | None = None) -> MaskGraphResult: ...


@dataclass(frozen=True)
class RandomFlip:
    axes: tuple[str, ...] = ("x", "y")
    probability: float = 0.5
    center: str = "bbox_center"

    def apply(self, result: MaskGraphResult, rng: np.random.Generator, *, seed: int | None = None) -> MaskGraphResult:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("flip probability must be in [0,1]")
        allowed = ("x", "y") if result.topology_graph.meta.ndim == 2 else ("x", "y", "z")
        axes = tuple(a for a in self.axes if a in allowed and float(rng.random()) < self.probability)
        if not axes:
            return result
        out = flip_graph(result, axes, center=self.center)
        return _annotate_last(out, seed=seed, random_op="RandomFlip", sampled={"axes": list(axes)})


@dataclass(frozen=True)
class RandomRotation:
    angles: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)
    axes_3d: tuple[tuple[float, float, float], ...] = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    center: str = "bbox_center"
    degrees: bool = True

    def apply(self, result: MaskGraphResult, rng: np.random.Generator, *, seed: int | None = None) -> MaskGraphResult:
        if not self.angles:
            return result
        angle = float(self.angles[int(rng.integers(0, len(self.angles)))])
        if result.topology_graph.meta.ndim == 2:
            out = rotate_graph(result, angle, center=self.center, degrees=self.degrees)
            sampled = {"angle": angle, "degrees": self.degrees}
        else:
            if not self.axes_3d:
                raise ValueError("3D random rotation requires at least one axis")
            axis = self.axes_3d[int(rng.integers(0, len(self.axes_3d)))]
            out = rotate_graph(result, angle, center=self.center, degrees=self.degrees, axis=axis)
            sampled = {"angle": angle, "degrees": self.degrees, "axis": list(axis)}
        return _annotate_last(out, seed=seed, random_op="RandomRotation", sampled=sampled)


@dataclass(frozen=True)
class RandomCrop:
    size: tuple[float, ...]
    keep_size: bool = True

    def apply(self, result: MaskGraphResult, rng: np.random.Generator, *, seed: int | None = None) -> MaskGraphResult:
        ndim = result.topology_graph.meta.ndim
        if len(self.size) != ndim or any(float(v) <= 0 for v in self.size):
            raise ValueError("crop size must contain one positive physical length per dimension")
        lo0, hi0 = _graph_bbox(result)
        size = np.asarray(self.size, dtype=float)
        span = hi0 - lo0
        room = span - size
        # If a requested crop is wider than the graph in a dimension, center
        # that dimension and treat the excess as padding space.  Otherwise
        # sample a contained crop uniformly.
        lo = np.empty(ndim, dtype=float)
        rand = rng.random(ndim)
        for k in range(ndim):
            if room[k] >= 0.0:
                lo[k] = lo0[k] + rand[k] * room[k]
            else:
                lo[k] = (lo0[k] + hi0[k] - size[k]) / 2.0
        hi = lo + size
        bounds = [float(x) for pair in zip(lo, hi) for x in pair]
        out = crop_graph_box(result, bounds=bounds, keep_size=self.keep_size)
        return _annotate_last(out, seed=seed, random_op="RandomCrop", sampled={"bounds": bounds, "keep_size": self.keep_size})


@dataclass
class GraphAugmentationPipeline:
    operations: list[RandomGraphAugmentation]

    def __call__(
        self,
        result: MaskGraphResult,
        *,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> MaskGraphResult:
        if rng is not None and seed is not None:
            raise ValueError("provide rng or seed, not both")
        if rng is None:
            rng = np.random.default_rng(0 if seed is None else int(seed))
        out = result
        for op in self.operations:
            out = op.apply(out, rng, seed=seed)
        if seed is not None:
            out.topology_graph.meta.augmentation_seed = int(seed)
            out.embedded_graph.metadata["augmentation_seed"] = int(seed)
        return out


__all__ = [
    "GraphAugmentationPipeline",
    "RandomCrop",
    "RandomFlip",
    "RandomRotation",
    "augment_graph",
    "augment_result",
]
