"""Export embedded mask graphs to exact-rational min_ipd instance dictionaries."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import numpy as np

from .config import ExportConfig
from .types import EmbeddedGraph, ValidationError


def _fmt_fraction(q: Fraction) -> int | str:
    return int(q.numerator) if q.denominator == 1 else f"{q.numerator}/{q.denominator}"


def _coord_fraction(value: float, *, mode: str, max_denominator: int) -> Fraction:
    if mode == "lattice_exact":
        nearest = int(round(float(value)))
        if abs(float(value) - nearest) > 1e-9:
            raise ValueError("lattice_exact requires integer physical coordinates; use scaled_exact")
        return Fraction(nearest)
    if mode == "scaled_exact":
        return Fraction(str(float(value)))
    if mode == "rationalized":
        return Fraction(float(value)).limit_denominator(int(max_denominator))
    raise ValueError(f"unknown coordinate mode {mode!r}")


def _rationalization_error(points_xyz: list[np.ndarray], *, ndim: int, mode: str, max_denominator: int) -> float:
    if mode != "rationalized":
        return 0.0
    err = 0.0
    for p in points_xyz:
        for value in p[:ndim]:
            q = _coord_fraction(float(value), mode=mode, max_denominator=max_denominator)
            err = max(err, abs(float(value) - float(q)))
    return float(err)


def _referenced_geometry(embedded: EmbeddedGraph) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    if not embedded.segments:
        return [], []
    used = sorted({s.u for s in embedded.segments} | {s.v for s in embedded.segments})
    old_to_new = {old: i for i, old in enumerate(used)}
    by_id = {v.id: np.asarray(v.xyz, dtype=float) for v in embedded.vertices}
    points = [by_id[i] for i in used]
    segments = [(old_to_new[s.u], old_to_new[s.v]) for s in embedded.segments]
    return points, segments


def _margin_fraction(config: ExportConfig) -> Fraction:
    if config.coordinate_mode == "rationalized":
        return Fraction(float(config.domain_margin)).limit_denominator(config.max_denominator)
    return Fraction(str(float(config.domain_margin)))


def to_min_ipd(
    embedded: EmbeddedGraph,
    *,
    domain: Any | None = None,
    config: ExportConfig | None = None,
    name: str = "mask2graph",
) -> dict[str, Any]:
    """Convert an embedded graph to a min_ipd 2D/3D instance dictionary.

    ``domain`` is either a 2D convex boundary as coordinate pairs or a 3D
    ``{"halfspaces": ...}`` object.  When omitted, a rational axis-aligned box
    with ``domain_margin`` is generated around all prescribed segments.
    """
    cfg = config or ExportConfig()
    points_xyz, segments = _referenced_geometry(embedded)
    if not points_xyz:
        raise ValueError("min_ipd export requires at least one prescribed segment")

    qpts: list[list[int | str]] = []
    for p in points_xyz:
        coords = p[: embedded.ndim]
        qpts.append([
            _fmt_fraction(_coord_fraction(float(v), mode=cfg.coordinate_mode, max_denominator=cfg.max_denominator))
            for v in coords
        ])

    export_error = _rationalization_error(
        points_xyz, ndim=embedded.ndim, mode=cfg.coordinate_mode, max_denominator=cfg.max_denominator
    )
    source_meta = dict(getattr(embedded, "metadata", {}))
    output_meta = {
        "name": name,
        "source": "mask2graph",
        "transform_exact": bool(source_meta.get("transform_exact", True)),
        "grid_aligned": bool(source_meta.get("grid_aligned", False)),
        "source_graph_hash": source_meta.get("source_graph_hash", ""),
        "graph_hash": source_meta.get("graph_hash", ""),
        "augmentation_history": source_meta.get("augmentation_history", []),
        "augmentation_seed": source_meta.get("augmentation_seed"),
        "coordinate_mode": cfg.coordinate_mode,
        "rationalization_error": float(export_error),
    }

    if embedded.ndim == 2:
        if domain is None:
            qx = [_coord_fraction(float(p[0]), mode=cfg.coordinate_mode, max_denominator=cfg.max_denominator) for p in points_xyz]
            qy = [_coord_fraction(float(p[1]), mode=cfg.coordinate_mode, max_denominator=cfg.max_denominator) for p in points_xyz]
            m = _margin_fraction(cfg)
            xmin, xmax = min(qx) - m, max(qx) + m
            ymin, ymax = min(qy) - m, max(qy) + m
            boundary = [[_fmt_fraction(xmin), _fmt_fraction(ymin)], [_fmt_fraction(xmax), _fmt_fraction(ymin)],
                        [_fmt_fraction(xmax), _fmt_fraction(ymax)], [_fmt_fraction(xmin), _fmt_fraction(ymax)]]
        else:
            boundary = []
            for pair in domain:
                if len(pair) != 2:
                    raise ValueError("2D domain vertices must be [x,y]")
                boundary.append([
                    _fmt_fraction(_coord_fraction(float(pair[0]), mode=cfg.coordinate_mode, max_denominator=cfg.max_denominator)),
                    _fmt_fraction(_coord_fraction(float(pair[1]), mode=cfg.coordinate_mode, max_denominator=cfg.max_denominator)),
                ])
        offset = len(qpts)
        qpts.extend(boundary)
        return {
            "schema": "min_ipd_instance_v1",
            "points": qpts,
            "segments": [[int(a), int(b)] for a, b in segments],
            "outer_boundary": list(range(offset, offset + len(boundary))),
            "holes": [],
            "metadata": output_meta,
        }

    if embedded.ndim == 3:
        if domain is None:
            qs = [
                tuple(_coord_fraction(float(v), mode=cfg.coordinate_mode, max_denominator=cfg.max_denominator) for v in p[:3])
                for p in points_xyz
            ]
            m = _margin_fraction(cfg)
            mins = [min(p[i] for p in qs) - m for i in range(3)]
            maxs = [max(p[i] for p in qs) + m for i in range(3)]
            hs = [
                [Fraction(1), Fraction(0), Fraction(0), maxs[0]],
                [Fraction(-1), Fraction(0), Fraction(0), -mins[0]],
                [Fraction(0), Fraction(1), Fraction(0), maxs[1]],
                [Fraction(0), Fraction(-1), Fraction(0), -mins[1]],
                [Fraction(0), Fraction(0), Fraction(1), maxs[2]],
                [Fraction(0), Fraction(0), Fraction(-1), -mins[2]],
            ]
            domain_obj = {"halfspaces": [[_fmt_fraction(v) for v in h] for h in hs]}
        else:
            if not isinstance(domain, dict) or "halfspaces" not in domain:
                raise ValueError("3D domain must be {'halfspaces': ...}")
            domain_obj = {"halfspaces": []}
            for h in domain["halfspaces"]:
                if len(h) != 4:
                    raise ValueError("3D halfspaces must be [a,b,c,r]")
                domain_obj["halfspaces"].append([
                    _fmt_fraction(_coord_fraction(float(v), mode=cfg.coordinate_mode, max_denominator=cfg.max_denominator))
                    for v in h
                ])
        return {
            "schema": "min_ipd_instance3_v1",
            "points": qpts,
            "segments": [[int(a), int(b)] for a, b in segments],
            "domain": domain_obj,
            "metadata": output_meta,
        }
    raise ValueError("embedded graph dimension must be 2 or 3")


def validate_min_ipd_export(instance: dict[str, Any]) -> bool:
    """Validate with min_ipd when available; otherwise perform schema sanity checks."""
    schema = instance.get("schema")
    if schema not in {"min_ipd_instance_v1", "min_ipd_instance3_v1"}:
        raise ValidationError("not a recognized min_ipd instance schema")
    try:
        if schema == "min_ipd_instance_v1":
            from min_ipd.instance import instance_from_dict  # type: ignore
            from min_ipd.geometry import sha256_hex  # type: ignore

            instance_from_dict(instance, sha256_hex(b"mask2graph"))
        else:
            from min_ipd.instance3 import instance3_from_dict  # type: ignore
            from min_ipd.geometry import sha256_hex  # type: ignore

            instance3_from_dict(instance, sha256_hex(b"mask2graph"))
    except ImportError:
        if not isinstance(instance.get("points"), list) or not isinstance(instance.get("segments"), list):
            raise ValidationError("invalid min_ipd export")
    return True
