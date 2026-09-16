"""Optional matplotlib visualization helpers for masks and extracted graphs.

The core package has no hard matplotlib dependency.  Importing this module is
safe without matplotlib installed; plotting functions raise an actionable error
only when called.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .types import EmbeddedGraph, Mask2Graph, MaskGraphResult


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise ModuleNotFoundError(
            "visualization needs matplotlib; install mask2graph[viz] or "
            "mask2graph[notebook]"
        ) from exc
    return plt


def _topology_graph(value: Mask2Graph | MaskGraphResult) -> Mask2Graph:
    return value.topology_graph if isinstance(value, MaskGraphResult) else value


def graph_bounds(value: Mask2Graph | EmbeddedGraph | MaskGraphResult) -> tuple[np.ndarray, np.ndarray]:
    """Return physical-coordinate lower/upper bounds for graph geometry."""
    if isinstance(value, MaskGraphResult):
        value = value.topology_graph
    if isinstance(value, Mask2Graph):
        ndim = value.meta.ndim
        pts: list[np.ndarray] = []
        pts.extend(np.asarray(n.xyz, dtype=np.float64)[:ndim] for n in value.nodes)
        for edge in value.edges:
            if len(edge.path_xyz):
                pts.extend(np.asarray(edge.path_xyz, dtype=np.float64)[:, :ndim])
    else:
        ndim = value.ndim
        pts = [np.asarray(v.xyz, dtype=np.float64)[:ndim] for v in value.vertices]
    if not pts:
        z = np.zeros(ndim, dtype=np.float64)
        return z.copy(), z.copy()
    arr = np.asarray(pts, dtype=np.float64)
    return arr.min(axis=0), arr.max(axis=0)


def central_crop_bounds(
    value: Mask2Graph | EmbeddedGraph | MaskGraphResult,
    *,
    fraction: float = 0.75,
) -> tuple[float, ...]:
    """Return centered axis-aligned crop bounds covering ``fraction`` of each span.

    This is a notebook convenience helper; it does not modify the graph.
    """
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    lo, hi = graph_bounds(value)
    center = (lo + hi) / 2.0
    half = (hi - lo) * float(fraction) / 2.0
    low = center - half
    high = center + half
    return tuple(float(x) for pair in zip(low, high) for x in pair)


def _prepare_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError("plot_mask supports one 2D mask; select a slice/frame first")
    return np.asarray(arr != 0, dtype=bool)


def _mask_extent(shape: tuple[int, int], spacing: tuple[float, float]) -> tuple[float, float, float, float]:
    h, w = shape
    sy, sx = (float(spacing[0]), float(spacing[1]))
    # Matches index_to_xyz: x=column*sx, y=row*sy and imshow(origin='upper').
    return (-0.5 * sx, (w - 0.5) * sx, (h - 0.5) * sy, -0.5 * sy)


def plot_mask(
    mask: np.ndarray,
    *,
    spacing: tuple[float, float] = (1.0, 1.0),
    ax=None,
    title: str | None = "Binary mask",
):
    """Display one 2D binary mask in the same physical coordinate convention as graphs."""
    plt = _plt()
    arr = _prepare_mask(mask)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(
        arr,
        cmap="gray",
        interpolation="nearest",
        origin="upper",
        extent=_mask_extent(arr.shape, spacing),
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title is not None:
        ax.set_title(title)
    return ax


def _plot_topology_2d(graph: Mask2Graph, ax, *, show_nodes: bool, show_node_ids: bool, full_paths: bool) -> None:
    from matplotlib.collections import LineCollection

    segments = []
    for edge in graph.edges:
        if full_paths or edge.simplified_xyz is None or len(edge.simplified_xyz) < 2:
            p = np.asarray(edge.path_xyz, dtype=np.float64)
        else:
            p = np.asarray(edge.simplified_xyz, dtype=np.float64)
        if len(p) >= 2:
            segments.extend(np.stack([p[:-1, :2], p[1:, :2]], axis=1))
    if segments:
        ax.add_collection(LineCollection(segments, linewidths=1.2))
        ax.autoscale_view()
    if show_nodes and graph.nodes:
        xy = np.asarray([n.xyz[:2] for n in graph.nodes], dtype=float)
        ax.scatter(xy[:, 0], xy[:, 1], s=18, zorder=3)
        if show_node_ids:
            for n in graph.nodes:
                ax.text(float(n.xyz[0]), float(n.xyz[1]), str(n.id), fontsize=7)


def _plot_embedded_2d(graph: EmbeddedGraph, ax, *, show_nodes: bool, show_node_ids: bool) -> None:
    from matplotlib.collections import LineCollection

    by_id = {v.id: v for v in graph.vertices}
    lines = [
        [(by_id[s.u].xyz[0], by_id[s.u].xyz[1]), (by_id[s.v].xyz[0], by_id[s.v].xyz[1])]
        for s in graph.segments
    ]
    if lines:
        ax.add_collection(LineCollection(lines, linewidths=1.2))
        ax.autoscale_view()
    if show_nodes and graph.vertices:
        xy = np.asarray([v.xyz[:2] for v in graph.vertices], dtype=float)
        ax.scatter(xy[:, 0], xy[:, 1], s=12, zorder=3)
        if show_node_ids:
            for v in graph.vertices:
                ax.text(float(v.xyz[0]), float(v.xyz[1]), str(v.id), fontsize=7)


def _ensure_3d_axis(ax):
    plt = _plt()
    if ax is None:
        fig = plt.figure(figsize=(8, 7))
        return fig.add_subplot(111, projection="3d")
    if not hasattr(ax, "get_zlim"):
        raise ValueError("3D graph plotting requires a matplotlib 3D axis")
    return ax


def _plot_topology_3d(graph: Mask2Graph, ax, *, show_nodes: bool, show_node_ids: bool, full_paths: bool) -> None:
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    segments = []
    for edge in graph.edges:
        if full_paths or edge.simplified_xyz is None or len(edge.simplified_xyz) < 2:
            p = np.asarray(edge.path_xyz, dtype=np.float64)
        else:
            p = np.asarray(edge.simplified_xyz, dtype=np.float64)
        if len(p) >= 2:
            segments.extend(np.stack([p[:-1, :3], p[1:, :3]], axis=1))
    if segments:
        ax.add_collection3d(Line3DCollection(segments, linewidths=1.2))
    if show_nodes and graph.nodes:
        xyz = np.asarray([n.xyz for n in graph.nodes], dtype=float)
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=18)
        if show_node_ids:
            for n in graph.nodes:
                ax.text(float(n.xyz[0]), float(n.xyz[1]), float(n.xyz[2]), str(n.id), fontsize=7)


def _plot_embedded_3d(graph: EmbeddedGraph, ax, *, show_nodes: bool, show_node_ids: bool) -> None:
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    by_id = {v.id: v for v in graph.vertices}
    lines = [
        [by_id[s.u].xyz[:3], by_id[s.v].xyz[:3]]
        for s in graph.segments
    ]
    if lines:
        ax.add_collection3d(Line3DCollection(lines, linewidths=1.2))
    if show_nodes and graph.vertices:
        xyz = np.asarray([v.xyz for v in graph.vertices], dtype=float)
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=12)
        if show_node_ids:
            for v in graph.vertices:
                ax.text(float(v.xyz[0]), float(v.xyz[1]), float(v.xyz[2]), str(v.id), fontsize=7)


def plot_graph(
    value: Mask2Graph | EmbeddedGraph | MaskGraphResult,
    *,
    representation: Literal["topology", "embedded"] = "embedded",
    background: np.ndarray | None = None,
    background_spacing: tuple[float, float] | None = None,
    ax=None,
    title: str | None = None,
    show_nodes: bool = True,
    show_node_ids: bool = False,
    full_paths: bool = False,
):
    """Plot topology or embedded graph geometry.

    ``background`` is supported for 2D graphs and uses the same physical
    coordinate convention as extraction, making direct mask/graph overlays safe.
    """
    if representation not in {"topology", "embedded"}:
        raise ValueError("representation must be 'topology' or 'embedded'")
    if isinstance(value, MaskGraphResult):
        top = value.topology_graph
        graph = top if representation == "topology" else value.embedded_graph
        default_spacing = tuple(float(v) for v in top.meta.spacing)
    elif isinstance(value, Mask2Graph):
        if representation == "embedded":
            raise ValueError("an EmbeddedGraph or MaskGraphResult is required for representation='embedded'")
        graph = value
        default_spacing = tuple(float(v) for v in value.meta.spacing)
    else:
        if representation == "topology":
            raise ValueError("a Mask2Graph or MaskGraphResult is required for representation='topology'")
        graph = value
        default_spacing = (1.0, 1.0)

    ndim = graph.meta.ndim if isinstance(graph, Mask2Graph) else graph.ndim
    plt = _plt()
    if ndim == 2:
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 8))
        if background is not None:
            spacing = background_spacing or default_spacing[:2]
            plot_mask(background, spacing=(float(spacing[0]), float(spacing[1])), ax=ax, title=None)
        if isinstance(graph, Mask2Graph):
            _plot_topology_2d(graph, ax, show_nodes=show_nodes, show_node_ids=show_node_ids, full_paths=full_paths)
        else:
            _plot_embedded_2d(graph, ax, show_nodes=show_nodes, show_node_ids=show_node_ids)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    else:
        if background is not None:
            raise ValueError("background mask overlay is only supported for 2D graphs")
        ax = _ensure_3d_axis(ax)
        if isinstance(graph, Mask2Graph):
            _plot_topology_3d(graph, ax, show_nodes=show_nodes, show_node_ids=show_node_ids, full_paths=full_paths)
        else:
            _plot_embedded_3d(graph, ax, show_nodes=show_nodes, show_node_ids=show_node_ids)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        # Equal-ish data aspect where the matplotlib version supports it.
        lo, hi = graph_bounds(graph)
        span = np.maximum(hi - lo, 1e-12)
        try:
            ax.set_box_aspect(span)
        except Exception:  # pragma: no cover - older matplotlib fallback
            pass
    if title is not None:
        ax.set_title(title)
    return ax


def plot_result(
    result: MaskGraphResult,
    *,
    representation: Literal["topology", "embedded"] = "embedded",
    background: np.ndarray | None = None,
    ax=None,
    title: str | None = None,
    show_nodes: bool = True,
    show_node_ids: bool = False,
    full_paths: bool = False,
):
    """Named convenience wrapper around :func:`plot_graph`."""
    return plot_graph(
        result,
        representation=representation,
        background=background,
        background_spacing=tuple(float(v) for v in result.topology_graph.meta.spacing[:2]),
        ax=ax,
        title=title,
        show_nodes=show_nodes,
        show_node_ids=show_node_ids,
        full_paths=full_paths,
    )


def plot_augmentation_sequence(
    mask: np.ndarray,
    original: MaskGraphResult,
    rotated: MaskGraphResult,
    flipped: MaskGraphResult,
    cropped: MaskGraphResult,
    *,
    representation: Literal["topology", "embedded"] = "embedded",
):
    """Five-panel notebook view: mask, graph, rotation, flip, crop (2D only)."""
    if original.topology_graph.meta.ndim != 2:
        raise ValueError("plot_augmentation_sequence currently supports 2D results")
    plt = _plt()
    fig, axes = plt.subplots(1, 5, figsize=(24, 5))
    plot_mask(mask, spacing=tuple(float(v) for v in original.topology_graph.meta.spacing), ax=axes[0], title="1. Binary mask")
    plot_result(original, representation=representation, ax=axes[1], title="2. Extracted graph")
    plot_result(rotated, representation=representation, ax=axes[2], title="3. Rotated graph")
    plot_result(flipped, representation=representation, ax=axes[3], title="4. Flipped graph")
    plot_result(cropped, representation=representation, ax=axes[4], title="5. Cropped graph")
    fig.tight_layout()
    return fig, axes


__all__ = [
    "central_crop_bounds",
    "graph_bounds",
    "plot_augmentation_sequence",
    "plot_graph",
    "plot_mask",
    "plot_result",
]
