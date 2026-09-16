import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mask2graph import (
    ExtractConfig,
    central_crop_bounds,
    crop_graph_box,
    flip_graph,
    graph_bounds,
    mask_to_graph,
    plot_augmentation_sequence,
    plot_graph,
    plot_mask,
    plot_result,
    rotate_graph,
)


def _mask2d():
    m = np.zeros((21, 25), dtype=np.uint8)
    m[10, 3:22] = 1
    m[4:17, 12] = 1
    return m


def test_plot_mask_and_result_2d():
    mask = _mask2d()
    cfg = ExtractConfig()
    cfg.simplify.method = "rdp"
    result = mask_to_graph(mask, config=cfg)

    ax = plot_mask(mask)
    assert ax.get_title() == "Binary mask"
    plt.close(ax.figure)

    ax = plot_result(result, background=mask, show_nodes=False, title="overlay")
    assert ax.get_title() == "overlay"
    assert len(ax.lines) + len(ax.collections) > 0
    plt.close(ax.figure)


def test_bounds_and_sequence_2d():
    mask = _mask2d()
    cfg = ExtractConfig()
    cfg.simplify.method = "rdp"
    original = mask_to_graph(mask, config=cfg)
    lo, hi = graph_bounds(original)
    assert lo.shape == (2,)
    assert hi.shape == (2,)
    assert np.all(hi >= lo)

    bounds = central_crop_bounds(original, fraction=0.8)
    assert len(bounds) == 4
    rotated = rotate_graph(original, 30.0)
    flipped = flip_graph(rotated, "x")
    cropped = crop_graph_box(flipped, bounds=central_crop_bounds(flipped, fraction=0.8), config=cfg)

    fig, axes = plot_augmentation_sequence(mask, original, rotated, flipped, cropped)
    assert len(axes) == 5
    plt.close(fig)


def test_plot_graph_3d():
    mask = np.zeros((9, 9, 9), dtype=np.uint8)
    mask[2:7, 4, 4] = 1
    cfg = ExtractConfig()
    cfg.simplify.method = "rdp"
    result = mask_to_graph(mask, config=cfg)
    ax = plot_graph(result, representation="embedded", show_nodes=True, title="3d")
    assert hasattr(ax, "get_zlim")
    assert ax.get_title() == "3d"
    plt.close(ax.figure)
