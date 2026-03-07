from __future__ import annotations

import time
import tracemalloc
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maskgraph import ExtractConfig, extract_graph
from maskgraph.api import extract_raw_graph
from maskgraph.degree import compute_degree_map
from maskgraph.preprocess import preprocess_mask
from maskgraph.skeleton import skeletonize_mask


def _synthetic_2d(shape: tuple[int, int] = (512, 512)) -> np.ndarray:
    m = np.zeros(shape, dtype=np.uint8)
    for i in range(32, shape[0] - 32, 32):
        m[i, 32 : shape[1] - 32] = 1
    for j in range(64, shape[1] - 64, 96):
        m[32 : shape[0] - 32, j] = 1
    return m


def _synthetic_3d(shape: tuple[int, int, int] = (96, 96, 96)) -> np.ndarray:
    m = np.zeros(shape, dtype=np.uint8)
    for z in range(10, shape[0] - 10, 12):
        m[z, 20:76, 20] = 1
        m[z, 20:76, 75] = 1
        m[z, 20, 20:76] = 1
        m[z, 75, 20:76] = 1
    return m


def run_once(mask: np.ndarray, spacing: tuple[float, ...]) -> dict[str, float]:
    cfg = ExtractConfig()
    mask_bool = mask != 0
    tracemalloc.start()
    t_pre0 = time.perf_counter()
    processed = preprocess_mask(mask_bool, cfg.cleanup)
    t_pre1 = time.perf_counter()
    skeleton = skeletonize_mask(processed, cfg.skeleton)
    t_skel = time.perf_counter()
    degree = compute_degree_map(skeleton)
    raw = extract_raw_graph(skeleton, degree, spacing=spacing, config=cfg)
    t_trace = time.perf_counter()
    t0 = time.perf_counter()
    graph, _ = extract_graph(mask, spacing=spacing, return_debug=True)
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "total_seconds": t1 - t0,
        "preprocess_seconds": t_pre1 - t_pre0,
        "skeletonize_seconds": t_skel - t_pre1,
        "trace_seconds": t_trace - t_skel,
        "peak_memory_mb": peak / (1024 * 1024),
        "nodes": float(len(graph.nodes)),
        "edges": float(len(graph.edges)),
        "raw_nodes": float(len(raw.nodes)),
        "raw_edges": float(len(raw.edges)),
    }


def main() -> None:
    b2d = run_once(_synthetic_2d(), spacing=(1.0, 1.0))
    b3d = run_once(_synthetic_3d(), spacing=(1.0, 1.0, 1.0))
    print("2D benchmark:", b2d)
    print("3D benchmark:", b3d)


if __name__ == "__main__":
    main()
