from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mask2graph import ExtractConfig, extract_graph


def main() -> None:
    mask = np.zeros((21, 21), dtype=np.uint8)
    mask[10, 2:19] = 1
    mask[4:17, 10] = 1

    cfg = ExtractConfig()
    cfg.cleanup.max_hole_size = 9.0
    cfg.cleanup.max_hole_radius = 1.5
    cfg.normalize.junction_dilation_iters = 1
    cfg.normalize.prune_spurs_below = 2.0

    graph, debug = extract_graph(mask, config=cfg, return_debug=True)
    print(f"nodes={len(graph.nodes)} edges={len(graph.edges)}")
    print(f"first edge length={graph.edges[0].length:.3f}")
    print(
        "cleanup removed_objects="
        f"{debug.cleanup_report.n_removed_objects} filled_holes={debug.cleanup_report.n_filled_holes}"
    )


if __name__ == "__main__":
    main()
