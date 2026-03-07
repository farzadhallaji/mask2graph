from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maskgraph import ExtractConfig, extract_graph


def main() -> None:
    mask = np.zeros((21, 21, 21), dtype=np.uint8)
    mask[10, 10, 4:17] = 1
    mask[5:11, 10, 10] = 1

    cfg = ExtractConfig()
    cfg.cleanup.min_object_size = 2.0
    cfg.normalize.junction_dilation_iters = 1
    cfg.normalize.contract_short_edges_below = 1.0

    graph = extract_graph(mask, spacing=(1.0, 1.0, 1.0), config=cfg)
    print(f"nodes={len(graph.nodes)} edges={len(graph.edges)}")
    print(f"junctions={sum(1 for n in graph.nodes if n.type == 'junction')}")


if __name__ == "__main__":
    main()
