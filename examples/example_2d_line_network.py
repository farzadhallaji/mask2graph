from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maskgraph import extract_graph


def main() -> None:
    mask = np.zeros((21, 21), dtype=np.uint8)
    mask[10, 2:19] = 1
    mask[4:17, 10] = 1

    graph = extract_graph(mask)
    print(f"nodes={len(graph.nodes)} edges={len(graph.edges)}")
    print(f"first edge length={graph.edges[0].length:.3f}")


if __name__ == "__main__":
    main()
