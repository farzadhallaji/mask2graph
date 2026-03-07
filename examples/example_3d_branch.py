from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maskgraph import extract_graph


def main() -> None:
    mask = np.zeros((21, 21, 21), dtype=np.uint8)
    mask[10, 10, 4:17] = 1
    mask[5:11, 10, 10] = 1

    graph = extract_graph(mask, spacing=(1.0, 1.0, 1.0))
    print(f"nodes={len(graph.nodes)} edges={len(graph.edges)}")
    print(f"junctions={sum(1 for n in graph.nodes if n.type == 'junction')}")


if __name__ == "__main__":
    main()
