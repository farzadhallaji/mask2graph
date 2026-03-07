from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maskgraph import extract_graph


def main() -> None:
    mask = np.zeros((24, 24), dtype=np.uint8)
    mask[6, 6:18] = 1
    mask[17, 6:18] = 1
    mask[6:18, 6] = 1
    mask[6:18, 17] = 1

    graph = extract_graph(mask)
    print(f"nodes={len(graph.nodes)} edges={len(graph.edges)}")
    print(f"self_loops={sum(1 for e in graph.edges if e.is_self_loop)}")


if __name__ == "__main__":
    main()
