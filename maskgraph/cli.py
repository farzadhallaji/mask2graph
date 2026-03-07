"""Minimal CLI for graph extraction."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .api import extract_graph
from .config import ExtractConfig
from .debug import save_debug_artifacts
from .serialize import to_json


def _load_mask(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        data = np.load(path)
        if len(data.files) == 0:
            raise ValueError("npz archive contains no arrays")
        return data[data.files[0]]
    raise ValueError("input must be .npy or .npz")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="maskgraph")
    sub = parser.add_subparsers(dest="command", required=True)
    p_extract = sub.add_parser("extract", help="Extract graph from mask")
    p_extract.add_argument("input", type=Path)
    p_extract.add_argument("output", type=Path)
    p_extract.add_argument("--debug-dir", type=Path, default=None)

    args = parser.parse_args(argv)
    if args.command == "extract":
        mask = _load_mask(args.input)
        cfg = ExtractConfig()
        if args.debug_dir is None:
            graph = extract_graph(mask, config=cfg, return_debug=False)
            args.output.write_text(to_json(graph), encoding="utf-8")
        else:
            graph, debug = extract_graph(mask, config=cfg, return_debug=True)
            args.output.write_text(to_json(graph), encoding="utf-8")
            save_debug_artifacts(debug, args.debug_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
