#!/usr/bin/env python3
"""Fast repository/config sanity check before expensive mask2graph runs."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mask2graph import load_run_config  # noqa: E402


def main() -> int:
    required = [
        ROOT / "IMPLEMENTATION_RULES.md",
        ROOT / "CONFIG_POLICY.md",
        ROOT / "pyproject.toml",
        ROOT / "configs",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        raise SystemExit(f"missing required paper setup paths: {missing}")

    configs = sorted((ROOT / "configs").glob("*.yaml"))
    if not configs:
        raise SystemExit("no runnable YAML configs found under configs/")
    checked = []
    for path in configs:
        cfg = load_run_config(path)
        if not cfg.input.path.is_file():
            raise SystemExit(f"configured input does not exist: {cfg.input.path}")
        checked.append({"config": str(path.relative_to(ROOT)), "name": cfg.name, "ndim": cfg.input.expected_ndim})

    notebook = ROOT / "notebooks" / "mask_graph_augmentation_demo.ipynb"
    if notebook.is_file():
        raw = notebook.read_text(encoding="utf-8")
        forbidden = ["rotate_graph(", "flip_graph(", "crop_graph_box(", "cfg.simplify.method ="]
        bad = [token for token in forbidden if token in raw]
        if bad:
            raise SystemExit(f"notebook contains duplicated behavioral policy: {bad}")

    print(json.dumps({"status": "ok", "configs": checked}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
