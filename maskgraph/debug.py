"""Debug artifact helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .types import DebugArtifacts


def save_debug_artifacts(artifacts: DebugArtifacts, debug_dir: str | Path) -> None:
    path = Path(debug_dir)
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "mask_input.npy", artifacts.mask_input)
    np.save(path / "mask_processed.npy", artifacts.mask_processed)
    np.save(path / "skeleton.npy", artifacts.skeleton)
    np.save(path / "degree_map.npy", artifacts.degree_map)
    np.save(path / "node_candidates.npy", artifacts.node_candidates)
    np.save(path / "node_labels.npy", artifacts.node_labels)
    np.save(path / "pruned_skeleton.npy", artifacts.pruned_skeleton)
    np.save(path / "component_labels.npy", artifacts.component_labels)
