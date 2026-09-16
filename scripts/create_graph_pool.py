#!/usr/bin/env python3
"""Create a graph pool from MassRoads and DRIVE label masks."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mask2graph import ExtractConfig, extract_graph, to_json  # noqa: E402

DEFAULT_MASS_ROADS_LABELS = Path("/home/ri/Desktop/Projects/Datasets/ProcessedDatasets/MassRoads/train/labels")
DEFAULT_DRIVE_LABELS = Path("/home/ri/Desktop/Projects/Datasets/ProcessedDatasets/DRIVE/labels")
DEFAULT_OUTPUT = ROOT / "graph_pool"
DEFAULT_SEED = 20260916
DEFAULT_MASS_ROADS_COUNT = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample MassRoads masks, use all DRIVE masks, copy original "
            "mask files, and save full-mask mask2graph JSON graphs."
        )
    )
    parser.add_argument("--mass-roads-labels", type=Path, default=DEFAULT_MASS_ROADS_LABELS)
    parser.add_argument("--drive-labels", type=Path, default=DEFAULT_DRIVE_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mass-roads-count", type=int, default=DEFAULT_MASS_ROADS_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true", help="remove output directory before generating")
    return parser.parse_args()


def make_config() -> ExtractConfig:
    cfg = ExtractConfig()
    cfg.geometry.compute_curvature = False
    cfg.simplify.method = "none"
    cfg.validation.enabled = False
    cfg.validation.validate_coverage = False
    cfg.validation.validate_topology = False
    cfg.validation.validate_embedding = False
    return cfg


def load_binary_mask(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    if arr.ndim > 2:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"expected 2D mask after optional channel selection, got {arr.shape}")
    return np.asarray(arr != 0, dtype=bool)


def select_inputs(mass_roads_labels: Path, drive_labels: Path, mass_roads_count: int, seed: int) -> tuple[list[Path], list[Path], int]:
    mass_roads_all = sorted(mass_roads_labels.glob("*.npy"))
    drive_all = sorted(drive_labels.glob("*.npy"))
    if len(mass_roads_all) < mass_roads_count:
        raise ValueError(f"MassRoads has only {len(mass_roads_all)} masks, need {mass_roads_count}")
    if not drive_all:
        raise ValueError(f"no DRIVE .npy masks found in {drive_labels}")
    rng = random.Random(seed)
    mass_roads_selected = sorted(rng.sample(mass_roads_all, mass_roads_count), key=lambda p: p.name)
    return mass_roads_selected, drive_all, len(mass_roads_all)


def process_dataset(dataset_name: str, paths: list[Path], output_root: Path, config: ExtractConfig) -> dict:
    dataset_dir = output_root / dataset_name
    mask_dir = dataset_dir / "masks"
    graph_dir = dataset_dir / "graphs"
    mask_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    start_dataset = perf_counter()
    for idx, source_path in enumerate(paths, start=1):
        start = perf_counter()
        copied_mask_path = mask_dir / source_path.name
        graph_path = graph_dir / f"{source_path.stem}.graph.json"
        try:
            shutil.copy2(source_path, copied_mask_path)
            mask = load_binary_mask(source_path)
            graph = extract_graph(mask, config=config, return_debug=False)
            graph_path.write_text(to_json(graph) + "\n", encoding="utf-8")
            record = {
                "source_path": str(source_path),
                "mask_path": str(copied_mask_path.relative_to(output_root)),
                "graph_path": str(graph_path.relative_to(output_root)),
                "shape": list(mask.shape),
                "foreground_pixels": int(mask.sum()),
                "nodes": len(graph.nodes),
                "edges": len(graph.edges),
                "graph_hash": graph.meta.graph_hash,
                "elapsed_seconds": round(perf_counter() - start, 6),
                "status": "ok",
            }
            print(
                f"[{dataset_name} {idx:03d}/{len(paths):03d}] "
                f"{source_path.name} -> nodes={len(graph.nodes)} edges={len(graph.edges)} "
                f"time={record['elapsed_seconds']:.2f}s",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            record = {
                "source_path": str(source_path),
                "mask_path": str(copied_mask_path.relative_to(output_root)),
                "graph_path": str(graph_path.relative_to(output_root)),
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": round(perf_counter() - start, 6),
            }
            print(f"[{dataset_name} {idx:03d}/{len(paths):03d}] FAILED {source_path.name}: {exc}", flush=True)
        records.append(record)

    dataset_manifest = {
        "count": len(paths),
        "ok": sum(1 for record in records if record["status"] == "ok"),
        "failed": sum(1 for record in records if record["status"] != "ok"),
        "elapsed_seconds": round(perf_counter() - start_dataset, 6),
        "records": records,
    }
    (dataset_dir / "manifest.json").write_text(
        json.dumps(dataset_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return dataset_manifest


def main() -> int:
    args = parse_args()
    output_root = args.output.resolve()
    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    mass_roads_selected, drive_all, mass_roads_total = select_inputs(
        args.mass_roads_labels,
        args.drive_labels,
        args.mass_roads_count,
        args.seed,
    )
    config = make_config()
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "random_seed": args.seed,
        "mass_roads_source": str(args.mass_roads_labels),
        "mass_roads_total_available": mass_roads_total,
        "mass_roads_selected_count": len(mass_roads_selected),
        "drive_source": str(args.drive_labels),
        "drive_selected_count": len(drive_all),
        "config": config.to_dict(),
        "datasets": {},
    }

    (output_root / "mass_roads_selected.txt").write_text(
        "\n".join(path.name for path in mass_roads_selected) + "\n",
        encoding="utf-8",
    )
    manifest["datasets"]["mass_roads"] = process_dataset("mass_roads", mass_roads_selected, output_root, config)
    manifest["datasets"]["drive"] = process_dataset("drive", drive_all, output_root, config)
    manifest["summary"] = {
        "total_requested": len(mass_roads_selected) + len(drive_all),
        "total_ok": sum(dataset["ok"] for dataset in manifest["datasets"].values()),
        "total_failed": sum(dataset["failed"] for dataset in manifest["datasets"].values()),
    }
    manifest["failures"] = [
        record
        for dataset in manifest["datasets"].values()
        for record in dataset["records"]
        if record["status"] != "ok"
    ]
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest["summary"], indent=2), flush=True)
    return 0 if manifest["summary"]["total_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
