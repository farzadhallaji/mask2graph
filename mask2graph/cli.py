"""Single YAML-driven paper CLI for mask2graph."""

from __future__ import annotations

import argparse
from pathlib import Path

from .runner import run_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mask2graph",
        description="Run one strict self-contained mask2graph YAML experiment.",
    )
    parser.add_argument("config", type=Path, help="self-contained experiment YAML path")
    args = parser.parse_args(argv)
    result = run_experiment(args.config)
    print(result.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
