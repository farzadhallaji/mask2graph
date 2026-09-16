"""Single YAML-driven paper CLI for mask2graph."""

from __future__ import annotations

import argparse
from pathlib import Path

from ._version import __version__

from .runner import run_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mask2graph",
        description="Run one strict self-contained mask2graph YAML experiment.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("config", type=Path, nargs="?", help="self-contained experiment YAML path")
    args = parser.parse_args(argv)
    if args.config is None:
        parser.error("the following arguments are required: config")
    result = run_experiment(args.config)
    print(result.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
