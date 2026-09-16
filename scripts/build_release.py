#!/usr/bin/env python3
"""Build wheel + sdist with the configured setuptools backend and hash them."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir()

    # Calling the PEP 517 backend directly keeps this script usable in offline
    # release environments once the build-system requirements are installed.
    from setuptools import build_meta

    sdist_name = build_meta.build_sdist(str(DIST))
    wheel_name = build_meta.build_wheel(str(DIST))
    artifacts = [DIST / sdist_name, DIST / wheel_name]
    checksums = "".join(f"{sha256(p)}  {p.name}\n" for p in sorted(artifacts))
    (DIST / "SHA256SUMS").write_text(checksums, encoding="utf-8")
    print(checksums, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
