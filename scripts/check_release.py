#!/usr/bin/env python3
"""Fail loudly when release metadata or the public package surface is inconsistent."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tomllib
from typing import NoReturn

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mask2graph  # noqa: E402


def fail(message: str) -> NoReturn:
    raise SystemExit(message)


def main() -> int:
    version = mask2graph.__version__
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    if pyproject["project"].get("dynamic") != ["version"]:
        fail("pyproject.toml must use the package version as a dynamic field")
    attr = pyproject.get("tool", {}).get("setuptools", {}).get("dynamic", {}).get("version", {}).get("attr")
    if attr != "mask2graph._version.__version__":
        fail("pyproject.toml version source is not mask2graph._version.__version__")

    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    if str(citation.get("version")) != version:
        fail(f"CITATION.cff version {citation.get('version')!r} != package version {version!r}")

    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    if f"## {version} " not in changelog:
        fail(f"CHANGELOG.md has no release section for {version}")

    required = [
        "README.md", "LICENSE", "CHANGELOG.md", "CITATION.cff", "CONFIG_POLICY.md",
        "IMPLEMENTATION_RULES.md", "RELEASE.md", "pyproject.toml",
    ]
    missing = [name for name in required if not (ROOT / name).is_file()]
    if missing:
        fail(f"missing release files: {missing}")

    try:
        tracked = subprocess.check_output(["git", "ls-files", "--error-unmatch", "IMPLEMENTATION_RULES.md"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        tracked = None
    if (ROOT / ".git").exists() and tracked != "IMPLEMENTATION_RULES.md":
        fail("IMPLEMENTATION_RULES.md exists but is not tracked by git")

    out = subprocess.check_output([sys.executable, "-m", "mask2graph", "--version"], cwd=ROOT, text=True).strip()
    if out != f"mask2graph {version}":
        fail(f"CLI version mismatch: {out!r}")

    print(json.dumps({"status": "ok", "version": version, "cli": out}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
