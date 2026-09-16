from pathlib import Path
import subprocess
import sys
import tomllib

import yaml

import mask2graph

ROOT = Path(__file__).resolve().parents[1]


def test_release_version_is_single_and_consistent():
    version = mask2graph.__version__
    assert version == "1.4.0"
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["dynamic"] == ["version"]
    assert pyproject["tool"]["setuptools"]["dynamic"]["version"]["attr"] == "mask2graph._version.__version__"
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert citation["version"] == version
    assert f"## {version} " in (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")


def test_module_cli_reports_same_version():
    out = subprocess.check_output([sys.executable, "-m", "mask2graph", "--version"], cwd=ROOT, text=True).strip()
    assert out == f"mask2graph {mask2graph.__version__}"


def test_repo_contract_is_present():
    assert (ROOT / "IMPLEMENTATION_RULES.md").is_file()
    assert (ROOT / "CONFIG_POLICY.md").is_file()
    assert (ROOT / "RELEASE.md").is_file()
