# Releasing mask2graph

`mask2graph` has one public Python package, one package version source, and one paper-facing YAML execution path.

## Release checks

From a clean checkout:

```bash
python -m pytest -q
python scripts/check_paper_setup.py
python scripts/check_release.py
python scripts/build_release.py
```

`check_release.py` verifies that the package version, citation metadata, changelog, public import, CLI version, and repository contract are aligned.

`build_release.py` creates a wheel and source distribution under `dist/` and writes `dist/SHA256SUMS`.

Before publication, install the wheel into a fresh environment and run at least the import/CLI smoke test. For paper experiments, also validate the intended YAML and retain the generated run archive.

## Publication

The built wheel and sdist are the public package artifacts. The source repository/tag is the canonical auditable implementation. Do not publish the historical `package-only` branch as a second implementation.
