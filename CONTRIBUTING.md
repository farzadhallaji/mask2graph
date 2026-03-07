# Contributing to maskgraph

Thanks for helping improve `maskgraph`.

## Development setup

```bash
python -m pip install -e ".[dev]"
```

## Run tests

```bash
python -m pytest -q
```

## Contribution guidelines

- Keep changes focused and small when possible.
- Add or update tests for behavior changes.
- Prefer deterministic behavior and stable serialization.
- Include minimal reproducible masks in bug reports/issues.

## Pull requests

- Describe the motivation and expected behavior.
- Link related issues when available.
- Include benchmark notes when touching core extraction logic.
