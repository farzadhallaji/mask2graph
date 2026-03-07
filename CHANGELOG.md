# Changelog

## 1.0.0 - 2026-03-07

Breaking release with package rename and stable API baseline:

- package name changed from `maskgraph` to `mask2graph`,
- import path changed to `mask2graph` only (no compatibility alias),
- deterministic 2D/3D extraction, conservative cleanup, and graph normalization pipeline retained,
- NetworkX interoperability via `to_networkx(...)` included.

Migration note:

- Replace all imports such as `from maskgraph import ...` with `from mask2graph import ...`.

## 0.1.0 - 2026-03-07

First stable release:

- deterministic 2D/3D mask-to-graph extraction with stable ordering,
- conservative mask cleanup with auditable cleanup reports,
- junction-region stabilization and iterative graph normalization,
- tiny-cycle filtering and short internal-edge contraction controls,
- JSON serialization/deserialization with schema versioning,
- NetworkX interoperability via `to_networkx(...)`,
- expanded test coverage for extraction, normalization, and conversion utilities.
