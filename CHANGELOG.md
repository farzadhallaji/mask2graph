# Changelog

## 1.4.0 - 2026-09-16 — unified public package release

- consolidated mask extraction, topology graphs, embedded graphs, augmentation, geometric cropping, visualization, strict YAML experiments, and `min_ipd` export into one supported `mask2graph` package;
- made `mask2graph._version.__version__` the single package-version source and aligned package metadata/citation/release checks;
- added release-quality wheel/sdist packaging, `python -m mask2graph`, `mask2graph --version`, and release validation/build scripts;
- added the repository-wide implementation contract and release-readiness checks;
- kept the YAML experiment runner as the only paper-facing behavioral CLI path.

## 1.3.0 - 2026-09-16 — strict paper-run configuration

- added one self-contained strict YAML as the only paper-facing CLI behavior source;
- unknown/missing/wrongly typed YAML values fail before extraction;
- added a single YAML runner shared by CLI and notebook;
- every YAML run archives source/resolved config, environment metadata, code snapshot, stage outputs, and summary;
- moved the augmentation notebook onto the same YAML execution path;
- added reproducibility/config-policy tests and paper-setup sanity script;
- batched graph plotting with matplotlib collections for large graph visualization.

## 1.2.0 - 2026-09-16 — graph augmentation

- added graph-to-graph 2D/3D rotation, reflection, translation, axis permutation, transform composition/inversion, and deterministic random augmentation pipelines;
- added geometric full-polyline box clipping with boundary endpoints, radius interpolation, multiple fragments, cycle handling, provenance, and keep-size crop-as-translation;
- added explicit source/current raster-index validity so transformed or synthetic coordinates are never silently rounded into fake pixel/voxel provenance;
- added exact-transform/grid-alignment metadata and rationalization-error reporting in `min_ipd` exports;
- added rigid/crop topology validation and augmentation regression tests.

## 1.1.0 - 2026-09-16

Unified mask-to-PSLG release:

- preserves the existing 2D/3D mask2graph API while adding `mask_to_graph(...)`,
- replaces floating junction centroids with supported lattice anchors and deterministic junction-support MST provenance,
- certifies branch coverage and records raw/junction-cleaned/logical/embedded topology diagnostics,
- keeps original skeleton branch samples while adding arclength, tangent, turning-angle, curvature, radius-extrema, chord and tortuosity profiles,
- adds non-destructive RDP and minimum-segment DAG simplification,
- adds 2D/3D embedded-graph validation with no undeclared intersections, duplicate/zero edges, or node-on-edge degeneracies,
- adds explicit boundary endpoints, deterministic graph/config hashes, graph cropping helpers, and full-result serialization,
- adds exact-rational `min_ipd` 2D/3D export plus a `mask2graph pslg` CLI,
- keeps destructive spur/component/cycle cleanup opt-in and records whether such a policy changed topology,
- expands the test suite with unified pipeline, 3D, cycle, simplification, export, and determinism regressions.

## 1.0.0 - 2026-03-07

Breaking release with package rename and stable API baseline:

- package name changed to `mask2graph`,
- import path is `mask2graph` only (no compatibility alias),
- deterministic 2D/3D extraction, conservative cleanup, and graph normalization pipeline retained,
- NetworkX interoperability via `to_networkx(...)` included.

Migration note:

- Replace all old package imports with `from mask2graph import ...`.

## 0.1.0 - 2026-03-07

First stable release:

- deterministic 2D/3D mask-to-graph extraction with stable ordering,
- conservative mask cleanup with auditable cleanup reports,
- junction-region stabilization and iterative graph normalization,
- tiny-cycle filtering and short internal-edge contraction controls,
- JSON serialization/deserialization with schema versioning,
- NetworkX interoperability via `to_networkx(...)`,
- expanded test coverage for extraction, normalization, and conversion utilities.
