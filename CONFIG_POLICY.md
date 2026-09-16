# Configuration policy

Paper-facing `mask2graph` runs are controlled by exactly one self-contained YAML file.
The command-line interface accepts the YAML path and no behavioral flags.

## Strictness

- Every required key must be present.
- Unknown keys are errors at every level.
- YAML scalar types are checked without string-to-number or number-to-string coercion.
- Every behavior-affecting extraction, augmentation, visualization, export, input, and output setting is explicit in YAML.
- Relative paths are resolved relative to the YAML file, not the process working directory.
- Random augmentation requires an explicit integer seed.

The low-level geometry API remains available for unit tests and library use. It is not the paper experiment configuration surface.

## Run archive

Every YAML-driven run creates the configured run directory and archives:

- `source.yaml`: byte-for-byte source configuration;
- `resolved.yaml`: fully resolved typed configuration with absolute paths;
- `environment.json`: Python/platform/package/git metadata;
- `code_snapshot/`: the first-party `mask2graph` source plus top-level policy/build files;
- stage graph JSON files requested by the YAML;
- figures requested by the YAML;
- final `min_ipd` JSON when requested.

`overwrite` is explicit. A pre-existing nonempty run directory is an error unless the YAML sets `overwrite: true`.

## One execution path

The current paper path is:

```text
mask2graph experiment.yaml
  -> strict YAML validation
  -> load mask exactly as configured
  -> mask_to_graph(...)
  -> configured graph-to-graph augmentations
  -> optional figures / JSON stages / min_ipd export
  -> archive provenance
```

The notebook calls this same runner. It does not duplicate extraction or augmentation policy in cells.
