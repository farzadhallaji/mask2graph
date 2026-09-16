# mask2graph

Current unified release: **1.4.0**. The package has one supported `mask2graph` import tree, one package-version source, and one YAML-driven paper execution path.

`mask2graph` converts 2D/3D binary masks into deterministic topology graphs and validated straight-line embedded graphs/PSLGs, with radii, geometric profiles, provenance, and optional exact `min_ipd` export.

<img width="1280" height="1278" alt="image" src="https://github.com/user-attachments/assets/823138e9-a4a2-4c28-93b6-72dfed814737" />

## Why mask2graph

Most mask-to-graph pipelines fail at dense junctions, lose loops, or produce unstable IDs across runs. `mask2graph` is built for reproducibility and explicit topology handling:

- deterministic output ordering and JSON payloads
- loop/self-loop preservation
- one API for 2D and 3D masks
- spacing-aware cleanup and geometry
- auditable debug artifacts

## Pipeline

`binary mask -> conservative cleanup -> 2D/3D skeleton -> supported/MST junction resolution -> maximal branch tracing -> topology diagnostics -> optional graph cleanup -> non-destructive geometric simplification -> global embedded-graph validation -> PSLG / min_ipd export`

Interactive graph-augmentation notebook: [`notebooks/mask_graph_augmentation_demo.ipynb`](notebooks/mask_graph_augmentation_demo.ipynb).
It shows the binary mask, extracted graph, graph rotation, graph flip, and geometric graph crop in sequence.

Legacy real-mask overlay notebook (Mass Roads): `mass_roads_graph_overlay.ipynb`.

## Install

From source:

```bash
python -m pip install -e .
```

From a release wheel:

```bash
python -m pip install mask2graph-1.4.0-py3-none-any.whl
```

From GitHub:

```bash
python -m pip install "git+https://github.com/farzadhallaji/mask2graph.git"
```

Install optional NetworkX interoperability dependency:

```bash
python -m pip install -e ".[interop]"
```

Install visualization/notebook support:

```bash
python -m pip install -e ".[viz]"
# or, for JupyterLab as well:
python -m pip install -e ".[notebook]"
```


## Canonical paper-facing run: one YAML

Paper behavior is controlled by one strict self-contained YAML file. The CLI has
no behavioral flags:

```bash
mask2graph configs/retinal_augmentation_demo.yaml
```

The same configuration can be called from Python with a single path:

```python
from mask2graph import run_experiment
run = run_experiment("configs/retinal_augmentation_demo.yaml")
```

Unknown or missing keys and wrong YAML scalar types fail before extraction. Every
run archives `source.yaml`, `resolved.yaml`, `environment.json`, a first-party
`code_snapshot/`, configured stage outputs, and a run summary. See
`CONFIG_POLICY.md` and `IMPLEMENTATION_RULES.md`.

The low-level Python functions below remain available for library use and unit
tests; they are not a second paper configuration path.

## Quickstart

```python
import numpy as np
from mask2graph import ExtractConfig, extract_graph, to_json

mask = np.zeros((11, 11), dtype=np.uint8)
mask[5, 2:9] = 1
mask[2:6, 5] = 1

cfg = ExtractConfig()
cfg.cleanup.max_hole_size = 16.0
cfg.cleanup.max_hole_radius = 1.5
cfg.normalize.junction_dilation_iters = 1
cfg.normalize.prune_spurs_below = 8.0

graph = extract_graph(mask, config=cfg, spacing=(1.0, 1.0))
payload = to_json(graph)
print(len(graph.nodes), len(graph.edges), len(payload))
```

For the complete topology + straight-line pipeline:

```python
from mask2graph import mask_to_graph, graph_to_min_ipd

result = mask_to_graph(mask, spacing=(1.0, 1.0), config=cfg)
print(result.diagnostics.logical_beta0, result.diagnostics.logical_beta1)
print(len(result.embedded_graph.vertices), len(result.embedded_graph.segments))

instance = graph_to_min_ipd(
    result.embedded_graph,
    config=cfg.export,
    name="sample",
)
```

`extract_graph(...)` remains the backward-compatible topology-graph API. `mask_to_graph(...)` is the unified API and returns both the topology graph (full skeleton branch provenance retained) and the validated straight-line embedded graph.

## Visual notebook: mask -> graph -> rotate -> flip -> crop

The canonical notebook is `notebooks/mask_graph_augmentation_demo.ipynb`. It
loads `configs/retinal_augmentation_demo.yaml`, calls `run_experiment(...)`, and
displays the archived figures for the original mask, extracted graph, rotation,
flip, crop, and full sequence. It does not duplicate angles, crop fractions,
spacing, seed, simplification, or export policy in notebook cells.

To change behavior, copy/edit the YAML. For final minimum-segment PSLG runs, set
`extract.simplify.method: optimal` in YAML.

## Configuration reference

`ExtractConfig` has nine groups: `cleanup`, `skeleton`, `junction`, `normalize`, `geometry`, `simplify`, `validation`, `export`, and `determinism`.

All size/length/area thresholds are interpreted in the same units as `spacing` passed to `extract_graph(...)`.
If `spacing=(1, 1)` they behave like pixel units; with physical spacing they behave in physical units.

### `cleanup` (`CleanupConfig`)

Conservative mask repair before skeletonization.

- `enabled` (`bool`, default `True`): turns cleanup stage on/off as a whole.
- `min_object_size` (`float`, default `0.0`): removes tiny disconnected foreground components below this area/volume.
- `max_hole_size` (`float`, default `0.0`): fills enclosed holes up to this area/volume.
- `max_hole_radius` (`float`, default `0.0`): fills enclosed holes up to this max inscribed radius.

Practical guidance:

- Start with `0.0` values for strict no-op behavior.
- For road masks, often useful as a starting point:
  - `min_object_size=4..16`
  - `max_hole_size=9..49`
  - `max_hole_radius=1.0..2.5`
- If narrow separators are getting filled, reduce `max_hole_radius` first.

### `skeleton` (`SkeletonConfig`)

Skeletonization backend selection.

- `method_2d` (`str`, default `"zhang"`): 2D skeleton method.
- `method_3d` (`str`, default `"lee"`): 3D skeleton method.

Practical guidance:

- Keep defaults unless benchmarking shows a clear dataset-specific advantage.

### `normalize` (`NormalizeConfig`)

Graph-side artifact cleanup after tracing.

- `junction_dilation_iters` (`int`, default `0`): expands junction seed regions on the skeleton before clustering into logical nodes.
- `min_component_length` (`float`, default `0.0`): removes connected graph components shorter than this total edge length.
- `prune_spurs_below` (`float`, default `0.0`): prunes short dangling spur edges below this length.
- `min_cycle_length` (`float`, default `0.0`): enables tiny-cycle filtering by max cycle length.
- `max_cycle_area` (`float`, default `0.0`): adds area gate for tiny-cycle filtering (2D area gate).
- `cycle_length_to_radius_ratio` (`float`, default `0.0`): removes cycles whose length is small relative to local radius (helps suppress tiny bubble loops).
- `contract_short_edges_below` (`float`, default `0.0`): contracts short internal edges (mostly junction clutter).
- `normalization_max_iter` (`int`, default `10`): max outer iterations for iterative cleanup.
- `prune_iterations` (`int`, default `100`): max internal iterations for spur pruning.
- `contract_degree2` (`bool`, default `True`): contracts degree-2 chains into longer geometric edges.

Practical guidance:

- If you see red-node clusters at junctions:
  - increase `junction_dilation_iters` to `1..3`
  - then raise `contract_short_edges_below`.
- If tiny whiskers remain:
  - increase `prune_spurs_below`.
- If micro-loops remain:
  - tune `min_cycle_length`, then `max_cycle_area`, then `cycle_length_to_radius_ratio`.

### `junction` (`JunctionConfig`)

- `resolution="mst"` (default): records a deterministic minimum-spanning tree inside each clustered junction support.
- `dilation_iters`: optional junction-zone stabilization.

Logical junction coordinates are always supported skeleton lattice samples, never floating centroids. Original support samples and MST edges are retained as provenance.

### `geometry` (`GeometryConfig`)

Controls spacing-aware arclength, tangent, turning-angle and curvature profiles. EDT radius profiles and min/median/mean/max statistics are retained independently.

### `simplify` (`SimplifyConfig`)

Non-destructive straight-line simplification. The original `path_index`/`path_xyz` is never discarded.

- `enabled` (`bool`, default `True`)
- `epsilon` (`float`, default `1.0`)
- `method`: `"optimal"` (minimum-segment DAG), `"rdp"`, or `"none"`
- `protect_angle_degrees`: protects high-turn samples from shortcuts
- `topology_guard`: validates the resulting global embedded graph
- `fallback_to_original`: falls back to unsimplified branch geometry if simplification alone creates an invalid embedding

### `validation` (`ValidationConfig`)

Controls branch-coverage, topology and embedded-geometry checks. With destructive graph cleanup disabled, the logical graph must preserve the junction-cleaned component count and cycle rank.

### `export` (`ExportConfig`)

Controls exact `min_ipd` coordinate conversion: `lattice_exact`, `scaled_exact`, or bounded-denominator `rationalized`, together with the automatically generated convex-domain margin.

### `determinism` (`DeterminismConfig`)

Output stability controls.

- `float_decimals` (`int`, default `6`): rounding precision for floating values.
- `sort_nodes` (`bool`, default `True`): deterministic node ordering/reindexing.
- `sort_edges` (`bool`, default `True`): deterministic edge ordering/reindexing.

Practical guidance:

- Keep defaults for reproducible JSON outputs and stable diffs.
- Lower `float_decimals` only if payload size matters more than precision.

### Example production profile (roads)

```python
from mask2graph import ExtractConfig

cfg = ExtractConfig()
cfg.cleanup.min_object_size = 9.0
cfg.cleanup.max_hole_size = 36.0
cfg.cleanup.max_hole_radius = 2.0

cfg.normalize.junction_dilation_iters = 2
cfg.normalize.prune_spurs_below = 10.0
cfg.normalize.min_component_length = 25.0
cfg.normalize.min_cycle_length = 16.0
cfg.normalize.max_cycle_area = 24.0
cfg.normalize.cycle_length_to_radius_ratio = 6.0
cfg.normalize.contract_short_edges_below = 3.0
cfg.normalize.normalization_max_iter = 10
```

## Conservative cleanup contract

Mask cleanup is optional and topology-conservative by default. It is intentionally limited to:

- tiny disconnected foreground removal (`min_object_size`)
- tiny enclosed background hole fill (`max_hole_size`, `max_hole_radius`)

It does not perform global opening/closing, branch separation, or hole carving. Those are topology-editing operations and should be explicit downstream choices.

When `return_debug=True`, `DebugArtifacts.cleanup_report` returns counts and physical-size/radius metrics for every cleanup action.

## Graph-side normalization

Residual tracing artifacts are handled in graph space:

- junction-region stabilization (`junction_dilation_iters`)
- short spur pruning (`prune_spurs_below`)
- tiny-cycle removal (`min_cycle_length`, `max_cycle_area`, `cycle_length_to_radius_ratio`)
- short internal-edge contraction (`contract_short_edges_below`)
- bounded iterative cleanup (`normalization_max_iter`)

This separation keeps mask repair conservative while still allowing robust post-trace cleanup.

## NetworkX interoperability

Use `to_networkx(...)` to bridge into the NetworkX ecosystem.

```python
from mask2graph import to_networkx

nx_graph = to_networkx(graph, multigraph=True)
print(nx_graph.number_of_nodes(), nx_graph.number_of_edges())
```

`multigraph=True` preserves parallel edges and self-loops. Node and edge attributes include geometry and topology metadata from `Mask2Graph`.

## Output schema

Serialized payload (`to_json`) includes:

- `schema_version`
- `meta`: version, ndim, shape, spacing, config snapshot, input/config/processed-mask/graph hashes
- `diagnostics`: raw/cleaned/logical/embedded component and cycle-rank data plus coverage/simplification status
- `nodes[]`: supported geometry, topology labels, junction support/MST provenance, boundary flags, radius statistics
- `edges[]`: full original branch path, non-destructive simplified path, lengths/tortuosity, radius/arclength/tangent/angle/curvature profiles, loop/provenance fields

See `mask2graph/types.py` and `mask2graph/serialize.py` for exact fields.

## Examples

- `examples/example_2d_line_network.py`
- `examples/example_2d_loop.py`
- `examples/example_3d_branch.py`

## Validation

Run tests:

```bash
python -m pytest -q
```

## Contributing

Please read `CONTRIBUTING.md`. Bug reports with small reproducible masks are especially useful.

## Citation

If this project helps your work, cite via `CITATION.cff`.

## License

MIT (`LICENSE`).

## Graph-to-graph augmentation

The extracted topology/embedded graph pair can be augmented directly without
rasterization or reskeletonization:

```python
from mask2graph import (
    mask_to_graph, rotate_graph, flip_graph, translate_graph, crop_graph_box,
)

result = mask_to_graph(mask)
result = rotate_graph(result, 90)
result = flip_graph(result, "x")
result = crop_graph_box(result, bounds=(32, 288, 16, 272), keep_size=True)
```

Rigid transforms preserve topology and transform both graph representations with
the same matrix.  Cropping instead clips complete original branch paths, creates
explicit crop-boundary endpoints, recomputes branch geometry, and regenerates the
straight embedded graph.  Source pixel/voxel indices remain provenance and are
never fabricated after non-grid transforms.

For reproducible stochastic augmentation:

```python
from mask2graph import GraphAugmentationPipeline, RandomFlip, RandomRotation, RandomCrop

pipeline = GraphAugmentationPipeline([
    RandomFlip(("x", "y"), 0.5),
    RandomRotation((0, 90, 180, 270)),
    RandomCrop((256, 256), keep_size=True),
])
augmented = pipeline(result, seed=1234)
```

See [`docs/AUGMENTATION.md`](docs/AUGMENTATION.md) for exactness, cropping,
provenance, 3D rotations, and validation semantics.
