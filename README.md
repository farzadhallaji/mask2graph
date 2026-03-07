# maskgraph (alpha)

Deterministically convert 2D/3D binary masks into topology-preserving geometric graphs (nodes, edges, radii, and metadata) for downstream analysis.

`maskgraph` is currently **alpha**: it is usable and tested on representative synthetic and real masks, but APIs/config defaults may still evolve.

## Why this exists

Most mask-to-graph pipelines lose loops, behave inconsistently at dense junctions, or return unstable output ordering.  
`maskgraph` focuses on:

- deterministic output (stable IDs/order/JSON bytes),
- loop-preserving extraction,
- one API for both 2D and 3D masks,
- optional debug artifacts for inspection.

## Visual pipeline

Pipeline: `binary mask -> conservative cleanup -> skeleton -> junction stabilization -> node/edge tracing -> iterative graph normalization`.

Real-mask overlay notebook (Mass Roads): `mass_roads_graph_overlay.ipynb`.

## Install

From source:

```bash
python -m pip install -e .
```

From GitHub:

```bash
python -m pip install "git+https://github.com/farzadhallaji/mask2graph.git"
```

## 10-line example

```python
import numpy as np
from maskgraph import extract_graph

mask = np.zeros((11, 11), dtype=np.uint8)
mask[5, 2:9] = 1
mask[2:6, 5] = 1

graph = extract_graph(mask, spacing=(1.0, 1.0))
print(len(graph.nodes), len(graph.edges))
print(graph.nodes[0].type, graph.edges[0].length)
```

## API quickstart

```python
from maskgraph import ExtractConfig, extract_graph, to_json

cfg = ExtractConfig()
graph = extract_graph(mask, config=cfg, spacing=(1.0, 1.0))
payload = to_json(graph)
```

## Conservative cleanup

Cleanup is optional and topology-conservative by default. The default cleanup config is effectively no-op unless you opt in with thresholds.

- removes tiny disconnected foreground components (`min_object_size`)
- fills tiny enclosed background holes (`max_hole_size` and `max_hole_radius`)
- uses full connectivity (8-neighbor in 2D, 26-neighbor in 3D)
- uses physical units when `spacing` is provided

Cleanup does not do branch separation, hole carving, global erosion/dilation, or aggressive morphology.
Those are topology-editing operations and should be explicit downstream choices.

```python
from maskgraph import ExtractConfig

cfg = ExtractConfig()
cfg.cleanup.min_object_size = 4.0
cfg.cleanup.max_hole_size = 16.0
cfg.cleanup.max_hole_radius = 1.5
```

When `return_debug=True`, `DebugArtifacts.cleanup_report` provides counts and physical sizes/radii for auditable cleanup changes.

## Graph-side artifact cleanup

Residual topology artifacts are handled after tracing at the graph stage, not by aggressive mask morphology:

- stabilizes junction neighborhoods by region-based node candidates (`junction_dilation_iters`)
- prunes short spurs (`prune_spurs_below`)
- removes tiny cycles (`min_cycle_length`, `max_cycle_area`, `cycle_length_to_radius_ratio`)
- contracts short internal edges around unstable junctions (`contract_short_edges_below`)
- iterates normalization until stable or `normalization_max_iter` is reached

Default normalization values are conservative (mostly disabled) so behavior remains predictable unless users opt in.

```python
from maskgraph import ExtractConfig

cfg = ExtractConfig()
cfg.normalize.junction_dilation_iters = 1
cfg.normalize.prune_spurs_below = 8.0
cfg.normalize.min_cycle_length = 12.0
cfg.normalize.max_cycle_area = 16.0
cfg.normalize.cycle_length_to_radius_ratio = 8.0
cfg.normalize.contract_short_edges_below = 2.0
cfg.normalize.normalization_max_iter = 8
```

This split is intentional:

- mask cleanup repairs likely annotation noise
- graph cleanup removes residual skeletonization/tracing artifacts
- topology-editing operators (carving/separation/opening/closing) are not part of default cleanup behavior

## Output schema

Serialized payload (`to_json`) has:

- `schema_version` (`"1"`),
- `meta`: version, ndim, shape, spacing, config snapshot, input hashes,
- `nodes[]`: `id`, `xyz`, `index`, `type`, `degree`, `voxel_count`, optional radii,
- `edges[]`: `id`, `u`, `v`, `path_xyz`, `path_index`, `length`, `voxel_length`, optional radii/profile, `is_self_loop`.

See `maskgraph/types.py` and `maskgraph/serialize.py` for exact fields.

## Examples

- `examples/example_2d_line_network.py`
- `examples/example_2d_loop.py`
- `examples/example_3d_branch.py`

## Benchmarks

Run:

```bash
python benchmarks/benchmark_extract.py
```

The benchmark currently reports 2D and 3D runtime/memory plus graph sizes on synthetic masks.

## Test status

Run:

```bash
python -m pytest -q
```

CI workflow setup is prepared, but GitHub Actions is not yet enabled in this repository push because the publishing token did not include `workflow` scope.

## Limitations (alpha)

- focused on thin-structure masks (roads, vessels, neurites, similar morphology),
- not yet benchmarked across broad public datasets with published error bars,
- no dedicated CLI configuration surface yet (CLI uses defaults),
- performance optimization for very large 3D volumes is still early-stage.

## Roadmap

- richer CLI/config overrides,
- broader dataset regression suite (2D/3D),
- algorithmic/parameter docs with recommended presets,
- improved large-volume performance and memory behavior.

## Contributing

Please read `CONTRIBUTING.md`. Bug reports with small reproducible masks are especially useful.

## Citation

If this project helps your work, cite via `CITATION.cff`.

## License

MIT (`LICENSE`).
