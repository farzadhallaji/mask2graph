# mask2graph

`mask2graph` converts 2D/3D binary masks into deterministic geometric graphs (nodes, edges, radii, and metadata) for downstream analysis, simulation, and learning systems.

## Why mask2graph

Most mask-to-graph pipelines fail at dense junctions, lose loops, or produce unstable IDs across runs. `mask2graph` is built for reproducibility and explicit topology handling:

- deterministic output ordering and JSON payloads
- loop/self-loop preservation
- one API for 2D and 3D masks
- spacing-aware cleanup and geometry
- auditable debug artifacts

## Pipeline

`binary mask -> conservative mask cleanup -> skeleton -> junction stabilization -> node/edge tracing -> iterative graph normalization`

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

Install optional NetworkX interoperability dependency:

```bash
python -m pip install -e ".[interop]"
```

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

`multigraph=True` preserves parallel edges and self-loops. Node and edge attributes include geometry and topology metadata from `MaskGraph`.

## Output schema

Serialized payload (`to_json`) includes:

- `schema_version`
- `meta`: version, ndim, shape, spacing, config snapshot, input hashes
- `nodes[]`: ids, geometry, topology labels, optional radii
- `edges[]`: connectivity, path geometry/index, lengths, optional radii/profile, loop flag

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
