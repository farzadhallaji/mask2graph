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

Pipeline: `binary mask -> skeleton -> node/edge tracing -> normalized graph`.

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
