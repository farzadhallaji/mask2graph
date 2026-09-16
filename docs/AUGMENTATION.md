# Graph-to-graph augmentation

`mask2graph` performs augmentation **after** mask extraction.  The augmentation
layer never rasterizes an extracted graph and never reskeletonizes it.

```text
mask -> mask_to_graph -> MaskGraphResult
                         |
                         +-- rigid transforms: rotation / flip / translation / axis permutation
                         +-- geometric box crop
                         v
                    MaskGraphResult -> graph_to_min_ipd
```

The central invariant is that augmentation must not silently corrupt topology,
provenance, or exactness.

## Rigid/isometric transforms

`rotate_graph`, `flip_graph`, `translate_graph`, `permute_axes`, and
`apply_transform` transform both the topology graph and straight embedded graph
with the same affine isometry.  They preserve graph incidence, `beta0`, `beta1`,
branch lengths, radii, chord lengths, tortuosity, turning angles, and curvature
magnitudes.  Tangent vectors are transformed by the same orthogonal matrix.

```python
from mask2graph import mask_to_graph, rotate_graph, flip_graph, translate_graph

result = mask_to_graph(mask)
result = rotate_graph(result, 90)
result = flip_graph(result, "x")
result = translate_graph(result, (32, 0))
```

### 2D rotations

`rotate_graph(result, angle, center="bbox_center")` interprets `angle` in
degrees by default.  `center` can be `bbox_center`, `centroid`, `origin`, or an
explicit `(x, y)` coordinate.

Multiples of 90 degrees use signed-permutation matrices and are marked exact.
An arbitrary angle is a floating geometric transform and is marked
`transform_exact=False`.

### 3D rotations

3D rotation avoids Euler-angle ambiguity.  Supply one of:

```python
rotate_graph(result, 90, axis=(0, 0, 1))
rotate_graph(result, quaternion=(w, x, y, z))
rotate_graph(result, matrix=R)
```

Coordinate-axis quarter turns are exact signed permutations.  General rotations
are not claimed exact.

### Flips and axis permutations

2D supports `x`, `y`, and `xy`.  3D supports any combination of `x`, `y`, `z`.

```python
flip_graph(result, "xyz")
permute_axes(result, "zxy")
```

## Raster-index provenance

The original source raster index is never fabricated after augmentation.

* `Node.source_index` is immutable provenance to the source mask sample.
* `Node.index` is a current raster index only when `Node.index_valid=True`.
* `Edge.source_path_index` preserves source-mask path provenance.
* `Edge.path_index` is current-grid geometry only when
  `Edge.path_index_valid=True`.
* `GraphMeta.grid_aligned` summarizes whether all transformed geometry lies on
  the configured lattice.

An arbitrary rotation therefore preserves source provenance but sets current
index validity false instead of rounding geometry back onto pixels.

## Exactness and min_ipd export

`GraphMeta.transform_exact` is true only while every applied transform belongs to
the exact rational-affine class used by this layer (for example signed axis
permutations/flips and translations).  A general trigonometric rotation sets it
false permanently for that augmentation chain.

When `ExportConfig.coordinate_mode="rationalized"`, `to_min_ipd` reports the
maximum coordinate rationalization error in output metadata.  Export never
silently turns an arbitrary rotation into an exact claim.

## Geometric box crop

Cropping is intentionally different from a rigid transform because it may change
topology.  `crop_graph_box` clips the **complete source branch polyline** stored in
`Edge.path_xyz`, not only the simplified straight representation.

```python
cropped = crop_graph_box(
    result,
    bounds=(xmin, xmax, ymin, ymax),
)

cropped3 = crop_graph_box(
    result3,
    bounds=(xmin, xmax, ymin, ymax, zmin, zmax),
)
```

A crossing branch is clipped geometrically.  New intersection vertices are
explicit `boundary_endpoint` nodes.  If the crop boundary intersects a segment
between raster samples, the new vertex has no fabricated source pixel index.
Radius values are linearly interpolated at the intersection.

One source branch can produce zero, one, or multiple crop fragments.  Each output
edge records:

* `source_edge_id`;
* `source_arc_range`;
* `crop_fragment_index`;
* `created_by="crop"`.

Pure cycles are recomputed after clipping: a contained cycle stays a cycle; a cut
cycle becomes one or more open paths.

Cropping does not prune short fragments.  Destructive cleanup remains a separate
policy layer.

### keep-size crop

`keep_size=True` is implemented as exactly two operations: geometric crop, then
translation of the crop lower corner to the origin.

```python
patch = crop_graph_box(result, bounds=(100, 356, 80, 336), keep_size=True)
```

The augmentation history therefore contains both `crop` and `translation`.

## Transform composition

`GraphTransform` represents `x' = A x + b`.

```python
T = compose_transforms([T1, T2, T3])  # T3(T2(T1(x)))
T = T3 @ T2 @ T1                      # same conventional composition order
out = apply_transform(result, T)
```

Exact transforms provide `T.inverse()`.

## Deterministic random augmentation

Randomness is confined to sampling wrappers.  Low-level geometric operators are
fully deterministic for explicit parameters.

```python
from mask2graph import (
    GraphAugmentationPipeline, RandomFlip, RandomRotation, RandomCrop,
)

pipeline = GraphAugmentationPipeline([
    RandomFlip(("x", "y"), probability=0.5),
    RandomRotation((0, 90, 180, 270)),
    RandomCrop((256, 256), keep_size=True),
])

augmented = pipeline(result, seed=1234)
```

The same source graph, configuration, and seed produce byte-identical serialized
results.  Metadata records the source graph hash, seed, sampled operations,
matrices/translations, crop bounds, exactness/alignment flags, and output graph
hash.

## Validation

Rigid transforms require unchanged vertex/edge incidence, `beta0`, `beta1`,
lengths and radius data.  The embedded graph is revalidated for undeclared
intersections/incidences.

Crop validation instead requires:

* every output point lies in the crop box;
* every vertex flagged on the current crop boundary lies on that boundary;
* every cropped edge has source-edge provenance;
* no invalid embedded intersections are introduced;
* embedded subdivision has the same topology as the newly cropped topology graph.

A crop is **not** required to preserve the source `beta0` or `beta1`; its
`TopologyDiagnostics` records topology before and after.

## V1 scope

V1 deliberately contains only rigid/isometric transforms and geometric crop.
Elastic deformation, nonlinear warps, and random node jitter are excluded because
they can create or destroy geometric incidences and require a different validation
model.

## Visual notebook walkthrough

`notebooks/mask_graph_augmentation_demo.ipynb` is the canonical visual walkthrough,
but behavior comes only from `configs/retinal_augmentation_demo.yaml`. The notebook
calls the same `run_experiment(...)` function as the CLI and displays the archived
figures for the source mask, extracted graph, rotation, flip, crop, and full sequence.
No angles, crop fractions, spacing, seed, simplification, or export policy are
duplicated in notebook cells. See `CONFIG_POLICY.md`.

The plotting layer remains optional and available through `mask2graph[viz]` /
`mask2graph[notebook]`; the core extractor has no matplotlib dependency.
