# Unified 2D/3D mask-to-PSLG pipeline

The production path is:

```
binary mask/volume
  -> conservative optional mask cleanup
  -> topology-preserving skeletonization
  -> full-neighbour pixel/voxel graph
  -> supported junction clustering + deterministic MST provenance
  -> maximal branch tracing with exact coverage invariant
  -> topology graph with complete original branch samples
  -> optional explicit graph cleanup policy
  -> spacing-aware radius/arclength/tangent/angle/curvature profiles
  -> non-destructive branch simplification
  -> validated straight-line embedded graph
  -> exact-rational min_ipd instance
```

## Trust boundary

With destructive graph cleanup thresholds left at zero, extraction requires every skeleton adjacency outside a contracted junction support to be owned by exactly one traced branch. Digital micro-cycles that collapse to a two-point `junction -> sample -> junction` walk are absorbed into the junction cleanup rather than emitted as degenerate self-loops.

Junction clusters use an actual skeleton lattice sample as their public coordinate. A deterministic MST is retained as support provenance, so no floating centroid is invented as a graph vertex.

`TopologyDiagnostics` records raw digital topology and the junction-cleaned topology separately. The raw full-neighbour pixel graph can contain artificial cycles inside thick junctions; correctness comparisons therefore use the junction-cleaned graph.

## Topology graph versus embedded graph

`Mask2Graph` is the topology graph. Its edges retain complete `path_index` and `path_xyz` arrays from the skeleton. Degree-two image samples are geometric provenance, not logical graph vertices.

`EmbeddedGraph` is the straight-line expansion used for PSLG/min_ipd export. It contains logical topology nodes plus only the geometric support points selected by simplification.

## Simplification

The default `optimal` simplifier builds a DAG over the ordered branch samples. An arc `i -> j` is admissible when the subpath stays within `epsilon` of its chord and the shortcut does not erase a protected high-turn sample. Shortest path in the unit-cost DAG yields the minimum number of straight segments under that admissibility model.

The original branch path is never modified. RDP remains available as a fast baseline. After simplification the complete embedded graph is checked globally. In 2D, undeclared intersections/overlaps are rejected; in 3D, true geometric intersections, overlaps, duplicate/zero segments and node-on-unrelated-segment degeneracies are rejected. If requested, simplification falls back to original branch samples when only the simplified embedding is invalid.

## Cleanup policy

Component deletion, spur pruning, tiny-cycle removal and short internal-edge contraction are explicit topology-editing policies and are disabled by zero thresholds in the default scientific profile. Geometric simplification is separate and is required not to alter graph topology.

## Boundary endpoints

Skeleton endpoints on the mask/volume boundary are retained as `boundary_endpoint` nodes with `on_image_boundary=True` and `boundary_axes` metadata. Cropping never silently deletes such endpoints.

## Exact min_ipd export

Use:

```python
from mask2graph import ExtractConfig, mask_to_graph, graph_to_min_ipd

cfg = ExtractConfig()
result = mask_to_graph(mask, spacing=(1.0, 1.0), config=cfg)
instance = graph_to_min_ipd(result.embedded_graph, config=cfg.export)
```

2D emits `min_ipd_instance_v1`; 3D emits `min_ipd_instance3_v1`. If no domain is supplied, a convex axis-aligned rational box is generated with a configurable margin. `lattice_exact` is appropriate for unit lattice coordinates, `scaled_exact` preserves decimal physical spacing as rational text, and `rationalized` uses a bounded denominator.

When `min_ipd` is installed, `validate_min_ipd_export(...)` calls its authoritative instance parser. Otherwise it performs local schema sanity checks.

## Determinism

Neighbour iteration, component ordering, MST tie-breaking, cycle anchors, edge orientation, simplifier tie-breaking and IDs are deterministic. The metadata stores input, processed-mask, configuration and graph hashes. Identical mask + spacing + config must serialize identically.

## Cropping

`crop_graph_box`, `crop_graph_radius`, and `crop_graph_connected_subgraph` return deterministic whole-edge topology subgraphs. They do not geometrically clip an edge halfway through a branch.
