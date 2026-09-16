"""Public package exports for mask2graph."""

from ._version import __version__

from .api import (
    compute_degree_map,
    estimate_radii,
    extract_graph,
    extract_raw_graph,
    mask_to_graph,
    normalize_graph,
    preprocess_mask,
    skeletonize_mask,
)
from .augment import (
    GraphAugmentationPipeline,
    RandomCrop,
    RandomFlip,
    RandomRotation,
    augment_graph,
    augment_result,
)
from .config import (
    CleanupConfig,
    DeterminismConfig,
    ExportConfig,
    ExtractConfig,
    GeometryConfig,
    JunctionConfig,
    NormalizeConfig,
    SimplifyConfig,
    SkeletonConfig,
    ValidationConfig,
)
from .crop import crop_graph, crop_graph_box, crop_graph_connected_subgraph, crop_graph_radius
from .networkx_utils import to_networkx
from .paper_config import (
    AugmentationRunConfig,
    ConfigError,
    InputRunConfig,
    OutputRunConfig,
    PaperRunConfig,
    VisualizationRunConfig,
    dump_resolved_yaml,
    load_run_config,
)
from .runner import PaperRunResult, RunStage, execute_run, load_configured_mask, run_experiment
from .pslg import to_min_ipd, validate_min_ipd_export
from .serialize import from_json, result_from_dict, result_from_json, result_to_dict, result_to_json, to_json
from .transforms import (
    GraphTransform,
    affine_transform,
    apply_transform,
    axis_permutation_transform,
    compose_transforms,
    flip_graph,
    flip_transform,
    graph_is_grid_aligned,
    permute_axes,
    resolve_center,
    rotate_graph,
    rotation_transform,
    translate_graph,
    translation_transform,
)
from .types import (
    DebugArtifacts,
    Edge,
    EmbeddedGraph,
    EmbeddedSegment,
    EmbeddedVertex,
    ExtractionError,
    GraphMeta,
    Mask2Graph,
    MaskGraphResult,
    Node,
    SerializationError,
    TopologyDiagnostics,
    ValidationError,
)
from .validate import validate_embedded_graph, validate_skeleton_coverage, validate_topology
from .visualize import (
    central_crop_bounds,
    graph_bounds,
    plot_augmentation_sequence,
    plot_graph,
    plot_mask,
    plot_result,
)


def mask_to_min_ipd(mask, *, spacing=None, config=None, domain=None, name="mask2graph"):
    """Convenience mask -> validated embedded graph -> min_ipd dictionary."""
    result = mask_to_graph(mask, spacing=spacing, config=config)
    cfg = config or ExtractConfig()
    instance = to_min_ipd(result.embedded_graph, domain=domain, config=cfg.export, name=name)
    validate_min_ipd_export(instance)
    return instance


def graph_to_min_ipd(embedded_graph, *, domain=None, config=None, name="mask2graph"):
    """Named adapter matching the mask->graph->min_ipd pipeline terminology."""
    return to_min_ipd(embedded_graph, domain=domain, config=config, name=name)


__all__ = [
    "__version__", "CleanupConfig", "DebugArtifacts", "DeterminismConfig", "Edge", "EmbeddedGraph",
    "EmbeddedSegment", "EmbeddedVertex", "ExportConfig", "ExtractConfig", "ExtractionError",
    "GeometryConfig", "GraphAugmentationPipeline", "GraphMeta", "GraphTransform", "JunctionConfig",
    "Mask2Graph", "MaskGraphResult", "Node", "NormalizeConfig", "RandomCrop", "RandomFlip",
    "RandomRotation", "SerializationError", "SimplifyConfig", "SkeletonConfig", "TopologyDiagnostics",
    "ValidationConfig", "ValidationError", "affine_transform", "apply_transform", "augment_graph",
    "augment_result", "axis_permutation_transform", "compose_transforms", "compute_degree_map",
    "crop_graph", "crop_graph_box", "crop_graph_connected_subgraph", "crop_graph_radius", "estimate_radii",
    "extract_graph", "extract_raw_graph", "flip_graph", "flip_transform", "from_json", "graph_is_grid_aligned",
    "graph_to_min_ipd", "mask_to_graph", "mask_to_min_ipd", "normalize_graph", "permute_axes",
    "preprocess_mask", "resolve_center", "result_from_dict", "result_from_json", "result_to_dict", "result_to_json", "rotate_graph",
    "rotation_transform", "skeletonize_mask", "to_json", "to_min_ipd", "to_networkx", "translate_graph",
    "translation_transform", "validate_embedded_graph", "validate_min_ipd_export",
    "validate_skeleton_coverage", "validate_topology", "central_crop_bounds", "graph_bounds",
    "plot_augmentation_sequence", "plot_graph", "plot_mask", "plot_result",
    "AugmentationRunConfig", "ConfigError", "InputRunConfig", "OutputRunConfig",
    "PaperRunConfig", "VisualizationRunConfig", "dump_resolved_yaml", "load_run_config",
    "PaperRunResult", "RunStage", "execute_run", "load_configured_mask", "run_experiment",
]
