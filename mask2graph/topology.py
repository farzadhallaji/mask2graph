"""Topology diagnostics for raw, junction-cleaned, logical, and embedded graphs."""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
from numpy.typing import NDArray

from .types import EmbeddedGraph, Mask2Graph, TopologyDiagnostics
from .utils.connectivity import iter_neighbors, label_components


def _raw_edge_count(mask: NDArray[np.bool_]) -> int:
    shape = tuple(int(v) for v in mask.shape)
    count = 0
    for row in np.argwhere(mask):
        a = tuple(int(v) for v in row)
        for b in iter_neighbors(a, shape):
            if mask[b] and a < b:
                count += 1
    return count


def _graph_beta(node_ids: list[int], pairs: list[tuple[int, int]]) -> tuple[int, int]:
    if not node_ids:
        return 0, 0
    adj: dict[int, set[int]] = defaultdict(set)
    self_loops = 0
    for u, v in pairs:
        if u == v:
            self_loops += 1
        else:
            adj[u].add(v)
            adj[v].add(u)
    seen: set[int] = set()
    beta0 = 0
    for start in sorted(node_ids):
        if start in seen:
            continue
        beta0 += 1
        q = deque([start])
        seen.add(start)
        while q:
            cur = q.popleft()
            for nxt in adj.get(cur, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
    beta1 = len(pairs) - len(node_ids) + beta0
    # Formula already counts self-loops as edges and is valid for multigraphs.
    return int(beta0), int(beta1)


def build_topology_diagnostics(
    skeleton: NDArray[np.bool_],
    graph: Mask2Graph,
    *,
    expected_trace_segments: int,
    covered_trace_segments: int,
    junction_mst_edge_count: int | None = None,
) -> TopologyDiagnostics:
    raw_v = int(np.count_nonzero(skeleton))
    raw_e = _raw_edge_count(skeleton)
    _, raw_beta0 = label_components(skeleton)
    raw_beta1 = raw_e - raw_v + raw_beta0

    # "cleaned" means the topology after deterministic junction-zone cleanup
    # and digital micro-cycle absorption, before optional graph-side pruning.
    # That object is exactly the raw logical graph produced by the tracer.
    logical_beta0, logical_beta1 = _graph_beta(
        [n.id for n in graph.nodes], [(e.u, e.v) for e in graph.edges]
    )
    cleaned_e = len(graph.edges)
    cleaned_beta0 = logical_beta0
    cleaned_beta1 = logical_beta1
    return TopologyDiagnostics(
        raw_skeleton_vertices=raw_v,
        raw_skeleton_edges=raw_e,
        raw_beta0=int(raw_beta0),
        raw_beta1=int(raw_beta1),
        cleaned_skeleton_edges=cleaned_e,
        cleaned_beta0=cleaned_beta0,
        cleaned_beta1=cleaned_beta1,
        logical_vertices=len(graph.nodes),
        logical_edges=len(graph.edges),
        logical_beta0=logical_beta0,
        logical_beta1=logical_beta1,
        expected_trace_segments=int(expected_trace_segments),
        covered_trace_segments=int(covered_trace_segments),
        coverage_complete=expected_trace_segments == covered_trace_segments,
    )


def update_logical_diagnostics(diag: TopologyDiagnostics, graph: Mask2Graph) -> None:
    beta0, beta1 = _graph_beta(
        [n.id for n in graph.nodes], [(e.u, e.v) for e in graph.edges]
    )
    diag.logical_vertices = len(graph.nodes)
    diag.logical_edges = len(graph.edges)
    diag.logical_beta0 = beta0
    diag.logical_beta1 = beta1


def update_embedded_diagnostics(diag: TopologyDiagnostics, embedded: EmbeddedGraph) -> None:
    beta0, beta1 = _graph_beta(
        [v.id for v in embedded.vertices], [(s.u, s.v) for s in embedded.segments]
    )
    diag.embedded_vertices = len(embedded.vertices)
    diag.embedded_edges = len(embedded.segments)
    diag.embedded_beta0 = beta0
    diag.embedded_beta1 = beta1
