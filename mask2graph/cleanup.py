"""Explicit graph-cleanup policy helpers.

Destructive cleanup is intentionally separate from extraction.  The existing
``NormalizeConfig`` fields remain the policy surface; this module exposes a
small predicate used by callers and documentation to distinguish topology edits
from geometry-only simplification.
"""

from __future__ import annotations

from .config import NormalizeConfig


def topology_edit_enabled(config: NormalizeConfig) -> bool:
    return any(
        value > 0
        for value in (
            config.min_component_length,
            config.prune_spurs_below,
            config.min_cycle_length,
            config.max_cycle_area,
            config.cycle_length_to_radius_ratio,
            config.contract_short_edges_below,
        )
    )
