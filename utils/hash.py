"""Deterministic hashing helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_array(mask: NDArray[np.bool_]) -> str:
    arr = np.ascontiguousarray(mask.astype(np.bool_, copy=False))
    return _sha256(arr.tobytes(order="C"))


def stable_config_json(config_dict: dict[str, Any]) -> str:
    return json.dumps(config_dict, sort_keys=True, separators=(",", ":"))


def hash_extraction_inputs(
    input_mask: NDArray[np.bool_],
    processed_mask: NDArray[np.bool_],
    config_dict: dict[str, Any],
    version: str,
) -> tuple[str, str, str]:
    input_hash = hash_array(input_mask)
    processed_hash = hash_array(processed_mask)
    config_payload = stable_config_json(
        {
            "config": config_dict,
            "input_hash": input_hash,
            "processed_hash": processed_hash,
            "version": version,
        }
    )
    combined_hash = _sha256(config_payload.encode("utf-8"))
    return input_hash, processed_hash, combined_hash
