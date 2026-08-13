"""Shared helpers for the Parquet decompression experiment."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from experiments.parquet_decompression import EXPERIMENT_NAME, SCHEMA_VERSION


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def dataset_id(config: dict[str, Any]) -> str:
    digest = hashlib.sha256(canonical_json(config).encode()).hexdigest()[:12]
    return (
        f"{config['profile']}-{config['codec']}-"
        f"r{config['rows']}-rg{config['row_group_rows']}-{digest}"
    )


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_new_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = read_json(path)
    if manifest.get("experiment") != EXPERIMENT_NAME:
        raise ValueError(f"Unexpected experiment in {path}: {manifest.get('experiment')!r}")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported manifest schema in {path}")
    if not manifest.get("files"):
        raise ValueError(f"Manifest has no Parquet files: {path}")
    return manifest


def scalar(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def compare_signatures(
    expected: Any,
    actual: Any,
    *,
    path: str = "signature",
    relative_tolerance: float = 1e-9,
    absolute_tolerance: float = 1e-7,
) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise AssertionError(f"{path}: key mismatch")
        for key in expected:
            compare_signatures(
                expected[key],
                actual[key],
                path=f"{path}.{key}",
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise AssertionError(f"{path}: list mismatch")
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            compare_signatures(
                left,
                right,
                path=f"{path}[{index}]",
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )
        return
    if isinstance(expected, float) and isinstance(actual, (int, float)):
        if not math.isclose(
            expected,
            float(actual),
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        ):
            raise AssertionError(f"{path}: expected {expected!r}, got {actual!r}")
        return
    if expected != actual:
        raise AssertionError(f"{path}: expected {expected!r}, got {actual!r}")
