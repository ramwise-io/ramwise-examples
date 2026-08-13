"""Shared helpers for the Spark RAPIDS fallback experiment."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def stable_id(prefix: str, value: Any, length: int = 12) -> str:
    digest = hashlib.sha256(canonical_json(value).encode()).hexdigest()[:length]
    return f"{prefix}-{digest}"


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_new_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def compare_values(
    expected: Any,
    actual: Any,
    *,
    path: str = "result",
    relative_tolerance: float = 1e-8,
    absolute_tolerance: float = 1e-6,
) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise AssertionError(f"{path}: key mismatch")
        for key in expected:
            compare_values(
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
            compare_values(
                left,
                right,
                path=f"{path}[{index}]",
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )
        return
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not math.isclose(
            float(expected),
            float(actual),
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        ):
            raise AssertionError(f"{path}: expected {expected!r}, got {actual!r}")
        return
    if expected != actual:
        raise AssertionError(f"{path}: expected {expected!r}, got {actual!r}")

