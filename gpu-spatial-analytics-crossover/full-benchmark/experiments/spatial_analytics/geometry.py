"""Deterministic synthetic planar geometries with independently controlled axes."""

from __future__ import annotations

import math

import numpy as np


def make_points(rows: int, seed: int = 20260813) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.random(rows, dtype=np.float64), rng.random(rows, dtype=np.float64)


def polygon_rings(
    count: int,
    vertices: int,
    radius_fraction: float = 0.38,
    wobble: float = 0.12,
) -> list[np.ndarray]:
    """Create disjoint, valid rings in a near-square grid over [0, 1]^2."""
    if count < 1 or vertices < 4:
        raise ValueError("count must be positive and vertices must be at least four")
    side = math.ceil(math.sqrt(count))
    cell = 1.0 / side
    radius = radius_fraction * cell
    angles = np.linspace(0.0, 2.0 * np.pi, vertices, endpoint=False)
    rings: list[np.ndarray] = []
    for index in range(count):
        row, column = divmod(index, side)
        cx = (column + 0.5) * cell
        cy = (row + 0.5) * cell
        phase = index * 0.6180339887498948
        local_radius = radius * (1.0 + wobble * np.sin(3.0 * angles + phase))
        coordinates = np.column_stack(
            (cx + local_radius * np.cos(angles), cy + local_radius * np.sin(angles))
        )
        rings.append(np.vstack((coordinates, coordinates[0])))
    return rings


def linestring_coordinates(count: int, vertices: int) -> list[np.ndarray]:
    """Create non-intersecting, vertically distributed wavy lines."""
    if count < 1 or vertices < 2:
        raise ValueError("count must be positive and vertices must be at least two")
    y = np.linspace(0.0, 1.0, vertices)
    spacing = 1.0 / count
    amplitude = min(0.18 * spacing, 0.01)
    lines = []
    for index in range(count):
        center = (index + 0.5) * spacing
        x = center + amplitude * np.sin(2.0 * np.pi * y + index * 0.37)
        lines.append(np.column_stack((x, y)))
    return lines


def quadtree_scale(max_depth: int, span: float = 1.0) -> float:
    """Smallest documented Morton-cell scale, with a tiny numeric margin."""
    if max_depth < 1 or max_depth > 15:
        raise ValueError("cuSpatial quadtree depth must be in [1, 15]")
    return span / ((1 << max_depth) + 2) * 1.000001


def quadtree_max_size(rows: int) -> int:
    """Keep roughly 50K leaves while respecting cuSpatial's size floor."""
    if rows < 1:
        raise ValueError("rows must be positive")
    target = max(64, math.ceil(rows / 50_000))
    return min(4096, 1 << (target - 1).bit_length())
