"""Benchmark one deterministic spatial CPU/GPU case in a fresh process."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .common import (
    atomic_json,
    distance_signature,
    elapsed,
    environment,
    gpu_sync,
    memory_snapshot,
    pair_signature,
)
from .compat import initialize_single_gpu_context
from .geometry import (
    linestring_coordinates,
    make_points,
    polygon_rings,
    quadtree_max_size,
    quadtree_scale,
)


def median_trials(function: Callable[[], dict[str, Any]], warmups: int, trials: int) -> dict[str, Any]:
    for _ in range(warmups):
        function()
    measured = [function() for _ in range(trials)]
    timing_keys = sorted(key for key in measured[0] if key.endswith("_seconds"))
    result = {key: statistics.median(float(row[key]) for row in measured) for key in timing_keys}
    result["trial_timings"] = [{key: row[key] for key in timing_keys} for row in measured]
    for key, value in measured[-1].items():
        if key not in timing_keys:
            result[key] = value
    return result


def host_geometries(case: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    import shapely

    x, y = make_points(case["points"], case.get("seed", 20260813))
    if case["operation"] == "nearest":
        lines = np.asarray(
            [shapely.LineString(v) for v in linestring_coordinates(case["features"], case["vertices"])],
            dtype=object,
        )
        return x, y, lines
    if case["operation"] in {"direct_pip", "indexed_join"}:
        polygons = np.asarray(
            [
                shapely.Polygon(v)
                for v in polygon_rings(
                    case["features"], case["vertices"], case.get("radius_fraction", 0.38)
                )
            ],
            dtype=object,
        )
        return x, y, polygons
    return x, y, None


def cpu_window(x: np.ndarray, y: np.ndarray, case: dict[str, Any]) -> dict[str, Any]:
    low, high = case["window"]

    def trial() -> dict[str, Any]:
        with elapsed() as timing:
            mask = (x > low) & (x < high) & (y > low) & (y < high)
        with elapsed() as materialize:
            selected_x = x[mask]
            selected_y = y[mask]
        return {
            "query_seconds": timing["seconds"],
            "materialize_seconds": materialize["seconds"],
            "matches": int(mask.sum()),
            "checksum": float(selected_x.sum() + selected_y.sum()),
        }

    result = median_trials(trial, case["warmups"], case["trials"])
    result["memory"] = memory_snapshot()
    return result


def gpu_points(x: np.ndarray, y: np.ndarray):
    import cudf
    import cupy as cp
    import cuspatial

    with elapsed(gpu_sync) as upload:
        xy = cudf.DataFrame({"x": cp.asarray(x), "y": cp.asarray(y)}).interleave_columns()
        points = cuspatial.GeoSeries.from_points_xy(xy)
    return points, upload["seconds"]


def gpu_window(x: np.ndarray, y: np.ndarray, case: dict[str, Any]) -> dict[str, Any]:
    import cuspatial

    points, upload_seconds = gpu_points(x, y)
    low, high = case["window"]

    def trial() -> dict[str, Any]:
        with elapsed(gpu_sync) as timing:
            selected = cuspatial.points_in_spatial_window(points, low, high, low, high)
        with elapsed(gpu_sync) as materialize:
            checksum = float((selected.points.x.sum() + selected.points.y.sum()))
        return {
            "query_seconds": timing["seconds"],
            "materialize_seconds": materialize["seconds"],
            "matches": len(selected),
            "checksum": checksum,
        }

    result = median_trials(trial, case["warmups"], case["trials"])
    result["ingest_seconds"] = upload_seconds
    result["memory"] = memory_snapshot()
    return result


def cpu_spatial(x: np.ndarray, y: np.ndarray, features: np.ndarray, case: dict[str, Any]) -> dict[str, Any]:
    import shapely
    from shapely import STRtree

    workers = int(case.get("cpu_workers", 8))
    if case["operation"] == "direct_pip":
        after_ingest = memory_snapshot()
        def trial() -> dict[str, Any]:
            with elapsed() as query:
                matrix = shapely.contains_xy(features[:, None], x[None, :], y[None, :])
            with elapsed() as materialize:
                feature_ids, point_ids = np.nonzero(matrix)
            return {
                "query_seconds": query["seconds"],
                "materialize_seconds": materialize["seconds"],
                "matches": len(point_ids),
                "pair_signature": pair_signature(point_ids, feature_ids),
            }
        result = median_trials(trial, case["warmups"], case["trials"])
        result.update({"ingest_seconds": 0.0, "index_build_seconds": 0.0})
        result["cpu_workers"] = 1
        result["memory"] = memory_snapshot()
        result["memory_after_ingest"] = after_ingest
        return result

    with elapsed() as ingest:
        points = shapely.points(x, y)
    after_ingest = memory_snapshot()
    edges = np.linspace(0, len(points), workers + 1, dtype=np.int64)
    point_chunks = [
        (int(start), int(stop)) for start, stop in zip(edges[:-1], edges[1:]) if stop > start
    ]
    executor = ThreadPoolExecutor(max_workers=workers)
    if case["operation"] == "indexed_join":
        with elapsed() as build:
            tree = STRtree(features)
        after_index = memory_snapshot()
        def trial() -> dict[str, Any]:
            def candidates_for(bounds: tuple[int, int]) -> np.ndarray:
                start, stop = bounds
                pairs = tree.query(points[start:stop])
                pairs[0] += start
                return pairs

            with elapsed() as candidates_time:
                candidate_parts = list(executor.map(candidates_for, point_chunks))

            def exact_for(candidate_pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                keep = shapely.within(
                    points[candidate_pairs[0]], features[candidate_pairs[1]]
                )
                return candidate_pairs[0, keep], candidate_pairs[1, keep]

            with elapsed() as exact_time:
                exact_parts = list(executor.map(exact_for, candidate_parts))
            with elapsed() as materialize:
                point_ids = np.concatenate([part[0] for part in exact_parts])
                feature_ids = np.concatenate([part[1] for part in exact_parts])
            return {
                "candidate_seconds": candidates_time["seconds"],
                "exact_seconds": exact_time["seconds"],
                "query_seconds": candidates_time["seconds"] + exact_time["seconds"],
                "materialize_seconds": materialize["seconds"],
                "candidate_pairs": sum(part.shape[1] for part in candidate_parts),
                "matches": len(point_ids),
                "pair_signature": pair_signature(point_ids, feature_ids),
            }
        result = median_trials(trial, case["warmups"], case["trials"])
        result.update({"ingest_seconds": ingest["seconds"], "index_build_seconds": build["seconds"]})
    else:
        with elapsed() as build:
            tree = STRtree(features)
        after_index = memory_snapshot()
        def trial() -> dict[str, Any]:
            def nearest_for(bounds: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
                start, stop = bounds
                indices, distances = tree.query_nearest(
                    points[start:stop], all_matches=False, return_distance=True
                )
                return indices[0] + start, distances

            with elapsed() as query:
                nearest_parts = list(executor.map(nearest_for, point_chunks))
            with elapsed() as materialize:
                point_ids = np.concatenate([part[0] for part in nearest_parts])
                distances = np.concatenate([part[1] for part in nearest_parts])
            return {
                "query_seconds": query["seconds"],
                "materialize_seconds": materialize["seconds"],
                "matches": len(distances),
                "distance_signature": distance_signature(point_ids, distances),
                "distance_sum": float(distances.sum()),
                "_validation_point_ids": point_ids,
                "_validation_distances": distances,
            }
        result = median_trials(trial, case["warmups"], case["trials"])
        result.update({"ingest_seconds": ingest["seconds"], "index_build_seconds": build["seconds"]})
    result["memory"] = memory_snapshot()
    result["memory_after_ingest"] = after_ingest
    result["memory_after_index"] = after_index
    result["cpu_workers"] = workers
    executor.shutdown()
    del points
    gc.collect()
    return result


def gpu_spatial(x: np.ndarray, y: np.ndarray, features: np.ndarray, case: dict[str, Any]) -> dict[str, Any]:
    import cudf
    import cuspatial
    import geopandas as gpd

    points, point_ingest = gpu_points(x, y)
    with elapsed(gpu_sync) as feature_ingest:
        gpu_features = cuspatial.from_geopandas(gpd.GeoSeries(features))
    after_ingest = memory_snapshot()
    if case["operation"] == "direct_pip":
        def trial() -> dict[str, Any]:
            with elapsed(gpu_sync) as query:
                matrix = cuspatial.point_in_polygon(points, gpu_features)
            with elapsed(gpu_sync) as materialize:
                host = matrix.to_pandas().to_numpy()
                point_ids, feature_ids = np.nonzero(host)
            return {
                "query_seconds": query["seconds"],
                "materialize_seconds": materialize["seconds"],
                "matches": len(point_ids),
                "pair_signature": pair_signature(point_ids, feature_ids),
            }
        result = median_trials(trial, case["warmups"], case["trials"])
        result["index_build_seconds"] = 0.0
    else:
        depth = case.get("max_depth", 15)
        scale = quadtree_scale(depth)
        max_size = case.get("max_size") or quadtree_max_size(len(points))
        with elapsed(gpu_sync) as build:
            point_indices, quadtree = cuspatial.quadtree_on_points(
                points, 0.0, 1.0, 0.0, 1.0, scale, depth, max_size
            )
        leaf_point_count = int(
            quadtree.loc[~quadtree["is_internal_node"], "length"].sum()
        )
        if len(point_indices) != len(points) or leaf_point_count != len(points):
            raise AssertionError(
                "quadtree does not account for every input point: "
                f"indices={len(point_indices)}, leaves={leaf_point_count}, "
                f"points={len(points)}, max_size={max_size}, depth={depth}"
            )
        after_index = memory_snapshot()
        if case["operation"] == "indexed_join":
            boxes = cuspatial.polygon_bounding_boxes(gpu_features)
            def trial() -> dict[str, Any]:
                with elapsed(gpu_sync) as candidates_time:
                    quad_pairs = cuspatial.join_quadtree_and_bounding_boxes(
                        quadtree, boxes, 0.0, 1.0, 0.0, 1.0, scale, depth
                    )
                candidate_point_tests = int(
                    quadtree.iloc[quad_pairs["quad_offset"]]["length"].sum()
                )
                with elapsed(gpu_sync) as exact_time:
                    matches = cuspatial.quadtree_point_in_polygon(
                        quad_pairs, quadtree, point_indices, points, gpu_features
                    )
                with elapsed(gpu_sync) as materialize:
                    host = matches.to_pandas()
                    original = point_indices.iloc[
                        cudf.Series(host["point_index"].to_numpy())
                    ].to_pandas().to_numpy()
                    feature_ids = host["polygon_index"].to_numpy()
                return {
                    "candidate_seconds": candidates_time["seconds"],
                    "exact_seconds": exact_time["seconds"],
                    "materialize_seconds": materialize["seconds"],
                    "query_seconds": candidates_time["seconds"] + exact_time["seconds"],
                    "candidate_quadrants": len(quad_pairs),
                    "candidate_pairs": candidate_point_tests,
                    "matches": len(host),
                    "pair_signature": pair_signature(original, feature_ids),
                }
        else:
            radius = case.get("expansion_radius", 0.75 / case["features"])
            boxes = cuspatial.linestring_bounding_boxes(gpu_features, radius)
            def trial() -> dict[str, Any]:
                with elapsed(gpu_sync) as candidates_time:
                    quad_pairs = cuspatial.join_quadtree_and_bounding_boxes(
                        quadtree, boxes, 0.0, 1.0, 0.0, 1.0, scale, depth
                    )
                candidate_point_tests = int(
                    quadtree.iloc[quad_pairs["quad_offset"]]["length"].sum()
                )
                with elapsed(gpu_sync) as exact_time:
                    nearest = cuspatial.quadtree_point_to_nearest_linestring(
                        quad_pairs, quadtree, point_indices, points, gpu_features
                    )
                with elapsed(gpu_sync) as materialize:
                    host = nearest.to_pandas()
                    original = point_indices.iloc[
                        cudf.Series(host.iloc[:, 0].to_numpy())
                    ].to_pandas().to_numpy()
                    distances = host.iloc[:, 2].to_numpy()
                return {
                    "candidate_seconds": candidates_time["seconds"],
                    "exact_seconds": exact_time["seconds"],
                    "materialize_seconds": materialize["seconds"],
                    "query_seconds": candidates_time["seconds"] + exact_time["seconds"],
                    "candidate_quadrants": len(quad_pairs),
                    "candidate_pairs": candidate_point_tests,
                    "matches": len(host),
                    "distance_signature": distance_signature(original, distances),
                    "distance_sum": float(distances.sum()),
                    "_validation_point_ids": original,
                    "_validation_distances": distances,
                }
        result = median_trials(trial, case["warmups"], case["trials"])
        result["index_build_seconds"] = build["seconds"]
        result["quadtree_max_size"] = max_size
        result["quadtree_leaf_point_count"] = leaf_point_count
    result["ingest_seconds"] = point_ingest + feature_ingest["seconds"]
    result["memory"] = memory_snapshot()
    result["memory_after_ingest"] = after_ingest
    if case["operation"] != "direct_pip":
        result["memory_after_index"] = after_index
    return result


def validate(cpu: dict[str, Any], gpu: dict[str, Any], operation: str) -> dict[str, Any]:
    if cpu["matches"] != gpu["matches"]:
        raise AssertionError(f"match counts differ: CPU={cpu['matches']} GPU={gpu['matches']}")
    if operation in {"direct_pip", "indexed_join"}:
        if cpu["pair_signature"] != gpu["pair_signature"]:
            raise AssertionError("CPU/GPU spatial join pairs differ")
        return {"matches_equal": True, "pair_signatures_equal": True}
    if operation == "nearest":
        relative = abs(cpu["distance_sum"] - gpu["distance_sum"]) / max(abs(cpu["distance_sum"]), 1.0)
        cpu_ids = np.asarray(cpu.pop("_validation_point_ids"))
        gpu_ids = np.asarray(gpu.pop("_validation_point_ids"))
        cpu_distances = np.asarray(cpu.pop("_validation_distances"))
        gpu_distances = np.asarray(gpu.pop("_validation_distances"))
        row_count = len(cpu_ids)
        if (
            np.any(cpu_ids < 0)
            or np.any(cpu_ids >= row_count)
            or np.any(gpu_ids < 0)
            or np.any(gpu_ids >= row_count)
        ):
            raise AssertionError("nearest-distance point ID is outside the input range")
        cpu_seen = np.zeros(row_count, dtype=np.bool_)
        gpu_seen = np.zeros(row_count, dtype=np.bool_)
        cpu_seen[cpu_ids] = True
        gpu_seen[gpu_ids] = True
        if not cpu_seen.all() or not gpu_seen.all():
            raise AssertionError("nearest-distance point IDs are not a complete unique covering")
        cpu_aligned = np.empty(row_count, dtype=np.float64)
        gpu_aligned = np.empty(row_count, dtype=np.float64)
        cpu_aligned[cpu_ids] = cpu_distances
        gpu_aligned[gpu_ids] = gpu_distances
        absolute_errors = np.abs(cpu_aligned - gpu_aligned)
        maximum_absolute = float(absolute_errors.max(initial=0.0))
        maximum_relative = float(
            np.max(
                absolute_errors
                / np.maximum(np.abs(cpu_aligned), np.finfo(np.float64).eps),
                initial=0.0,
            )
        )
        if relative > 1e-10 or not np.allclose(
            cpu_aligned, gpu_aligned, rtol=1e-10, atol=1e-10
        ):
            raise AssertionError(
                "nearest distances differ: "
                f"sum_relative={relative}, max_absolute={maximum_absolute}, "
                f"max_relative={maximum_relative}"
            )
        return {
            "matches_equal": True,
            "distance_sum_relative_error": relative,
            "maximum_absolute_distance_error": maximum_absolute,
            "maximum_relative_distance_error": maximum_relative,
        }
    relative = abs(cpu["checksum"] - gpu["checksum"]) / max(abs(cpu["checksum"]), 1.0)
    if relative > 1e-10:
        raise AssertionError(f"window checksums differ by {relative}")
    return {"matches_equal": True, "checksum_relative_error": relative}


def run(case: dict[str, Any]) -> dict[str, Any]:
    initialize_single_gpu_context()
    x, y, features = host_geometries(case)
    if case["operation"] == "window":
        cpu = cpu_window(x, y, case)
        gpu = gpu_window(x, y, case)
    else:
        assert features is not None
        cpu = cpu_spatial(x, y, features, case)
        gpu = gpu_spatial(x, y, features, case)
    return {
        "schema_version": 1,
        "case": case,
        "environment": environment(),
        "cpu": cpu,
        "gpu": gpu,
        "correctness": validate(cpu, gpu, case["operation"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, help="JSON object")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_json(args.output, run(json.loads(args.case)))


if __name__ == "__main__":
    main()
