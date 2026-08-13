"""Validate cuSpatial nearest-linestring expansion radii against Shapely."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .compat import initialize_single_gpu_context
from .geometry import linestring_coordinates, make_points, quadtree_max_size, quadtree_scale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=10_000_000)
    parser.add_argument("--features", type=int, default=1024)
    parser.add_argument("--vertices", type=int, default=9)
    parser.add_argument("--multipliers", type=float, nargs="+", default=[0.75, 1.0, 1.5, 2.0])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import cudf
    import cupy as cp
    import cuspatial
    import geopandas as gpd
    import shapely
    from shapely import STRtree

    initialize_single_gpu_context()
    x, y = make_points(args.points)
    lines = np.asarray(
        [shapely.LineString(v) for v in linestring_coordinates(args.features, args.vertices)],
        dtype=object,
    )
    points = shapely.points(x, y)
    tree = STRtree(lines)
    edges = np.linspace(0, len(points), 9, dtype=np.int64)

    def nearest(bounds: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        start, stop = bounds
        indices, distances = tree.query_nearest(
            points[start:stop], all_matches=False, return_distance=True
        )
        return indices[0] + start, distances

    with ThreadPoolExecutor(max_workers=8) as executor:
        pieces = list(executor.map(nearest, zip(edges[:-1], edges[1:])))
    reference = np.empty(args.points, dtype=np.float64)
    for ids, distances in pieces:
        reference[ids] = distances

    xy = cudf.DataFrame({"x": cp.asarray(x), "y": cp.asarray(y)}).interleave_columns()
    gpu_points = cuspatial.GeoSeries.from_points_xy(xy)
    gpu_lines = cuspatial.from_geopandas(gpd.GeoSeries(lines))
    depth = 15
    scale = quadtree_scale(depth)
    point_indices, quadtree = cuspatial.quadtree_on_points(
        gpu_points, 0.0, 1.0, 0.0, 1.0, scale, depth, quadtree_max_size(args.points)
    )
    evidence = []
    for multiplier in args.multipliers:
        radius = multiplier / args.features
        boxes = cuspatial.linestring_bounding_boxes(gpu_lines, radius)
        pairs = cuspatial.join_quadtree_and_bounding_boxes(
            quadtree, boxes, 0.0, 1.0, 0.0, 1.0, scale, depth
        )
        result = cuspatial.quadtree_point_to_nearest_linestring(
            pairs, quadtree, point_indices, gpu_points, gpu_lines
        ).to_pandas()
        original = point_indices.iloc[cudf.Series(result.iloc[:, 0].to_numpy())].to_pandas().to_numpy()
        actual = np.empty(args.points, dtype=np.float64)
        actual[original] = result.iloc[:, 2].to_numpy()
        errors = np.abs(reference - actual)
        row = {
            "points": args.points,
            "features": args.features,
            "vertices": args.vertices,
            "multiplier": multiplier,
            "radius": radius,
            "result_rows": len(result),
            "quad_pairs": len(pairs),
            "maximum_absolute_error": float(errors.max()),
            "mismatches": int(
                np.count_nonzero(~np.isclose(reference, actual, rtol=1e-10, atol=1e-10))
            ),
        }
        evidence.append(row)
        print(
            f"multiplier={multiplier} radius={radius:.12g} rows={len(result)} "
            f"quad_pairs={len(pairs)} max_abs={errors.max():.12g} "
            f"mismatches={row['mismatches']}",
            flush=True,
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
