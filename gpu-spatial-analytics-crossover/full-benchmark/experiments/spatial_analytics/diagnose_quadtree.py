"""Locate cuSpatial indexed PIP correctness boundaries without timing claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .compat import initialize_single_gpu_context
from .geometry import make_points, polygon_rings, quadtree_scale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, nargs="+", default=[100_000, 1_000_000, 10_000_000])
    parser.add_argument("--depths", type=int, nargs="+", default=[12, 13, 14, 15])
    parser.add_argument("--features", type=int, default=1024)
    parser.add_argument("--vertices", type=int, default=17)
    parser.add_argument("--max-sizes", type=int, nargs="+", default=[128])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import cudf
    import cupy as cp
    import cuspatial
    import geopandas as gpd
    import shapely
    from shapely import STRtree

    initialize_single_gpu_context()
    polygons = np.asarray(
        [shapely.Polygon(ring) for ring in polygon_rings(args.features, args.vertices)],
        dtype=object,
    )
    gpu_polygons = cuspatial.from_geopandas(gpd.GeoSeries(polygons))
    boxes = cuspatial.polygon_bounding_boxes(gpu_polygons)
    tree = STRtree(polygons)
    evidence = []
    for rows in args.points:
        x, y = make_points(rows)
        host_points = shapely.points(x, y)
        cpu_matches = tree.query(host_points, predicate="within").shape[1]
        xy = cudf.DataFrame({"x": cp.asarray(x), "y": cp.asarray(y)}).interleave_columns()
        points = cuspatial.GeoSeries.from_points_xy(xy)
        for depth in args.depths:
            for max_size in args.max_sizes:
                scale = quadtree_scale(depth)
                point_indices, quadtree = cuspatial.quadtree_on_points(
                    points, 0.0, 1.0, 0.0, 1.0, scale, depth, max_size
                )
                pairs = cuspatial.join_quadtree_and_bounding_boxes(
                    quadtree, boxes, 0.0, 1.0, 0.0, 1.0, scale, depth
                )
                matches = cuspatial.quadtree_point_in_polygon(
                    pairs, quadtree, point_indices, points, gpu_polygons
                )
                cp.cuda.get_current_stream().synchronize()
                leaf_count = int((~quadtree["is_internal_node"]).sum())
                points_in_leaves = int(
                    quadtree.loc[~quadtree["is_internal_node"], "length"].sum()
                )
                row = {
                    "points": rows,
                    "features": args.features,
                    "vertices": args.vertices,
                    "depth": depth,
                    "max_size": max_size,
                    "cpu_matches": cpu_matches,
                    "gpu_matches": len(matches),
                    "quad_pairs": len(pairs),
                    "leaves": leaf_count,
                    "points_in_leaves": points_in_leaves,
                    "correct": len(matches) == cpu_matches,
                }
                evidence.append(row)
                print(
                    f"points={rows} depth={depth} max_size={max_size} "
                    f"cpu={cpu_matches} gpu={len(matches)} quad_pairs={len(pairs)} "
                    f"leaves={leaf_count} points_in_leaves={points_in_leaves} "
                    f"correct={len(matches) == cpu_matches}",
                    flush=True,
                )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
