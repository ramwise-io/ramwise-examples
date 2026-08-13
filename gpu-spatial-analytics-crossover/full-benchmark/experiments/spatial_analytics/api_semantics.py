"""Small correctness probe for cuSpatial's indexed join result semantics."""

from __future__ import annotations

import numpy as np

from .geometry import linestring_coordinates, make_points, polygon_rings, quadtree_scale
from .compat import initialize_single_gpu_context


def main() -> None:
    import cudf
    import cupy as cp
    import cuspatial
    import geopandas as gpd
    import shapely
    from shapely import STRtree

    initialize_single_gpu_context()

    x, y = make_points(10_000)
    host_points = shapely.points(x, y)
    host_polygons = np.asarray(
        [shapely.Polygon(ring) for ring in polygon_rings(16, 17)], dtype=object
    )
    host_lines = np.asarray(
        [shapely.LineString(line) for line in linestring_coordinates(16, 9)], dtype=object
    )

    point_pairs = STRtree(host_polygons).query(host_points, predicate="within")
    point_reference = set(map(tuple, point_pairs.T.tolist()))

    xy = cudf.DataFrame({"x": cp.asarray(x), "y": cp.asarray(y)}).interleave_columns()
    gpu_points = cuspatial.GeoSeries.from_points_xy(xy)
    gpu_polygons = cuspatial.from_geopandas(gpd.GeoSeries(host_polygons))
    gpu_lines = cuspatial.from_geopandas(gpd.GeoSeries(host_lines))
    max_depth = 12
    scale = quadtree_scale(max_depth)
    point_indices, quadtree = cuspatial.quadtree_on_points(
        gpu_points, 0.0, 1.0, 0.0, 1.0, scale, max_depth, 64
    )
    boxes = cuspatial.polygon_bounding_boxes(gpu_polygons)
    quad_pairs = cuspatial.join_quadtree_and_bounding_boxes(
        quadtree, boxes, 0.0, 1.0, 0.0, 1.0, scale, max_depth
    )
    matches = cuspatial.quadtree_point_in_polygon(
        quad_pairs, quadtree, point_indices, gpu_points, gpu_polygons
    ).to_pandas()

    # Public docs say point_index is an index into point_indices. Confirm which
    # mapping agrees with the CPU pair set before the benchmark relies on it.
    direct_pairs = set(zip(matches["point_index"], matches["polygon_index"]))
    mapped_points = point_indices.iloc[cudf.Series(matches["point_index"].to_numpy())]
    mapped_pairs = set(zip(mapped_points.to_pandas(), matches["polygon_index"]))
    print(f"cpu_matches={len(point_reference)}")
    print(f"gpu_matches={len(matches)}")
    print(f"direct_indices_equal={direct_pairs == point_reference}")
    print(f"mapped_indices_equal={mapped_pairs == point_reference}")
    if len(matches) != len(point_reference) or not (direct_pairs == point_reference or mapped_pairs == point_reference):
        raise AssertionError("cuSpatial point-in-polygon pairs disagree with Shapely")

    line_boxes = cuspatial.linestring_bounding_boxes(
        gpu_lines, expansion_radius=0.75 / len(host_lines)
    )
    line_quad_pairs = cuspatial.join_quadtree_and_bounding_boxes(
        quadtree, line_boxes, 0.0, 1.0, 0.0, 1.0, scale, max_depth
    )
    nearest = cuspatial.quadtree_point_to_nearest_linestring(
        line_quad_pairs, quadtree, point_indices, gpu_points, gpu_lines
    ).to_pandas()
    sample = np.arange(0, len(host_points), 101)
    _, reference_distances = STRtree(host_lines).query_nearest(
        host_points[sample], all_matches=False, return_distance=True
    )
    gpu_distance_by_point = np.full(len(host_points), np.nan)
    nearest_point_offsets = nearest.iloc[:, 0].to_numpy()
    original_points = point_indices.iloc[cudf.Series(nearest_point_offsets)].to_pandas().to_numpy()
    gpu_distance_by_point[original_points] = nearest.iloc[:, 2].to_numpy()
    max_error = np.nanmax(np.abs(gpu_distance_by_point[sample] - reference_distances))
    missing = int(np.isnan(gpu_distance_by_point[sample]).sum())
    print(f"nearest_rows={len(nearest)}")
    print(f"nearest_sample_missing={missing}")
    print(f"nearest_sample_max_abs_error={max_error:.12g}")
    if missing or max_error > 1e-9:
        raise AssertionError("cuSpatial nearest-linestring result disagrees with Shapely")


if __name__ == "__main__":
    main()
