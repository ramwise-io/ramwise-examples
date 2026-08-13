# Methodology

## Systems

- Host: Ubuntu 26.04, NVIDIA RTX PRO 4000 Blackwell SFF (24 GB), driver 595.71.05.
- Container: digest-pinned NVIDIA RAPIDS 25.04 CUDA 12.8 base.
- CPU allocation: eight fixed logical CPUs (`0-7`).
- CPU: Shapely 2.0.7/GEOS 3.13.1 and NumPy 2.0.2.
- GPU: cuSpatial/cuDF 25.04 and CuPy 13.4.1.

cuSpatial 25.04 is the final stable release and is currently classified as an
inactive RAPIDS project. A fail-closed compatibility shim corrects an invalid
Numba device ordinal only after CuPy and CUDA bindings independently confirm
exactly one visible GPU at ordinal zero.

## Geometry and matrix

Deterministic planar points lie in `[0,1)^2`. Regular polygons vary by count,
vertices, and radius. Deterministic lines vary by count and vertices. The
matrix covers rectangular window filtering, direct point-in-polygon, indexed
point/polygon joins, and nearest linestring queries up to 100M points.

Each case has one warmup and three measured trials. The published value is the
median within each fresh process and then the median of three randomized
fresh-process replications. GPU timers synchronize before and after each
region. CPU and GPU separately report ingest, index build, candidate, exact,
query, and result-materialization time.

The end-to-end boundary is:

```text
ingest + index build + query + materialization
```

For a reused resident index, the query-count break-even is:

```text
ceil((GPU fixed cost - CPU fixed cost) / (CPU query - GPU query))
```

There is no finite break-even when the GPU repeated query is not faster.

## CPU and GPU plans

- Window: vectorized NumPy mask versus cuSpatial spatial window.
- Direct point-in-polygon: Shapely `contains_xy` broadcast versus cuSpatial
  `point_in_polygon` (16 polygons, within the API's small direct-matrix scope).
- Indexed join: eight-thread Shapely STRtree bbox candidates plus exact
  `within` refinement versus cuSpatial quadtree/bbox candidates plus
  `quadtree_point_in_polygon`.
- Nearest: eight-thread STRtree `query_nearest` versus cuSpatial quadtree
  nearest-linestring.

## Correctness and publication gates

Window match count and checksum must agree. Join count and two independent
order-independent 64-bit pair reductions must agree. Nearest results must form
a complete unique covering of point IDs and pass pointwise distance tolerance.
Every quadtree must account for every input point in both its permutation and
leaf lengths before timing continues.

A performance row is public only if all three replications succeed. Any failed
replication excludes the whole case. The exclusions and targeted diagnostic
controls are public; raw process telemetry remains private.

## Scope

This is synthetic 2D Cartesian geometry already in memory. It does not include
file parsing, CRS transformation, invalid geometries, spherical distance,
updates, multi-GPU execution, or real-world skew. Whole-process RSS and
whole-device GPU use are sampled, not attributed allocation traces.

