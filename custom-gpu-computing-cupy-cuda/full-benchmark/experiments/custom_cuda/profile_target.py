"""Run one named custom CUDA kernel as a stable Nsight Compute target."""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from experiments.custom_cuda.common import environment_metadata
from experiments.custom_cuda.kernels import kernel_attributes, score_module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--features", type=int, required=True)
    parser.add_argument("--layout", choices=("aos", "soa"), required=True)
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--warmups", type=int, default=1)
    args = parser.parse_args()
    if args.rows < 1 or args.features < 1 or args.warmups < 0:
        parser.error("rows/features must be positive and warmups non-negative")
    if args.block < 32 or args.block > 1024 or args.block % 32:
        parser.error("block must be warp-aligned from 32 through 1024")

    import cupy as cp

    rng = np.random.default_rng(20260814)
    host = rng.standard_normal((args.rows, args.features), dtype=np.float32)
    means = cp.asarray(rng.normal(0.0, 0.2, args.features).astype(np.float32))
    inverse = cp.asarray(rng.uniform(0.5, 1.5, args.features).astype(np.float32))
    weights = cp.asarray(rng.normal(0.0, 1.0, args.features).astype(np.float32))
    if args.layout == "aos":
        values = cp.asarray(host)
        kernel_name = "score_aos"
    else:
        values = cp.asarray(np.ascontiguousarray(host.T))
        kernel_name = "score_soa"
    del host
    scores = cp.empty(args.rows, dtype=cp.float32)
    flags = cp.empty(args.rows, dtype=cp.uint8)
    module = score_module()
    module.compile()
    kernel = module.get_function(kernel_name)
    grid = (math.ceil(args.rows / args.block),)
    arguments = (
        values,
        means,
        inverse,
        weights,
        scores,
        flags,
        np.int32(args.rows),
        np.int32(args.features),
        np.float32(0.0),
    )
    for _ in range(args.warmups):
        kernel(grid, (args.block,), arguments)
    cp.cuda.Stream.null.synchronize()
    kernel(grid, (args.block,), arguments)
    cp.cuda.Stream.null.synchronize()
    checksum = {
        "score_sum": float(cp.sum(scores, dtype=cp.float64).get()),
        "flag_sum": int(cp.sum(flags, dtype=cp.int64).get()),
    }
    metadata = environment_metadata()
    print(
        json.dumps(
            {
                "rows": args.rows,
                "features": args.features,
                "layout": args.layout,
                "kernel": kernel_name,
                "grid": grid[0],
                "attributes": kernel_attributes(kernel, args.block, metadata),
                "checksum": checksum,
                "environment": metadata,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

