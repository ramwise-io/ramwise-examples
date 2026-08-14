"""Benchmark launch, transfer, fusion, layout, and launch-configuration costs."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from experiments.custom_cuda.common import atomic_json, environment_metadata, stable_id
from experiments.custom_cuda.kernels import (
    kernel_attributes,
    partial_fusion_kernel,
    score_module,
    touch_kernel,
)


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def one_shot_input_layout(implementation: str) -> str:
    """Return the single host input layout charged to a one-shot path."""
    return "soa" if implementation.startswith("raw_soa") else "aos"


def cuda_timed(call: Callable[[], Any]) -> tuple[Any, float, float]:
    import cupy as cp

    cp.cuda.Stream.null.synchronize()
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    wall_start = time.perf_counter()
    start_event.record()
    value = call()
    end_event.record()
    end_event.synchronize()
    wall_seconds = time.perf_counter() - wall_start
    device_seconds = cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0
    return value, wall_seconds, device_seconds


def numpy_score(
    values: np.ndarray,
    means: np.ndarray,
    inverse_stds: np.ndarray,
    weights: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    transformed = np.clip((values - means) * inverse_stds, -5.0, 5.0)
    scores = np.sum(transformed * weights, axis=1, dtype=np.float32)
    return scores, (scores > threshold).astype(np.uint8)


def quality(reference: tuple[np.ndarray, np.ndarray], candidate: Any) -> dict[str, Any]:
    import cupy as cp

    scores, flags = candidate
    if isinstance(scores, cp.ndarray):
        scores = cp.asnumpy(scores)
    if isinstance(flags, cp.ndarray):
        flags = cp.asnumpy(flags)
    reference_scores, reference_flags = reference
    difference = np.abs(reference_scores - scores)
    maximum = float(difference.max(initial=0.0))
    mean = float(difference.mean())
    flag_mismatches = int(np.count_nonzero(reference_flags != flags))
    if not np.allclose(reference_scores, scores, rtol=3e-4, atol=3e-4):
        raise RuntimeError(
            f"Score validation failed: max_abs={maximum}, mean_abs={mean}"
        )
    if flag_mismatches:
        near_threshold = np.abs(reference_scores) <= max(3e-4, maximum)
        unexplained = np.count_nonzero((reference_flags != flags) & ~near_threshold)
        if unexplained:
            raise RuntimeError(
                f"Flag validation failed: {flag_mismatches} mismatches, "
                f"{unexplained} away from the threshold"
            )
    return {
        "max_abs_score_error": maximum,
        "mean_abs_score_error": mean,
        "flag_mismatches": flag_mismatches,
        "status": "pass",
    }


def run_launch(args: argparse.Namespace) -> dict[str, Any]:
    import cupy as cp

    metadata = environment_metadata()
    compile_start = time.perf_counter()
    raw_touch = touch_kernel()
    raw_touch.compile()
    cp.cuda.Stream.null.synchronize()
    compile_seconds = time.perf_counter() - compile_start
    value = cp.zeros(1, dtype=cp.float32)

    def raw_call() -> None:
        raw_touch((1,), (32,), (value,))

    def cupy_call() -> None:
        cp.add(value, np.float32(1.0), out=value)

    implementations = {"raw_touch": raw_call, "cupy_add": cupy_call}
    for call in implementations.values():
        for _ in range(args.warmups):
            call()
        cp.cuda.Stream.null.synchronize()

    records = []
    for name, call in implementations.items():
        wall_samples: list[float] = []
        device_samples: list[float] = []
        for _ in range(args.trials):
            value.fill(0)

            def repeated() -> None:
                for _ in range(args.launches):
                    call()

            _, wall, device = cuda_timed(repeated)
            wall_samples.append(wall / args.launches)
            device_samples.append(device / args.launches)
            observed = float(cp.asnumpy(value)[0])
            if observed != float(args.launches):
                raise RuntimeError(
                    f"Launch validation failed for {name}: {observed}"
                )
        records.append(
            {
                "implementation": name,
                "launches_per_trial": args.launches,
                "wall_seconds_per_launch": median(wall_samples),
                "wall_seconds_per_launch_min": min(wall_samples),
                "wall_seconds_per_launch_max": max(wall_samples),
                "device_seconds_per_launch": median(device_samples),
                "device_seconds_per_launch_min": min(device_samples),
                "device_seconds_per_launch_max": max(device_samples),
            }
        )
    return {
        "mode": "launch",
        "condition": {"launches": args.launches},
        "compile_seconds": compile_seconds,
        "kernel": kernel_attributes(raw_touch, 32, metadata),
        "records": records,
        "environment": metadata,
    }


def pinned_array(size: int) -> tuple[np.ndarray, Any]:
    import cupy as cp

    owner = cp.cuda.alloc_pinned_memory(size)
    return np.frombuffer(owner, dtype=np.uint8, count=size), owner


def run_transfer(args: argparse.Namespace) -> dict[str, Any]:
    import cupy as cp

    metadata = environment_metadata()
    if args.memory == "pinned":
        host, pinned_owner = pinned_array(args.bytes)
    else:
        host = np.empty(args.bytes, dtype=np.uint8)
        pinned_owner = None
    host.fill(17)
    device = cp.empty(args.bytes, dtype=cp.uint8)
    stream = cp.cuda.Stream.null

    if args.direction == "h2d":
        def copy() -> None:
            device.set(host, stream=stream)
    else:
        device.fill(23)

        def copy() -> None:
            device.get(out=host, stream=stream, blocking=False)

    for _ in range(args.warmups):
        copy()
    stream.synchronize()
    wall_samples: list[float] = []
    device_samples: list[float] = []
    for _ in range(args.trials):
        _, wall, device_time = cuda_timed(copy)
        wall_samples.append(wall)
        device_samples.append(device_time)
    if args.direction == "h2d":
        observed = int(cp.asnumpy(device[:1])[0])
        expected = 17
    else:
        observed = int(host[0])
        expected = 23
    if observed != expected:
        raise RuntimeError(f"Transfer validation failed: {observed} != {expected}")
    wall = median(wall_samples)
    device_time = median(device_samples)
    del pinned_owner
    return {
        "mode": "transfer",
        "condition": {
            "bytes": args.bytes,
            "direction": args.direction,
            "memory": args.memory,
        },
        "records": [
            {
                "bytes": args.bytes,
                "direction": args.direction,
                "memory": args.memory,
                "wall_seconds": wall,
                "wall_seconds_min": min(wall_samples),
                "wall_seconds_max": max(wall_samples),
                "device_seconds": device_time,
                "device_seconds_min": min(device_samples),
                "device_seconds_max": max(device_samples),
                "wall_gib_per_second": args.bytes / wall / 2**30,
                "device_gib_per_second": args.bytes / device_time / 2**30,
                "quality_status": "pass",
            }
        ],
        "environment": metadata,
    }


def run_fusion(args: argparse.Namespace) -> dict[str, Any]:
    import cupy as cp

    metadata = environment_metadata()
    rng = np.random.default_rng(args.seed + args.replication)
    host_aos = rng.standard_normal(
        (args.rows, args.features), dtype=np.float32
    )
    host_soa = np.ascontiguousarray(host_aos.T)
    means = rng.normal(0.0, 0.2, args.features).astype(np.float32)
    inverse_stds = rng.uniform(0.5, 1.5, args.features).astype(np.float32)
    weights = rng.normal(0.0, 1.0, args.features).astype(np.float32)
    threshold = np.float32(0.0)

    cpu_samples: list[float] = []
    reference: tuple[np.ndarray, np.ndarray] | None = None
    for _ in range(args.warmups):
        reference = numpy_score(host_aos, means, inverse_stds, weights, threshold)
    for _ in range(args.trials):
        started = time.perf_counter()
        candidate = numpy_score(host_aos, means, inverse_stds, weights, threshold)
        cpu_samples.append(time.perf_counter() - started)
        reference = candidate
    assert reference is not None

    device_aos = cp.asarray(host_aos)
    device_soa = cp.asarray(host_soa)
    device_means = cp.asarray(means)
    device_inverse_stds = cp.asarray(inverse_stds)
    device_weights = cp.asarray(weights)
    padding = max(1, 32 - args.features % 32)
    row_stride = args.features + padding
    device_strided = cp.empty((args.rows, row_stride), dtype=cp.float32)
    device_strided[:, : args.features] = device_aos

    compile_start = time.perf_counter()
    module = score_module()
    module.compile()
    cp.cuda.Stream.null.synchronize()
    raw_compile_seconds = time.perf_counter() - compile_start
    partial = partial_fusion_kernel()
    partial_compile_start = time.perf_counter()
    probe = cp.zeros(1, dtype=cp.float32)
    partial_probe = partial(
        probe,
        probe,
        probe,
        probe,
        np.int32(1),
    )
    cp.cuda.Stream.null.synchronize()
    partial_compile_seconds = time.perf_counter() - partial_compile_start
    del partial_probe, probe
    raw_kernels = {
        "raw_aos": module.get_function("score_aos"),
        "raw_soa": module.get_function("score_soa"),
        "raw_strided": module.get_function("score_strided"),
    }

    def composed() -> tuple[Any, Any]:
        transformed = cp.clip(
            (device_aos - device_means) * device_inverse_stds, -5.0, 5.0
        )
        scores = cp.sum(transformed * device_weights, axis=1, dtype=cp.float32)
        return scores, (scores > threshold).astype(cp.uint8)

    def partial_fused() -> tuple[Any, Any]:
        transformed = partial(
            device_aos.ravel(),
            device_means,
            device_inverse_stds,
            device_weights,
            np.int32(args.features),
        )
        scores = cp.sum(
            transformed.reshape(args.rows, args.features), axis=1, dtype=cp.float32
        )
        return scores, (scores > threshold).astype(cp.uint8)

    resident_calls: dict[str, Callable[[], Any]] = {
        "cupy_composed": composed,
        "cupy_partial_fusion": partial_fused,
    }
    attributes: dict[tuple[str, int], dict[str, Any]] = {}
    for block in args.blocks:
        grid = (math.ceil(args.rows / block),)
        for variant in ("raw_aos", "raw_soa", "raw_strided"):
            if variant not in args.variants:
                continue
            kernel = raw_kernels[variant]
            key = f"{variant}_b{block}"
            if variant == "raw_aos":
                values = device_aos
                extra: tuple[Any, ...] = ()
            elif variant == "raw_soa":
                values = device_soa
                extra = ()
            else:
                values = device_strided
                extra = (np.int32(row_stride),)

            def raw_call(
                kernel: Any = kernel,
                grid: tuple[int] = grid,
                block: int = block,
                values: Any = values,
                extra: tuple[Any, ...] = extra,
            ) -> tuple[Any, Any]:
                scores = cp.empty(args.rows, dtype=cp.float32)
                flags = cp.empty(args.rows, dtype=cp.uint8)
                kernel(
                    grid,
                    (block,),
                    (
                        values,
                        device_means,
                        device_inverse_stds,
                        device_weights,
                        scores,
                        flags,
                        np.int32(args.rows),
                        np.int32(args.features),
                        *extra,
                        threshold,
                    ),
                )
                return scores, flags

            resident_calls[key] = raw_call
            attributes[(key, block)] = kernel_attributes(kernel, block, metadata)
    resident_calls = {
        name: call
        for name, call in resident_calls.items()
        if name in args.variants or name.startswith(tuple(f"{v}_b" for v in args.variants))
    }
    if not resident_calls:
        raise ValueError("No fusion implementations selected")

    validation: dict[str, dict[str, Any]] = {}
    for name, call in resident_calls.items():
        candidate, _, _ = cuda_timed(call)
        validation[name] = quality(reference, candidate)
        del candidate

    for _ in range(args.warmups):
        for call in resident_calls.values():
            candidate, _, _ = cuda_timed(call)
            del candidate

    wall_samples: dict[str, list[float]] = {name: [] for name in resident_calls}
    device_samples: dict[str, list[float]] = {name: [] for name in resident_calls}
    order = list(resident_calls)
    for trial in range(args.trials):
        random.Random(args.seed + args.replication * 1000 + trial).shuffle(order)
        for name in order:
            candidate, wall, device_time = cuda_timed(resident_calls[name])
            wall_samples[name].append(wall)
            device_samples[name].append(device_time)
            del candidate

    layout_wall_samples: list[float] = []
    layout_device_samples: list[float] = []
    for _ in range(args.trials):
        converted, wall, device_time = cuda_timed(
            lambda: cp.ascontiguousarray(device_aos.T)
        )
        layout_wall_samples.append(wall)
        layout_device_samples.append(device_time)
        del converted

    one_shot_names = [
        name
        for name in resident_calls
        if name in {"cupy_composed", "cupy_partial_fusion"}
        or name in {f"raw_aos_b{args.blocks[0]}", f"raw_soa_b{args.blocks[0]}"}
    ]
    one_shot_samples: dict[str, list[float]] = {name: [] for name in one_shot_names}

    def one_shot(name: str) -> tuple[np.ndarray, np.ndarray]:
        # Upload exactly one input layout. In particular, the SoA path must not
        # also pay for an unused AoS copy.
        input_layout = one_shot_input_layout(name)
        local_aos = None if input_layout == "soa" else cp.asarray(host_aos)
        local_means = cp.asarray(means)
        local_inverse = cp.asarray(inverse_stds)
        local_weights = cp.asarray(weights)
        if name == "cupy_composed":
            assert local_aos is not None
            transformed = cp.clip(
                (local_aos - local_means) * local_inverse, -5.0, 5.0
            )
            scores = cp.sum(transformed * local_weights, axis=1, dtype=cp.float32)
            flags = (scores > threshold).astype(cp.uint8)
        elif name == "cupy_partial_fusion":
            assert local_aos is not None
            transformed = partial(
                local_aos.ravel(),
                local_means,
                local_inverse,
                local_weights,
                np.int32(args.features),
            )
            scores = cp.sum(
                transformed.reshape(args.rows, args.features), axis=1, dtype=cp.float32
            )
            flags = (scores > threshold).astype(cp.uint8)
        else:
            block = args.blocks[0]
            scores = cp.empty(args.rows, dtype=cp.float32)
            flags = cp.empty(args.rows, dtype=cp.uint8)
            if input_layout == "soa":
                values = cp.asarray(host_soa)
                kernel = raw_kernels["raw_soa"]
            else:
                assert local_aos is not None
                values = local_aos
                kernel = raw_kernels["raw_aos"]
            kernel(
                (math.ceil(args.rows / block),),
                (block,),
                (
                    values,
                    local_means,
                    local_inverse,
                    local_weights,
                    scores,
                    flags,
                    np.int32(args.rows),
                    np.int32(args.features),
                    threshold,
                ),
            )
        return cp.asnumpy(scores), cp.asnumpy(flags)

    for name in one_shot_names:
        observed: tuple[np.ndarray, np.ndarray] | None = None
        for _ in range(args.trials):
            cp.cuda.Stream.null.synchronize()
            started = time.perf_counter()
            observed = one_shot(name)
            cp.cuda.Stream.null.synchronize()
            one_shot_samples[name].append(time.perf_counter() - started)
        assert observed is not None
        quality(reference, observed)

    records = [
        {
            "implementation": "numpy",
            "layout": "aos",
            "block_size": None,
            "resident_wall_seconds": median(cpu_samples),
            "resident_wall_seconds_min": min(cpu_samples),
            "resident_wall_seconds_max": max(cpu_samples),
            "resident_device_seconds": None,
            "one_shot_seconds": median(cpu_samples),
            "quality": {"status": "reference"},
            "kernel": None,
        }
    ]
    for name in sorted(resident_calls):
        if name.startswith("raw_"):
            base, block_text = name.rsplit("_b", 1)
            block = int(block_text)
            layout = base.removeprefix("raw_")
            kernel_info = attributes[(name, block)]
        else:
            block = None
            layout = "aos"
            kernel_info = None
        records.append(
            {
                "implementation": name,
                "layout": layout,
                "block_size": block,
                "resident_wall_seconds": median(wall_samples[name]),
                "resident_wall_seconds_min": min(wall_samples[name]),
                "resident_wall_seconds_max": max(wall_samples[name]),
                "resident_device_seconds": median(device_samples[name]),
                "resident_device_seconds_min": min(device_samples[name]),
                "resident_device_seconds_max": max(device_samples[name]),
                "one_shot_seconds": (
                    median(one_shot_samples[name]) if name in one_shot_samples else None
                ),
                "one_shot_seconds_min": (
                    min(one_shot_samples[name]) if name in one_shot_samples else None
                ),
                "one_shot_seconds_max": (
                    max(one_shot_samples[name]) if name in one_shot_samples else None
                ),
                "quality": validation[name],
                "kernel": kernel_info,
            }
        )
    return {
        "mode": "fusion",
        "condition": {
            "rows": args.rows,
            "features": args.features,
            "input_bytes": args.rows * args.features * 4,
            "blocks": args.blocks,
            "variants": args.variants,
            "strided_row_stride": row_stride,
        },
        "compile_seconds": {
            "raw_module": raw_compile_seconds,
            "cupy_partial_fusion": partial_compile_seconds,
        },
        "layout_conversion": {
            "aos_to_soa_wall_seconds": median(layout_wall_samples),
            "aos_to_soa_device_seconds": median(layout_device_samples),
        },
        "records": records,
        "environment": metadata,
    }


def parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def parse_csv_strings(value: str) -> list[str]:
    return [item for item in value.split(",") if item]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("launch", "transfer", "fusion"), required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--replication", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--launches", type=int, default=100)
    parser.add_argument("--bytes", type=int, default=1024 * 1024)
    parser.add_argument("--direction", choices=("h2d", "d2h"), default="h2d")
    parser.add_argument("--memory", choices=("pageable", "pinned"), default="pageable")
    parser.add_argument("--rows", type=int, default=65536)
    parser.add_argument("--features", type=int, default=16)
    parser.add_argument("--blocks", type=parse_csv_ints, default=[256])
    parser.add_argument(
        "--variants",
        type=parse_csv_strings,
        default=[
            "cupy_composed",
            "cupy_partial_fusion",
            "raw_aos",
            "raw_soa",
            "raw_strided",
        ],
    )
    args = parser.parse_args()
    if args.warmups < 1 or args.trials < 1 or args.replication < 0:
        parser.error("warmups/trials must be positive and replication non-negative")
    if args.mode == "launch" and args.launches < 1:
        parser.error("launches must be positive")
    if args.mode == "transfer" and args.bytes < 1:
        parser.error("bytes must be positive")
    if args.mode == "fusion" and (args.rows < 1 or args.features < 1):
        parser.error("rows and features must be positive")

    condition = {
        "mode": args.mode,
        "launches": args.launches if args.mode == "launch" else None,
        "bytes": args.bytes if args.mode == "transfer" else None,
        "direction": args.direction if args.mode == "transfer" else None,
        "memory": args.memory if args.mode == "transfer" else None,
        "rows": args.rows if args.mode == "fusion" else None,
        "features": args.features if args.mode == "fusion" else None,
        "blocks": args.blocks if args.mode == "fusion" else None,
        "variants": args.variants if args.mode == "fusion" else None,
        "replication": args.replication,
        "seed": args.seed,
    }
    run_id = stable_id("custom-cuda", condition)
    if args.mode == "launch":
        result = run_launch(args)
    elif args.mode == "transfer":
        result = run_transfer(args)
    else:
        result = run_fusion(args)
    result.update(
        {
            "schema_version": 1,
            "run_id": run_id,
            "replication": args.replication,
            "seed": args.seed,
            "warmups": args.warmups,
            "trials": args.trials,
        }
    )
    output = args.output.resolve()
    atomic_json(output, result)
    print(json.dumps({"run_id": run_id, "result": str(output)}, indent=2))


if __name__ == "__main__":
    main()
