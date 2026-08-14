"""Benchmark one algorithm/engine condition in a fresh process."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from experiments.gpu_ml_pipeline import EXPERIMENT_NAME, SCHEMA_VERSION
from experiments.gpu_ml_pipeline.common import (
    atomic_json,
    read_json,
    source_id,
    stable_id,
    warm_file_cache,
)
from experiments.gpu_ml_pipeline.estimators import (
    ALGORITHMS,
    ENGINES,
    SUPERVISED,
    allowed_engines,
    enable_accelerator,
    make_estimator,
)
from experiments.gpu_ml_pipeline.telemetry import GpuSampler


def synchronize(engine: str) -> None:
    if engine in {"accel", "native_gpu"}:
        import cupy as cp

        cp.cuda.runtime.deviceSynchronize()


def to_host(value: Any) -> np.ndarray:
    if hasattr(value, "get"):
        value = value.get()
    elif hasattr(value, "to_numpy"):
        value = value.to_numpy()
    return np.asarray(value)


def fit_model(model: Any, algorithm: str, x: Any, y: Any) -> Any:
    if algorithm in SUPERVISED:
        return model.fit(x, y)
    return model.fit(x)


def inference(model: Any, algorithm: str, x: Any) -> Any:
    if algorithm in SUPERVISED:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(x)
        return model.predict(x)
    if algorithm in {"pca", "umap"}:
        return model.transform(x)
    if algorithm == "kmeans":
        return model.predict(x)
    return None


def quality(
    model: Any,
    algorithm: str,
    x_train_host: np.ndarray,
    x_test: Any,
    y_test: np.ndarray,
    clusters_train: np.ndarray,
) -> dict[str, float | int]:
    from sklearn.metrics import accuracy_score, adjusted_rand_score, log_loss, roc_auc_score

    if algorithm in SUPERVISED:
        probabilities = to_host(model.predict_proba(x_test))
        positive = probabilities[:, 1] if probabilities.ndim == 2 else probabilities
        predicted = (positive >= 0.5).astype(np.int8)
        return {
            "accuracy": float(accuracy_score(y_test, predicted)),
            "log_loss": float(log_loss(y_test, probabilities)),
            "roc_auc": float(roc_auc_score(y_test, positive)),
        }
    if algorithm == "pca":
        ratios = to_host(model.explained_variance_ratio_)
        return {"explained_variance_ratio_sum": float(ratios.sum())}
    if algorithm in {"kmeans", "hdbscan"}:
        labels = to_host(model.labels_).reshape(-1)
        return {
            "adjusted_rand_score": float(adjusted_rand_score(clusters_train, labels)),
            "clusters_found": int(len(set(labels.tolist())) - (1 if -1 in labels else 0)),
            "noise_fraction": float(np.mean(labels == -1)),
        }
    if algorithm == "umap":
        from sklearn.manifold import trustworthiness

        embedding = to_host(model.embedding_)
        sample = min(2_000, len(embedding))
        return {
            "trustworthiness": float(
                trustworthiness(
                    x_train_host[:sample], embedding[:sample], n_neighbors=5
                )
            )
        }
    raise AssertionError(f"No quality implementation for {algorithm}")


def environment() -> dict[str, Any]:
    import cudf
    import cuml
    import cupy
    import pandas
    import sklearn
    import xgboost

    optional: dict[str, str] = {}
    try:
        import umap

        optional["umap"] = umap.__version__
    except Exception:
        optional["umap"] = "not-installed"
    try:
        import importlib.metadata

        optional["hdbscan"] = importlib.metadata.version("hdbscan")
    except Exception:
        optional["hdbscan"] = "not-installed"
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    cpu_model = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "host_hostname": os.environ.get("GPU_LAB_HOSTNAME", "unknown"),
        "host_os": os.environ.get("GPU_LAB_HOST_OS", "unknown"),
        "container_image": os.environ.get("GPU_LAB_IMAGE", "unknown"),
        "container_image_id": os.environ.get("GPU_LAB_IMAGE_ID", "unknown"),
        "gpu": gpu,
        "cpu_count": os.cpu_count(),
        "cpu_affinity": psutil.Process().cpu_affinity(),
        "cpu_model": cpu_model,
        "cuda_runtime": cupy.cuda.runtime.runtimeGetVersion(),
        "cudf": cudf.__version__,
        "cuml": cuml.__version__,
        "cupy": cupy.__version__,
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        **optional,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--algorithm", choices=ALGORITHMS, required=True)
    parser.add_argument("--engine", choices=ENGINES, required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--replication", type=int, default=0)
    parser.add_argument("--inference-batches", default="1,256,4096")
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument(
        "--output-root", type=Path, default=Path("/data/benchmarks/gpu-ml-pipeline/raw")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.engine not in allowed_engines(args.algorithm):
        raise ValueError(f"{args.engine} is not valid for {args.algorithm}")
    if args.warmups < 0 or args.trials < 1:
        raise ValueError("warmups must be non-negative and trials must be positive")
    if args.engine == "accel":
        enable_accelerator()
    manifest_path = args.manifest.resolve()
    manifest = read_json(manifest_path)
    config = manifest["config"]
    identity = {
        "algorithm": args.algorithm,
        "dataset_id": manifest["dataset_id"],
        "engine": args.engine,
        "git_commit": source_id(),
        "replication": args.replication,
        "seed": args.seed,
        "trials": args.trials,
        "warmups": args.warmups,
    }
    run_id = stable_id("gpu-ml", identity)
    output_dir = args.output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    split_paths = {
        split: {
            key: manifest_path.parent / value["path"]
            for key, value in details["files"].items()
        }
        for split, details in manifest["splits"].items()
    }
    warm_file_cache(
        [
            split_paths[split][key]
            for split in ("train", "test")
            for key in ("X", "y", "clusters")
        ]
    )
    load_started = time.perf_counter()
    x_train_host = np.load(split_paths["train"]["X"])
    y_train_host = np.load(split_paths["train"]["y"]).astype(np.int32, copy=False)
    clusters_train = np.load(split_paths["train"]["clusters"])
    x_test_host = np.load(split_paths["test"]["X"])
    y_test_host = np.load(split_paths["test"]["y"]).astype(np.int32, copy=False)
    load_seconds = time.perf_counter() - load_started

    transfer_seconds = 0.0
    x_train: Any = x_train_host
    y_train: Any = y_train_host
    x_test: Any = x_test_host
    if args.engine == "native_gpu":
        import cupy as cp

        synchronize(args.engine)
        transfer_started = time.perf_counter()
        x_train = cp.asarray(x_train_host)
        y_train = cp.asarray(y_train_host)
        x_test = cp.asarray(x_test_host)
        synchronize(args.engine)
        transfer_seconds = time.perf_counter() - transfer_started

    warm_rows = min(4_096, len(x_train_host))
    for index in range(args.warmups):
        warm = make_estimator(
            args.algorithm,
            args.engine,
            features=int(config["features"]),
            clusters=int(config["clusters"]),
            seed=args.seed + index,
        )
        fit_model(warm, args.algorithm, x_train[:warm_rows], y_train[:warm_rows])
        synchronize(args.engine)
        if inference(warm, args.algorithm, x_test[: min(32, len(x_test_host))]) is not None:
            synchronize(args.engine)

    fit_trials: list[dict[str, Any]] = []
    final_model: Any = None
    for trial in range(args.trials):
        model = make_estimator(
            args.algorithm,
            args.engine,
            features=int(config["features"]),
            clusters=int(config["clusters"]),
            seed=args.seed,
        )
        sampler = GpuSampler()
        sampler.start()
        synchronize(args.engine)
        started = time.perf_counter()
        fit_model(model, args.algorithm, x_train, y_train)
        synchronize(args.engine)
        elapsed = time.perf_counter() - started
        fit_trials.append(
            {"trial": trial, "elapsed_seconds": elapsed, "gpu": sampler.stop()}
        )
        final_model = model
    assert final_model is not None

    requested_batches = sorted(
        {int(value) for value in args.inference_batches.split(",") if int(value) > 0}
    )
    inference_results: list[dict[str, Any]] = []
    if args.algorithm != "hdbscan":
        for batch in [value for value in requested_batches if value <= len(x_test_host)]:
            values = x_test[:batch]
            inference(final_model, args.algorithm, values)
            synchronize(args.engine)
            compute_trials = []
            materialize_trials = []
            for trial in range(args.trials):
                synchronize(args.engine)
                started = time.perf_counter()
                output = inference(final_model, args.algorithm, values)
                synchronize(args.engine)
                compute = time.perf_counter() - started
                materialize_started = time.perf_counter()
                to_host(output)
                materialize = time.perf_counter() - materialize_started
                compute_trials.append(compute)
                materialize_trials.append(materialize)
            median_compute = statistics.median(compute_trials)
            median_materialize = statistics.median(materialize_trials)
            inference_results.append(
                {
                    "batch_size": batch,
                    "compute_seconds": median_compute,
                    "materialize_seconds": median_materialize,
                    "end_to_end_seconds": median_compute + median_materialize,
                    "rows_per_second": batch / (median_compute + median_materialize),
                    "compute_trials_seconds": compute_trials,
                    "materialize_trials_seconds": materialize_trials,
                }
            )

    quality_values = quality(
        final_model,
        args.algorithm,
        x_train_host,
        x_test,
        y_test_host,
        clusters_train,
    )
    proxy = False
    if args.engine == "accel":
        import cuml.accel

        proxy = bool(cuml.accel.is_proxy(final_model))
    fit_median = statistics.median(x["elapsed_seconds"] for x in fit_trials)
    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_NAME,
        "run_id": run_id,
        "git_commit": source_id(),
        "dataset": manifest,
        "condition": {
            "algorithm": args.algorithm,
            "engine": args.engine,
            "replication": args.replication,
            "seed": args.seed,
            "trials": args.trials,
            "warmups": args.warmups,
        },
        "environment": environment(),
        "dispatch": {"cuml_accel_proxy": proxy},
        "timing": {
            "load_seconds": load_seconds,
            "explicit_transfer_seconds": transfer_seconds,
            "fit_median_seconds": fit_median,
            "fit_trials": fit_trials,
            "one_shot_fit_seconds": load_seconds + transfer_seconds + fit_median,
            "inference": inference_results,
        },
        "memory": {
            "host_rss_bytes": psutil.Process().memory_info().rss,
            "input_bytes": int(x_train_host.nbytes + x_test_host.nbytes),
        },
        "quality": quality_values,
    }
    result_path = output_dir / "result.json"
    atomic_json(result_path, result)
    print(json.dumps({"result": str(result_path), "run_id": run_id}, indent=2))


if __name__ == "__main__":
    main()
