"""Benchmark complete CPU, GPU-resident, and hybrid ML pipelines."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from experiments.gpu_ml_pipeline import EXPERIMENT_NAME, SCHEMA_VERSION
from experiments.gpu_ml_pipeline.common import atomic_json, read_json, source_id, stable_id
from experiments.gpu_ml_pipeline.estimators import enable_accelerator
from experiments.gpu_ml_pipeline.telemetry import GpuSampler

MODES = ("cpu", "accel", "gpu_resident", "cpu_to_gpu", "ping_pong")


def synchronize(engine: str) -> None:
    if engine != "cpu":
        import cupy as cp

        cp.cuda.runtime.deviceSynchronize()


def to_host(value: Any) -> np.ndarray:
    if hasattr(value, "get"):
        value = value.get()
    elif hasattr(value, "to_numpy"):
        value = value.to_numpy()
    return np.asarray(value)


def feature_engineering(values: Any, xp: Any) -> Any:
    width = min(8, values.shape[1])
    extras = [values[:, index] ** 2 for index in range(width)]
    if width >= 4:
        extras.extend(
            [
                values[:, 0] * values[:, 1],
                values[:, 2] * values[:, 3],
                values[:, 0] / (xp.abs(values[:, 2]) + 1.0),
                xp.log1p(xp.abs(values[:, 3])),
            ]
        )
    return xp.column_stack([values, *extras]).astype(xp.float32, copy=False)


def timed_stage(
    stages: dict[str, float], name: str, engine: str, call: Callable[[], Any]
) -> Any:
    synchronize(engine)
    started = time.perf_counter()
    value = call()
    synchronize(engine)
    stages[name] = stages.get(name, 0.0) + time.perf_counter() - started
    return value


def cpu_components(seed: int, components: int) -> tuple[Any, Any, Any]:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    return (
        StandardScaler(),
        PCA(n_components=components, svd_solver="full", random_state=seed),
        LogisticRegression(C=1.0, max_iter=200, solver="lbfgs"),
    )


def gpu_components(components: int) -> tuple[Any, Any, Any]:
    from cuml.decomposition import PCA
    from cuml.linear_model import LogisticRegression
    from cuml.preprocessing import StandardScaler

    return (
        StandardScaler(output_type="cupy"),
        PCA(n_components=components, svd_solver="full", output_type="cupy"),
        LogisticRegression(C=1.0, max_iter=200, solver="qn", output_type="cupy"),
    )


def load_arrays(manifest_path: Path, limit: int | None = None) -> tuple[np.ndarray, ...]:
    manifest = read_json(manifest_path)
    base = manifest_path.parent
    arrays = []
    for split in ("train", "test"):
        files = manifest["splits"][split]["files"]
        x = np.load(base / files["X"]["path"])
        y = np.load(base / files["y"]["path"]).astype(np.int32, copy=False)
        if limit is not None:
            x = x[:limit]
            y = y[:limit]
        arrays.extend((x, y))
    return tuple(arrays)


def run_once(
    manifest_path: Path,
    mode: str,
    seed: int,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    import cupy as cp

    stages: dict[str, float] = {}
    transfer_bytes = {"host_to_device": 0, "device_to_host": 0, "count": 0}
    total_started = time.perf_counter()
    x_train_host, y_train_host, x_test_host, y_test_host = timed_stage(
        stages, "load", "cpu", lambda: load_arrays(manifest_path, limit)
    )
    components = min(16, max(2, x_train_host.shape[1] // 2))

    if mode in {"cpu", "accel"}:
        scaler, pca, model = cpu_components(seed, components)
        x_train_scaled = timed_stage(
            stages, "scale", mode, lambda: scaler.fit_transform(x_train_host)
        )
        x_test_scaled = timed_stage(
            stages, "scale", mode, lambda: scaler.transform(x_test_host)
        )
        x_train_features = timed_stage(
            stages,
            "feature_engineering",
            "cpu",
            lambda: feature_engineering(x_train_scaled, np),
        )
        x_test_features = timed_stage(
            stages,
            "feature_engineering",
            "cpu",
            lambda: feature_engineering(x_test_scaled, np),
        )
        x_train_reduced = timed_stage(
            stages, "pca", mode, lambda: pca.fit_transform(x_train_features)
        )
        x_test_reduced = timed_stage(
            stages, "pca", mode, lambda: pca.transform(x_test_features)
        )
        timed_stage(
            stages,
            "train",
            mode,
            lambda: model.fit(x_train_reduced, y_train_host),
        )
        probabilities = timed_stage(
            stages, "inference", mode, lambda: model.predict_proba(x_test_reduced)
        )
        probabilities_host = timed_stage(
            stages, "materialize", "cpu", lambda: np.asarray(probabilities)
        )
    elif mode == "gpu_resident":
        scaler, pca, model = gpu_components(components)
        x_train = timed_stage(stages, "host_to_device", mode, lambda: cp.asarray(x_train_host))
        x_test = timed_stage(stages, "host_to_device", mode, lambda: cp.asarray(x_test_host))
        y_train = timed_stage(stages, "host_to_device", mode, lambda: cp.asarray(y_train_host))
        transfer_bytes["host_to_device"] = x_train_host.nbytes + x_test_host.nbytes + y_train_host.nbytes
        transfer_bytes["count"] = 1
        x_train_scaled = timed_stage(stages, "scale", mode, lambda: scaler.fit_transform(x_train))
        x_test_scaled = timed_stage(stages, "scale", mode, lambda: scaler.transform(x_test))
        x_train_features = timed_stage(
            stages, "feature_engineering", mode, lambda: feature_engineering(x_train_scaled, cp)
        )
        x_test_features = timed_stage(
            stages, "feature_engineering", mode, lambda: feature_engineering(x_test_scaled, cp)
        )
        x_train_reduced = timed_stage(stages, "pca", mode, lambda: pca.fit_transform(x_train_features))
        x_test_reduced = timed_stage(stages, "pca", mode, lambda: pca.transform(x_test_features))
        timed_stage(stages, "train", mode, lambda: model.fit(x_train_reduced, y_train))
        probabilities = timed_stage(stages, "inference", mode, lambda: model.predict_proba(x_test_reduced))
        probabilities_host = timed_stage(stages, "materialize", "cpu", lambda: to_host(probabilities))
        transfer_bytes["device_to_host"] = probabilities_host.nbytes
        transfer_bytes["count"] += 1
    elif mode == "cpu_to_gpu":
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        _, pca, model = gpu_components(components)
        x_train_scaled = timed_stage(stages, "scale", "cpu", lambda: scaler.fit_transform(x_train_host))
        x_test_scaled = timed_stage(stages, "scale", "cpu", lambda: scaler.transform(x_test_host))
        x_train_features_host = timed_stage(
            stages, "feature_engineering", "cpu", lambda: feature_engineering(x_train_scaled, np)
        )
        x_test_features_host = timed_stage(
            stages, "feature_engineering", "cpu", lambda: feature_engineering(x_test_scaled, np)
        )
        x_train_features = timed_stage(
            stages, "host_to_device", mode, lambda: cp.asarray(x_train_features_host)
        )
        x_test_features = timed_stage(
            stages, "host_to_device", mode, lambda: cp.asarray(x_test_features_host)
        )
        y_train = timed_stage(stages, "host_to_device", mode, lambda: cp.asarray(y_train_host))
        transfer_bytes["host_to_device"] = x_train_features_host.nbytes + x_test_features_host.nbytes + y_train_host.nbytes
        transfer_bytes["count"] = 1
        x_train_reduced = timed_stage(stages, "pca", mode, lambda: pca.fit_transform(x_train_features))
        x_test_reduced = timed_stage(stages, "pca", mode, lambda: pca.transform(x_test_features))
        timed_stage(stages, "train", mode, lambda: model.fit(x_train_reduced, y_train))
        probabilities = timed_stage(stages, "inference", mode, lambda: model.predict_proba(x_test_reduced))
        probabilities_host = timed_stage(stages, "materialize", "cpu", lambda: to_host(probabilities))
        transfer_bytes["device_to_host"] = probabilities_host.nbytes
        transfer_bytes["count"] += 1
    elif mode == "ping_pong":
        scaler, pca, model = gpu_components(components)
        x_train = timed_stage(stages, "host_to_device", mode, lambda: cp.asarray(x_train_host))
        x_test = timed_stage(stages, "host_to_device", mode, lambda: cp.asarray(x_test_host))
        y_train = timed_stage(stages, "host_to_device", mode, lambda: cp.asarray(y_train_host))
        transfer_bytes["host_to_device"] += x_train_host.nbytes + x_test_host.nbytes + y_train_host.nbytes
        transfer_bytes["count"] += 1
        x_train_scaled = timed_stage(stages, "scale", mode, lambda: scaler.fit_transform(x_train))
        x_test_scaled = timed_stage(stages, "scale", mode, lambda: scaler.transform(x_test))
        x_train_scaled_host = timed_stage(stages, "device_to_host", "cpu", lambda: to_host(x_train_scaled))
        x_test_scaled_host = timed_stage(stages, "device_to_host", "cpu", lambda: to_host(x_test_scaled))
        transfer_bytes["device_to_host"] += x_train_scaled_host.nbytes + x_test_scaled_host.nbytes
        transfer_bytes["count"] += 1
        x_train_features_host = timed_stage(
            stages, "feature_engineering", "cpu", lambda: feature_engineering(x_train_scaled_host, np)
        )
        x_test_features_host = timed_stage(
            stages, "feature_engineering", "cpu", lambda: feature_engineering(x_test_scaled_host, np)
        )
        x_train_features = timed_stage(
            stages, "host_to_device", mode, lambda: cp.asarray(x_train_features_host)
        )
        x_test_features = timed_stage(
            stages, "host_to_device", mode, lambda: cp.asarray(x_test_features_host)
        )
        transfer_bytes["host_to_device"] += x_train_features_host.nbytes + x_test_features_host.nbytes
        transfer_bytes["count"] += 1
        x_train_reduced = timed_stage(stages, "pca", mode, lambda: pca.fit_transform(x_train_features))
        x_test_reduced = timed_stage(stages, "pca", mode, lambda: pca.transform(x_test_features))
        timed_stage(stages, "train", mode, lambda: model.fit(x_train_reduced, y_train))
        probabilities = timed_stage(stages, "inference", mode, lambda: model.predict_proba(x_test_reduced))
        probabilities_host = timed_stage(stages, "materialize", "cpu", lambda: to_host(probabilities))
        transfer_bytes["device_to_host"] += probabilities_host.nbytes
        transfer_bytes["count"] += 1
    else:
        raise ValueError(f"Unknown mode: {mode}")

    total_seconds = time.perf_counter() - total_started
    positive = probabilities_host[:, 1]
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

    predicted = (positive >= 0.5).astype(np.int8)
    return {
        "total_seconds": total_seconds,
        "stages": stages,
        "transfers": transfer_bytes,
        "quality": {
            "accuracy": float(accuracy_score(y_test_host, predicted)),
            "log_loss": float(log_loss(y_test_host, probabilities_host)),
            "roc_auc": float(roc_auc_score(y_test_host, positive)),
        },
        "test_rows": len(y_test_host),
    }


def main() -> None:
    from experiments.gpu_ml_pipeline.benchmark import environment

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--replication", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "accel":
        enable_accelerator()
    manifest_path = args.manifest.resolve()
    manifest = read_json(manifest_path)
    identity = {
        "dataset_id": manifest["dataset_id"],
        "git_commit": source_id(),
        "mode": args.mode,
        "replication": args.replication,
        "trials": args.trials,
        "warmups": args.warmups,
    }
    run_id = stable_id("gpu-ml-pipeline", identity)
    output_dir = args.output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    for _ in range(args.warmups):
        run_once(manifest_path, args.mode, args.seed, limit=4_096)
    trials = []
    for trial in range(args.trials):
        sampler = GpuSampler()
        sampler.start()
        value = run_once(manifest_path, args.mode, args.seed)
        value["trial"] = trial
        value["gpu"] = sampler.stop()
        trials.append(value)
    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT_NAME,
        "run_id": run_id,
        "git_commit": source_id(),
        "dataset": manifest,
        "condition": {
            "mode": args.mode,
            "replication": args.replication,
            "warmups": args.warmups,
            "trials": args.trials,
        },
        "environment": environment(),
        "total_median_seconds": statistics.median(x["total_seconds"] for x in trials),
        "quality": trials[-1]["quality"],
        "trials": trials,
    }
    result_path = output_dir / "result.json"
    atomic_json(result_path, result)
    print(json.dumps({"result": str(result_path), "run_id": run_id}, indent=2))


if __name__ == "__main__":
    main()
