from __future__ import annotations

from argparse import Namespace

from experiments.gpu_ml_pipeline.common import quality_matches, stable_id, warm_file_cache
from experiments.gpu_ml_pipeline.estimators import allowed_engines, make_estimator
from experiments.gpu_ml_pipeline.generate_dataset import dataset_config, validate_config
from experiments.gpu_ml_pipeline.matrix_runner import expand_jobs, matrix_identity
from experiments.gpu_ml_pipeline.pipeline_benchmark import MODES, feature_engineering
from experiments.gpu_ml_pipeline.pipeline_summarize import break_even_reuses
from experiments.gpu_ml_pipeline.summarize import break_even_batches


def test_dataset_config_caps_informative_features() -> None:
    args = Namespace(
        rows=1000,
        features=4,
        test_fraction=0.2,
        clusters=8,
        informative_features=16,
        cluster_scale=3.0,
        seed=7,
    )
    config = dataset_config(args)
    validate_config(config)
    assert config["informative_features"] == 4


def test_xgboost_excludes_cuml_accel() -> None:
    assert allowed_engines("xgboost") == ("cpu", "native_gpu")


def test_cpu_random_forest_uses_shared_sqrt_feature_policy() -> None:
    estimator = make_estimator(
        "random_forest", "cpu", features=64, clusters=16, seed=7
    )
    assert estimator.max_features == "sqrt"


def test_expand_jobs_respects_algorithm_engine_support() -> None:
    config = {
        "benchmark": {"replications": 2},
        "datasets": [
            {
                "name": "x",
                "algorithms": ["pca", "xgboost"],
                "engines": ["cpu", "accel", "native_gpu"],
            }
        ],
    }
    jobs = expand_jobs(config)
    assert len(jobs) == 10
    assert ("x", "xgboost", "accel", 0) not in jobs


def test_quality_control_uses_algorithm_tolerance() -> None:
    assert quality_matches("logistic_regression", {"roc_auc": 0.90}, {"roc_auc": 0.89})
    assert quality_matches("logistic_regression", {"roc_auc": 0.90}, {"roc_auc": 0.99})
    assert not quality_matches("logistic_regression", {"roc_auc": 0.90}, {"roc_auc": 0.80})


def test_break_even_batches_accounts_for_fixed_and_repeated_costs() -> None:
    assert break_even_batches(2.0, 6.0, 0.010, 0.002) == 500
    assert break_even_batches(6.0, 2.0, 0.010, 0.002) == 0
    assert break_even_batches(2.0, 6.0, 0.002, 0.010) is None


def test_matrix_identity_binds_source_commit() -> None:
    config = {"name": "smoke", "datasets": [{"name": "x"}]}
    assert matrix_identity(config, "a") == matrix_identity(config, "a")
    assert matrix_identity(config, "a") != matrix_identity(config, "b")
    assert stable_id("x", {"a": 1, "b": 2}) == stable_id("x", {"b": 2, "a": 1})


def test_feature_engineering_has_deterministic_width() -> None:
    import numpy as np

    values = np.ones((5, 16), dtype=np.float32)
    transformed = feature_engineering(values, np)
    assert transformed.shape == (5, 28)
    assert transformed.dtype == np.float32


def test_pipeline_modes_cover_residency_and_transitions() -> None:
    assert MODES == ("cpu", "accel", "gpu_resident", "cpu_to_gpu", "ping_pong")


def test_break_even_reuses_handles_faster_and_slower_inference() -> None:
    assert break_even_reuses(2.0, 6.0, 0.010, 0.002) == 500
    assert break_even_reuses(6.0, 2.0, 0.010, 0.002) == 0
    assert break_even_reuses(2.0, 6.0, 0.002, 0.010) is None


def test_file_cache_warmup_reads_complete_files(tmp_path) -> None:
    path = tmp_path / "input.bin"
    path.write_bytes(b"abcdefghij")
    warm_file_cache([path], chunk_bytes=3)
