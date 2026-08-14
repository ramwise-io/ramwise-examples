from __future__ import annotations

from typing import Any

ALGORITHMS = (
    "pca",
    "kmeans",
    "logistic_regression",
    "random_forest",
    "umap",
    "hdbscan",
    "xgboost",
)
ENGINES = ("cpu", "accel", "native_gpu")
SUPERVISED = {"logistic_regression", "random_forest", "xgboost"}


def allowed_engines(algorithm: str) -> tuple[str, ...]:
    if algorithm == "xgboost":
        return ("cpu", "native_gpu")
    return ENGINES


def enable_accelerator() -> None:
    import cuml.accel

    cuml.accel.install()


def make_estimator(
    algorithm: str,
    engine: str,
    *,
    features: int,
    clusters: int,
    seed: int,
) -> Any:
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    if engine not in allowed_engines(algorithm):
        raise ValueError(f"Engine {engine} is not supported for {algorithm}")

    gpu = engine == "native_gpu"
    components = min(16, max(2, features // 2))
    if algorithm == "pca":
        if gpu:
            from cuml.decomposition import PCA

            return PCA(n_components=components, svd_solver="full", output_type="cupy")
        from sklearn.decomposition import PCA

        return PCA(n_components=components, svd_solver="full", random_state=seed)
    if algorithm == "kmeans":
        if gpu:
            from cuml.cluster import KMeans

            return KMeans(
                n_clusters=clusters,
                init="k-means++",
                n_init=1,
                max_iter=100,
                random_state=seed,
                output_type="cupy",
            )
        from sklearn.cluster import KMeans

        return KMeans(
            n_clusters=clusters,
            init="k-means++",
            n_init=1,
            max_iter=100,
            random_state=seed,
        )
    if algorithm == "logistic_regression":
        if gpu:
            from cuml.linear_model import LogisticRegression

            return LogisticRegression(
                C=1.0, max_iter=200, solver="qn", output_type="cupy"
            )
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(C=1.0, max_iter=200, solver="lbfgs")
    if algorithm == "random_forest":
        if gpu:
            from cuml.ensemble import RandomForestClassifier

            return RandomForestClassifier(
                n_estimators=100,
                max_depth=16,
                max_features="sqrt",
                n_streams=1,
                random_state=seed,
                output_type="cupy",
            )
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=100,
            max_depth=16,
            max_features="sqrt",
            n_jobs=8,
            random_state=seed,
        )
    if algorithm == "umap":
        if gpu:
            from cuml.manifold import UMAP

            return UMAP(
                n_components=2,
                n_neighbors=15,
                min_dist=0.1,
                metric="euclidean",
                init="random",
                random_state=seed,
                output_type="cupy",
            )
        import umap

        return umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric="euclidean",
            init="random",
            random_state=seed,
        )
    if algorithm == "hdbscan":
        if gpu:
            from cuml.cluster import HDBSCAN

            return HDBSCAN(
                min_cluster_size=50,
                min_samples=10,
                metric="euclidean",
                output_type="cupy",
            )
        import hdbscan

        return hdbscan.HDBSCAN(
            min_cluster_size=50,
            min_samples=10,
            metric="euclidean",
        )
    if algorithm == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=1.0,
            tree_method="hist",
            device="cuda" if gpu else "cpu",
            n_jobs=8,
            random_state=seed,
            eval_metric="logloss",
        )
    raise AssertionError("unreachable")
