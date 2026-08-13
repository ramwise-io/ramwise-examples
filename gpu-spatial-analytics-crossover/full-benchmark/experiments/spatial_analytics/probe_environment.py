"""Report spatial packages and public APIs in the canonical GPU Lab image."""

from __future__ import annotations

import importlib
import importlib.metadata


PACKAGES = (
    "cudf",
    "cuspatial",
    "cupy",
    "duckdb",
    "geopandas",
    "scipy",
    "shapely",
    "numba",
    "numba-cuda",
)


def main() -> None:
    for package in PACKAGES:
        try:
            module = importlib.import_module(package)
            version = importlib.metadata.version(package)
            print(f"{package}={version}")
            if package == "cuspatial":
                public = sorted(name for name in dir(module) if not name.startswith("_"))
                print("cuspatial_public=" + ",".join(public))
        except Exception as exc:  # pragma: no cover - diagnostic only
            print(f"{package}=UNAVAILABLE ({type(exc).__name__}: {exc})")

    try:
        from numba import cuda

        print(f"numba_cuda_available={cuda.is_available()}")
        print(f"numba_cuda_devices={cuda.list_devices()}")
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"numba_cuda_probe=FAILED ({type(exc).__name__}: {exc})")


if __name__ == "__main__":
    main()
