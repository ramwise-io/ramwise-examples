#!/usr/bin/env bash
set -euo pipefail

mode="${1:-smoke}"
shift || true
case "${mode}" in
  smoke) config="experiments/vector_search_cuvs/matrix_smoke.json" ;;
  full) config="experiments/vector_search_cuvs/matrix_full.json" ;;
  *) echo "usage: $0 [smoke|full] [--retry-failed]" >&2; exit 2 ;;
esac

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
image="ramwise/cuvs-vector-search-break-even:26.08-cuda13.1"
data_root="${CUVS_DATA_ROOT:-${root}/.data}"
mkdir -p "${data_root}/generated" "${data_root}/benchmarks" "${data_root}/cache"

docker build --tag "${image}" "${root}"
exec docker run --rm --network none --gpus all --shm-size=16g \
  --mount type=bind,source="${root}",target=/workspace,readonly \
  --mount type=bind,source="${data_root}/generated",target=/data/generated \
  --mount type=bind,source="${data_root}/benchmarks",target=/data/benchmarks \
  --mount type=bind,source="${data_root}/cache",target=/data/cache \
  --env XDG_CACHE_HOME=/data/cache \
  "${image}" python -m experiments.vector_search_cuvs.matrix_runner \
  --config "/workspace/${config}" "$@"
