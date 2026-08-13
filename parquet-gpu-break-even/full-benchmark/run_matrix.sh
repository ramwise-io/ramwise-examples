#!/usr/bin/env bash
set -euo pipefail

mode="${1:-confirmation}"
case "${mode}" in
  confirmation) config="configs/isolated_confirmation.json" ;;
  rowgroup) config="configs/row_group_sweep.json" ;;
  *) echo "usage: $0 [confirmation|rowgroup] [matrix-runner options]" >&2; exit 2 ;;
esac
shift || true

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data_root="${PARQUET_BENCH_DATA_DIR:-${root}/benchmark-data}"
cpu_set="${PARQUET_BENCH_CPUSET:-0-7}"
image="${PARQUET_BENCH_IMAGE:-ramwise/parquet-gpu-study:26.08}"
source_id="$(git -C "${root}/../.." rev-parse HEAD 2>/dev/null || echo public-bundle)"

mkdir -p "${data_root}"
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,temperature.gpu \
  --format=csv,noheader
echo "benchmark CPU affinity: ${cpu_set}"
echo "benchmark data directory: ${data_root}"

exec docker run --rm \
  --network none \
  --cpuset-cpus="${cpu_set}" \
  --gpus all \
  --shm-size=16g \
  --mount type=bind,source="${root}",target=/benchmark,readonly \
  --mount type=bind,source="${data_root}",target=/benchmark-data \
  --env PARQUET_BENCH_DATA_ROOT=/benchmark-data \
  --env PARQUET_BENCH_HOSTNAME=benchmark-host \
  --env PARQUET_BENCH_SOURCE_ID="${source_id}" \
  --env MPLBACKEND=Agg \
  "${image}" \
  python -m experiments.parquet_decompression.matrix_runner \
  --config "/benchmark/${config}" \
  "$@"
