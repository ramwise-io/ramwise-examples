# syntax=docker/dockerfile:1.10

ARG CUDA_IMAGE="nvidia/cuda:13.1.1-base-ubuntu24.04@sha256:e8c8679ccd042249d4c4080a3fab5a6bb52ab6e771addffa2e6e4eafea797bd2"
FROM ${CUDA_IMAGE}

ARG MINIFORGE_VERSION="26.3.2-2"
ARG MINIFORGE_SHA256="42260ffe3830fb953d5eee1bbb32229ff06aa7c3833c1ed7a9a0420a95685d94"

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates curl git procps tini \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSLo /tmp/miniforge.sh \
        "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh" \
    && echo "${MINIFORGE_SHA256}  /tmp/miniforge.sh" | sha256sum --check --strict \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh

COPY conda-linux-64.lock /tmp/conda-linux-64.lock
RUN /opt/conda/bin/conda create --prefix /opt/benchmark --file /tmp/conda-linux-64.lock \
    && /opt/conda/bin/conda clean --all --yes \
    && rm /tmp/conda-linux-64.lock

COPY requirements-gpu.txt /tmp/requirements-gpu.txt
RUN /opt/benchmark/bin/python -m pip install --no-deps --require-hashes -r /tmp/requirements-gpu.txt \
    && rm /tmp/requirements-gpu.txt

ENV PATH="/opt/benchmark/bin:${PATH}" CONDA_PREFIX="/opt/benchmark" PYTHONUNBUFFERED=1
WORKDIR /benchmark
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"]
